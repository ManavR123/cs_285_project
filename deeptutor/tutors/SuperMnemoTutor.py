#
# based on SM2_mnemosyne.py <Peter.Bienstman@UGent.be>
#
import random

import numpy as np

from deeptutor.tutors.Card import Card
from deeptutor.tutors.Tutor import Tutor

HOUR = 60 * 60  # Seconds in an hour.
DAY = 24 * HOUR  # Seconds in a day.


class SuperMnemoTutor(Tutor):

    """Scheduler based on http://www.supermemo.com/english/ol/sm2.htm.
    Note that all intervals are in seconds, since time is stored as
    integer POSIX timestamps.

    Since the scheduling granularity is days, all cards due on the same time
    should become due at the same time. In order to keep the SQL query
    efficient, we do this by setting 'next_rep' the same for all cards that
    are due on the same day.

    In order to allow for the fact that the timezone and 'day_starts_at' can
    change after scheduling a card, we store 'next_rep' as midnight UTC, and
    bring local time and 'day_starts_at' only into play when querying the
    database.

    """

    def __init__(
        self,
        n_items,
        init_timestamp=0,
        non_memorised_cards_in_hand=10,
        fail_grade=0,
        pass_grade=2,
    ):
        self.non_memorised_cards_in_hand = non_memorised_cards_in_hand
        self.fail_grade = fail_grade
        self.pass_grade = pass_grade
        self.state = 1
        self.curr_step = 0
        self.card_of_id = [Card(i) for i in range(n_items)]
        self.curr_item = None
        self.now = init_timestamp
        self.unseen = set(range(n_items))
        self._card_ids_memorised = []
        self.reset()
        self.n_items = n_items

    def true_scheduled_interval(self, card):

        """Since 'next_rep' is always midnight UTC for retention reps, we need
        to take timezone and 'day_starts_at' into account to calculate the
        true scheduled interval when we are doing the actual repetition.
        This basically undoes the operations from 'adjusted_now'.
        Note that during the transition between different timezones, this is
        not well-defined, but the influence on the scheduler will be minor
        anyhow.

        """

        interval = card.next_rep - card.last_rep
        if card.grade < 2:
            return interval
        interval += HOUR
        return int(interval)

    def reset(self, new_only=False):

        """'_card_ids_in_queue' contains the _ids of the cards making up the
        queue.

        The corresponding fact._ids are also stored in '_fact_ids_in_queue',
        which is needed to make sure that no sister cards can be together in
        the queue at any time.

        '_fact_ids_memorised' has a different function and persists over the
        different stages invocations of 'rebuild_queue'. It can be used to
        control whether or not memorising a card will prevent a sister card
        from being pulled out of the 'unseen' pile, even after the queue has
        been rebuilt.

        '_card_id_last' is stored to avoid showing the same card twice in a
        row.

        'stage' stores the stage of the queue building, and is used to skip
        over unnecessary queries.

        """

        self._card_ids_in_queue = []
        self._card_id_last = None
        self.new_only = new_only
        if self.new_only == False:
            self.stage = 1
        else:
            self.stage = 3

    def set_initial_grade(self, cards, grade):

        """Sets the initial grades for a set of sister cards, making sure
        their next repetitions do no fall on the same day.

        Note that even if the initial grading happens when adding a card, it
        is seen as a repetition.

        """

        new_interval = self.calculate_initial_interval(grade)
        new_interval += self.calculate_interval_noise(new_interval)
        last_rep = self.now
        next_rep = last_rep + new_interval
        for card in cards:
            card.grade = grade
            card.easiness = 2.5
            card.acq_reps = 1
            card.acq_reps_since_lapse = 1
            card.last_rep = last_rep
            card.next_rep = next_rep
            next_rep += DAY

    def calculate_initial_interval(self, grade):

        """The first repetition is treated specially, and gives longer
        intervals, to allow for the fact that the user may have seen this
        card before.

        """

        return (0, 0, 1 * DAY, 3 * DAY, 4 * DAY, 7 * DAY)[grade]

    def calculate_interval_noise(self, interval):
        if interval == 0:
            noise = 0
        elif interval <= 10 * DAY:
            noise = random.choice([0, DAY])
        elif interval <= 60 * DAY:
            noise = random.uniform(-3 * DAY, 3 * DAY)
        else:
            noise = random.uniform(-0.05 * interval, 0.05 * interval)
        return int(noise)

    def cards_due_for_ret_rep(self):
        return sorted(
            range(self.n_items),
            key=lambda i: self.card_of_id[i].next_rep - self.card_of_id[i].last_rep,
            reverse=True,
        )

    def cards_to_relearn(self, grade=0):
        # TODO: only return cards incorrectly answered in stage 1
        return [
            i
            for i in range(self.n_items)
            if self.card_of_id[i].grade == grade and i not in self.unseen
        ]

    def cards_new_memorising(self, grade=0):
        return [
            i
            for i in range(self.n_items)
            if self.card_of_id[i].grade == grade and i not in self.unseen
        ]

    def cards_unseen(self, limit=50):
        return (
            random.sample(self.unseen, limit)
            if limit < len(self.unseen)
            else self.unseen
        )

    def card(self, card_id):
        return self.card_of_id[card_id]

    def interval_multiplication_factor(self, *args):
        return 1

    def rebuild_queue(self, learn_ahead=False):
        self._card_ids_in_queue = []

        # Stage 1
        #
        # Do the cards that are scheduled for today (or are overdue), but
        # first do those that have the shortest interval, as being a day
        # late on an interval of 2 could be much worse than being a day late
        # on an interval of 50.
        # Fetch maximum 50 cards at the same time, as a trade-off between
        # memory usage and redoing the query.
        if self.stage == 1:
            for _card_id in self.cards_due_for_ret_rep():
                self._card_ids_in_queue.append(_card_id)
            if len(self._card_ids_in_queue):
                return
            self.stage = 2

        # Stage 2
        #
        # Now rememorise the cards that we got wrong during the last stage.
        # Concentrate on only a limited number of non memorised cards, in
        # order to avoid too long intervals between repetitions.
        limit = self.non_memorised_cards_in_hand
        non_memorised_in_queue = 0
        if self.stage == 2:
            for _card_id in self.cards_to_relearn(grade=1):
                if _card_id not in self._card_ids_in_queue:
                    if non_memorised_in_queue < limit:
                        self._card_ids_in_queue.append(_card_id)
                        non_memorised_in_queue += 1
                    if non_memorised_in_queue >= limit:
                        break
            for _card_id in self.cards_to_relearn(grade=0):
                if _card_id not in self._card_ids_in_queue:
                    if non_memorised_in_queue < limit:
                        self._card_ids_in_queue.append(_card_id)
                        self._card_ids_in_queue.append(_card_id)
                        non_memorised_in_queue += 1
                    if non_memorised_in_queue >= limit:
                        break
            random.shuffle(self._card_ids_in_queue)
            # Only stop when we reach the non memorised limit. Otherwise, keep
            # going to add some extra cards to get more spread.
            if non_memorised_in_queue >= limit:
                return
            # If the queue is empty, we can skip stage 2 in the future.
            if len(self._card_ids_in_queue) == 0:
                self.stage = 3

        # Stage 3
        #
        # Now do the cards which have never been committed to long-term
        # memory, but which we have seen before.
        # Use <= in the stage check, such that earlier stages can use
        # cards from this stage to increase the hand.
        if self.stage <= 3:
            for _card_id in self.cards_new_memorising(grade=1):
                if _card_id not in self._card_ids_in_queue:
                    if non_memorised_in_queue < limit:
                        self._card_ids_in_queue.append(_card_id)
                        non_memorised_in_queue += 1
                    if non_memorised_in_queue >= limit:
                        break
            for _card_id in self.cards_new_memorising(grade=0):
                if _card_id not in self._card_ids_in_queue:
                    if non_memorised_in_queue < limit:
                        self._card_ids_in_queue.append(_card_id)
                        self._card_ids_in_queue.append(_card_id)
                        non_memorised_in_queue += 1
                    if non_memorised_in_queue >= limit:
                        break
            random.shuffle(self._card_ids_in_queue)
            # Only stop when we reach the grade 0 limit. Otherwise, keep
            # going to add some extra cards to get more spread.
            if non_memorised_in_queue >= limit:
                return
            # If the queue is empty, we can skip stage 3 in the future.
            if len(self._card_ids_in_queue) == 0:
                self.stage = 4

        # Stage 4
        #
        # Now add some cards we have yet to see for the first time.
        # Use <= in the stage check, such that earlier stages can use
        # cards from this stage to increase the hand.
        if self.stage <= 4:
            # Preferentially keep away from sister cards for as long as
            # possible.
            for _card_id in self.cards_unseen(limit=min(limit, 50)):
                if (
                    _card_id not in self._card_ids_in_queue
                    and _card_id not in self._card_ids_memorised
                ):
                    self._card_ids_in_queue.append(_card_id)
                    non_memorised_in_queue += 1
                    if non_memorised_in_queue >= limit:
                        if self.new_only == False:
                            self.stage = 2
                        else:
                            self.stage = 3
                        return
            # If the queue is close to empty, start pulling in sister cards.
            if len(self._card_ids_in_queue) <= 2:
                for _card_id in self.cards_unseen(limit=min(limit, 50)):
                    if _card_id not in self._card_ids_in_queue:
                        self._card_ids_in_queue.append(_card_id)
                        non_memorised_in_queue += 1
                        if non_memorised_in_queue >= limit:
                            if self.new_only == False:
                                self.stage = 2
                            else:
                                self.stage = 3
                            return
            # If the queue is still empty, go to learn ahead of schedule.
            if len(self._card_ids_in_queue) == 0:
                self.stage = 5

        # Stage 5
        #
        # If we get to here, there are no more scheduled cards or new cards
        # to learn. The user can signal that he wants to learn ahead by
        # calling rebuild_queue with 'learn_ahead' set to True.
        # Don't shuffle this queue, as it's more useful to review the
        # earliest scheduled cards first. We only put 50 cards at the same
        # time into the queue, in order to save memory.
        if self.new_only == False:
            self.stage = 2
        else:
            self.stage = 3

    def next_card(self, learn_ahead=False):
        # Populate queue if it is empty, and pop first card from the queue.
        if len(self._card_ids_in_queue) == 0:
            self.rebuild_queue(learn_ahead)
            if len(self._card_ids_in_queue) == 0:
                return None
        _card_id = self._card_ids_in_queue.pop(0)
        # Make sure we don't show the same card twice in succession.
        if self._card_id_last:
            while _card_id == self._card_id_last:
                # Make sure we have enough cards to vary, but exit in hopeless
                # situations.
                if len(self._card_ids_in_queue) == 0:
                    self.rebuild_queue(learn_ahead)
                    if len(self._card_ids_in_queue) == 0:
                        return None
                    if set(self._card_ids_in_queue) == set([_card_id]):
                        return db.card(_card_id, is_id_internal=True)
                _card_id = self._card_ids_in_queue.pop(0)
        self._card_id_last = _card_id
        return self.card(_card_id)

    def _next_item(self):
        if self.curr_item is not None:
            raise ValueError

        card = self.next_card()
        if card is None:
            raise ValueError
        self.curr_item = card._id
        return self.curr_item

    def grade_answer(self, card, new_grade, dry_run=False):
        # When doing a dry run, make a copy to operate on. This leaves the
        # original in the GUI intact.
        if dry_run:
            card = copy.copy(card)
        # Determine whether we learned on time or not (only relevant for
        # grades 2 or higher).
        if self.now - DAY >= card.next_rep:  # Already due yesterday.
            timing = "LATE"
        else:
            if self.now < card.next_rep:  # Not due today.
                timing = "EARLY"
            else:
                timing = "ON TIME"
        # Calculate the previously scheduled interval, i.e. the interval that
        # led up to this repetition.
        scheduled_interval = self.true_scheduled_interval(card)
        # If we memorise a card, keep track of its fact, so that we can avoid
        # pulling a sister card from the 'unseen' pile.
        if not dry_run and card.grade < 2 and new_grade >= 2:
            self._card_ids_memorised.append(card._id)
        if card.grade == -1:  # Unseen card.
            actual_interval = 0
        else:
            actual_interval = self.now - card.last_rep
        if card.grade == -1:
            # The card has not yet been given its initial grade.
            card.easiness = 2.5
            card.acq_reps = 1
            card.acq_reps_since_lapse = 1
            new_interval = self.calculate_initial_interval(new_grade)
        elif card.grade in [0, 1] and new_grade in [0, 1]:
            # In the acquisition phase and staying there.
            card.acq_reps += 1
            card.acq_reps_since_lapse += 1
            new_interval = 0
        elif card.grade in [0, 1] and new_grade in [2, 3, 4, 5]:
            # In the acquisition phase and moving to the retention phase.
            card.acq_reps += 1
            card.acq_reps_since_lapse += 1
            if new_grade == 2:
                new_interval = DAY
            elif new_grade == 3:
                new_interval = random.choice([1, 1, 2]) * DAY
            elif new_grade == 4:
                new_interval = random.choice([1, 2, 2]) * DAY
            elif new_grade == 5:
                new_interval = 2 * DAY
            # Make sure the second copy of a grade 0 card doesn't show
            # up again.
            if not dry_run and card.grade == 0:
                if card._id in self._card_ids_in_queue:
                    self._card_ids_in_queue.remove(card._id)
        elif card.grade in [2, 3, 4, 5] and new_grade in [0, 1]:
            # In the retention phase and dropping back to the
            # acquisition phase.
            card.ret_reps += 1
            card.lapses += 1
            card.acq_reps_since_lapse = 0
            card.ret_reps_since_lapse = 0
            new_interval = 0
        elif card.grade in [2, 3, 4, 5] and new_grade in [2, 3, 4, 5]:
            # In the retention phase and staying there.
            card.ret_reps += 1
            card.ret_reps_since_lapse += 1
            # Don't update the easiness when learning ahead.
            if timing in ["LATE", "ON TIME"]:
                if new_grade == 2:
                    card.easiness -= 0.16
                if new_grade == 3:
                    card.easiness -= 0.14
                if new_grade == 5:
                    card.easiness += 0.10
                if card.easiness < 1.3:
                    card.easiness = 1.3
            if card.ret_reps_since_lapse == 1:
                new_interval = 6 * DAY
            else:
                if new_grade == 2 or new_grade == 3:
                    if timing in ["ON TIME", "EARLY"]:
                        new_interval = actual_interval * card.easiness
                    else:
                        # Learning late and interval was too long, so don't
                        # increase the interval and use scheduled_interval
                        # again as opposed to the much larger
                        # actual_interval * card.easiness.
                        new_interval = scheduled_interval
                if new_grade == 4:
                    new_interval = actual_interval * card.easiness
                if new_grade == 5:
                    if timing in ["EARLY"]:
                        # Learning ahead and interval was too short. To avoid
                        # that the intervals increase explosively when learning
                        # ahead, take scheduled_interval as opposed to the
                        # much larger actual_interval * card.easiness.
                        new_interval = scheduled_interval
                    else:
                        new_interval = actual_interval * card.easiness
                # Pathological case which can occur when learning ahead a card
                # in a single card database many times on the same day, such
                # that actual_interval becomes 0.
                if new_interval < DAY:
                    new_interval = DAY
        # Allow plugins to modify new_interval by multiplying it.
        new_interval *= self.interval_multiplication_factor(card, new_interval)
        new_interval = int(new_interval)
        # When doing a dry run, stop here and return the scheduled interval.
        if dry_run:
            return new_interval
        # Add some randomness to interval.
        new_interval += self.calculate_interval_noise(new_interval)
        # Update card properties. 'last_rep' is the time the card was graded,
        # not when it was shown.
        card.grade = new_grade
        card.last_rep = self.now
        if new_grade >= 2:
            card.next_rep = card.last_rep + new_interval
        else:
            card.next_rep = card.last_rep

        return new_interval

    def _update(self, item, outcome, timestamp, delay):
        if self.curr_step > 0 and (self.curr_item is None or item != self.curr_item):
            raise ValueError

        self.now = timestamp
        try:
            self.unseen.remove(item)
        except KeyError:
            pass
        self.grade_answer(
            self.card(item), (self.fail_grade, self.pass_grade)[int(outcome)]
        )

        self.curr_item = None
