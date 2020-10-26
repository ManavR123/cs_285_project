class LeitnerTutor(Tutor):
    """sample item from an infinite leitner queue network"""

    def __init__(self, n_items, init_timestamp=0, arrival_prob=0.1):
        self.arrival_prob = arrival_prob
        self.n_items = n_items
        self.queues = None
        self.curr_q = None
        self.curr_item = None
        self.just_reset = False
        self.reset()

    def _next_item(self):
        if self.curr_item is not None:
            raise ValueError

        n_queues = len(self.queues)
        q_sampling_rates = 1 / np.sqrt(np.arange(1, n_queues, 1))
        q_sampling_rates = np.array(
            [
                x if not self.queues[i + 1].empty() else 0
                for i, x in enumerate(q_sampling_rates)
            ]
        )
        arrival_prob = self.arrival_prob if not self.queues[0].empty() else 0
        q_sampling_rates = np.concatenate(
            (np.array([arrival_prob]), normalize(q_sampling_rates) * (1 - arrival_prob))
        )
        p = normalize(q_sampling_rates)

        if self.queues[0].qsize() == self.n_items:  # no items have been shown yet
            self.curr_q = 0
        else:
            self.curr_q = np.random.choice(range(n_queues), p=p)
        self.curr_item = self.queues[self.curr_q].get(False)
        return self.curr_item

    def _update(self, item, outcome, timestamp, delay):
        if not self.just_reset and (self.curr_item is None or item != self.curr_item):
            raise ValueError

        if self.just_reset:
            for i in range(self.n_items):
                if i != item:
                    self.queues[0].put(i)

        next_q = max(1, self.curr_q + 2 * int(outcome) - 1)
        if next_q == len(self.queues):
            self.queues.append(Queue())
        self.queues[next_q].put(item)

        self.curr_item = None
        self.curr_q = None
        self.just_reset = False

    def reset(self):
        self.queues = [Queue()]

        self.curr_item = None
        self.curr_q = 0
        self.just_reset = True

    def train(self, gym_env, n_eps=10):
        arrival_probs = np.arange(0, 1, 0.01)
        n_eps_per_aprob = n_eps // arrival_probs.size
        assert n_eps_per_aprob > 0
        best_reward = None
        best_aprob = None
        for aprob in arrival_probs:
            self.arrival_prob = aprob
            reward = np.mean(run_eps(self, env, n_eps=n_eps_per_aprob))
            if best_reward is None or reward > best_reward:
                best_aprob = aprob
                best_reward = reward
        self.arrival_prob = best_aprob
        return run_eps(self, env, n_eps=n_eps)
