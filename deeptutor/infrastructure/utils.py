import copy

import numpy as np

item_difficulty_mean = 1
item_difficulty_std = 1

log_item_decay_exp_mean = 1
log_item_decay_exp_std = 1

log_delay_coef_mean = 0
log_delay_coef_std = 0.01


def sample_const_delay(d):
    return lambda: d


def sample_item_difficulties(n_items):
    return np.random.normal(item_difficulty_mean, item_difficulty_std, n_items)


def sample_student_ability():
    return 0


def sample_window_cw(n_windows):
    x = 1 / (np.arange(1, n_windows + 1, 1)) ** 2
    return x[::-1]


def sample_window_nw(n_windows):
    x = 1 / (np.arange(1, n_windows + 1, 1)) ** 2
    return x[::-1]


def sample_item_decay_exps(n_items):
    return np.exp(
        np.random.normal(log_item_decay_exp_mean, log_item_decay_exp_std, n_items)
    )


def sample_student_decay_exp():
    return 0


def sample_delay_coef():
    return np.exp(np.random.normal(log_delay_coef_mean, log_delay_coef_std))


def sample_item_decay_rates(n_items):
    return np.exp(np.random.normal(np.log(0.077), 1, n_items))


def sample_loglinear_coeffs(n_items):
    coeffs = np.array([1, 1, 0])
    coeffs = np.concatenate((coeffs, np.random.normal(0, 1, n_items)))
    return coeffs


def normalize(x):
    return x / x.sum()


def make_rl_student_env(env):
    env = copy.deepcopy(env)

    env.n_item_feats = int(np.log(2 * env.n_items))
    env.item_feats = np.random.normal(
        np.zeros(2 * env.n_items * env.n_item_feats),
        np.ones(2 * env.n_items * env.n_item_feats),
    ).reshape((2 * env.n_items, env.n_item_feats))
    env.observation_space = spaces.Box(
        np.concatenate((np.ones(env.n_item_feats) * -sys.maxsize, np.zeros(3))),
        np.concatenate((np.ones(env.n_item_feats + 2) * sys.maxsize, np.ones(1))),
    )

    def encode_item(self, item, outcome):
        return self.item_feats[self.n_items * outcome + item, :]

    def encode_delay(self, delay, outcome):
        v = np.zeros(2)
        v[outcome] = np.log(1 + delay)
        return v

    def vectorize_obs(self, item, outcome, delay):
        return np.concatenate(
            (
                self.encode_item(item, outcome),
                self.encode_delay(delay, outcome),
                np.array([outcome]),
            )
        )

    env._obs_orig = env._obs

    def _obs(self):
        item, outcome, timestamp, delay = env._obs_orig()
        return self.vectorize_obs(item, outcome, delay)

    env.encode_item = types.MethodType(encode_item, env)
    env.encode_delay = types.MethodType(encode_delay, env)
    env.vectorize_obs = types.MethodType(vectorize_obs, env)
    env._obs = types.MethodType(_obs, env)

    return env


def run_ep(agent, env):
    agent.reset()
    obs = env._reset()
    done = False
    totalr = []
    observations = []
    while not done:
        action = agent.act(obs)
        obs, r, done, _ = env._step(action)
        agent.learn(r)
        totalr.append(r)
        observations.append(obs)
    return np.mean(totalr), observations


def run_eps(agent, env, n_eps=100):
    tot_rew = []
    for i in range(n_eps):
        totalr, _ = run_ep(agent, env)
        tot_rew.append(totalr)
    return tot_rew
