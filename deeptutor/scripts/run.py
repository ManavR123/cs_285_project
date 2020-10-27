from __future__ import division

import copy
import os
import pickle
import random
import sys
import types
from queue import Queue

import gym
import numpy as np
import tensorflow as tf
from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from gym import spaces

from deeptutor.envs.DashEnv import *
from deeptutor.envs.EFCEnv import EFCEnv
from deeptutor.envs.HRLEnv import *
from deeptutor.infrastructure.utils import *
from deeptutor.tutors.LeitnerTutor import LeitnerTutor
from deeptutor.tutors.RandTutor import RandTutor
from deeptutor.tutors.RLTutor import RLTutor
from deeptutor.tutors.SuperMnemoTutor import SuperMnemoTutor
from deeptutor.tutors.ThresholdTutor import ThresholdTutor


def main():
    tf.compat.v1.disable_eager_execution()

    data_dir = os.path.join(".", "data")

    n_steps = 200
    n_items = 30
    const_delay = 5
    discount = 0.99

    n_reps = 10
    n_eps = 100

    env_kwargs = {
        "n_items": n_items,
        "n_steps": n_steps,
        "discount": discount,
        "sample_delay": sample_const_delay(const_delay),
    }

    reward_funcs = ["likelihood", "log_likelihood"]
    envs = [("EFC", EFCEnv), ("HLR", HLREnv), ("DASH", DASHEnv)]

    tutor_builders = [
        ("Random", RandTutor),
        ("Leitner", LeitnerTutor),
        ("SuperMnemo", SuperMnemoTutor),
        ("Threshold", ThresholdTutor),
        ("RL", RLTutor),
    ]

    R = np.zeros((len(envs) * len(reward_funcs), len(tutor_builders), n_eps, n_reps))
    for h, (base_env_name, base_env) in enumerate(envs):
        for m, reward_func in enumerate(reward_funcs):
            k = h * len(reward_funcs) + m
            env_name = (
                base_env_name + "-" + ("L" if reward_func == "likelihood" else "LL")
            )
            for j in range(n_reps):
                env = base_env(**env_kwargs, reward_func=reward_func)
                rl_env = make_rl_student_env(env)
                for i, (tutor_name, build_tutor) in enumerate(tutor_builders):
                    if tutor_name.startswith("RL"):
                        agent = build_tutor(n_items)
                        R[k, i, :, j] = agent.train(rl_env, n_eps=n_eps)
                    else:
                        if "Thresh" in tutor_name:
                            agent = build_tutor(n_items, env=env)
                        else:
                            agent = build_tutor(n_items)
                        R[k, i, :, j] = agent.train(env, n_eps=n_eps)
                    print(env_name, j, tutor_name, np.mean(R[k, i, :, j]))
                print()

    reward_logs = {
        "n_steps": n_steps,
        "n_items": n_items,
        "discount": discount,
        "const_delay": const_delay,
        "n_reps": n_reps,
        "n_eps": n_eps,
        "env_names": list(zip(*envs))[0],
        "tutor_names": list(zip(*tutor_builders))[0],
        "reward_funcs": reward_funcs,
        "rewards": R,
    }
    with open(os.path.join(d, "reward_logs.pkl"), "wb") as f:
        pickle.dump(reward_logs, f, pickle.HIGHEST_PROTOCOL)

    R = np.zeros((6, len(tutor_builders), n_eps, n_reps))
    for i in range(6):
        with open(os.path.join(data_dir, "reward_logs.pkl.%d" % i), "rb") as f:
            reward_logs = pickle.load(f)
            R[i] = reward_logs["rewards"]
            print(reward_logs["env_names"], reward_logs["reward_funcs"])

    for i in range(R.shape[0]):
        for j in range(R.shape[3]):
            R[i, :, :, j] = 100 * (R[i, :, :, j] - R[i, 0, :, j]) / abs(R[i, 0, :, j])

    def moving_avg(d, n=5):
        s = np.concatenate((np.zeros(1), np.cumsum(d).astype(float)))
        return (s[n:] - s[:-n]) / n

    r_means = lambda x: np.nanmean(x, axis=1)
    r_stderrs = lambda x: np.nanstd(x, axis=1) / np.sqrt(np.count_nonzero(x, axis=1))
    r_mins = lambda x: r_means(x) - r_stderrs(x)  # np.nanmin(x, axis=1)
    r_maxs = lambda x: r_means(x) + r_stderrs(x)  # np.nanmax(x, axis=1)

    for h, (env_name, _) in enumerate(envs):
        for m, reward_func in enumerate(reward_funcs):
            k = h * len(reward_funcs) + m
            for i, (tutor_name, _) in enumerate(tutor_builders):
                print(
                    env_name,
                    reward_func,
                    tutor_name,
                    np.nanmean(R[k, i, :, :]),
                    np.nanstd(R[k, i, :, :]),
                )


if __name__ == "__main__":
    main()
