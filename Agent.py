import pandas as pd
import numpy as np


class Q_Learning:
    def __init__(self, action_n, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9, total_step=100, greedy_decay=2):
        self.action_n = action_n
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.greedy_decay = greedy_decay
        self.total_step = total_step
        self.q_tabel = pd.DataFrame(columns=list(range(self.action_n)), dtype=np.float64)

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        self.check_state(s_)
        # if self.q_tabel.loc[s_, :].max() != 0:
        q_target = r + self.reward_decay * self.q_tabel.loc[s_, :].max() - self.q_tabel.loc[s, a]
        self.q_tabel.loc[s, a] = self.lr * (q_target + self.q_tabel.loc[s, a])

    def check_e_greedy(self, n_step):
        return self.e_greedy * (1 - pow((n_step / self.total_step), self.greedy_decay))

    def choose_action(self, obs, n_step):
        obs = str(obs)
        self.check_state(obs)
        greedy = self.check_e_greedy(n_step)
        if np.random.uniform(0, 1) < greedy:
            act = np.random.randint(self.action_n)
        else:
            act = self.q_tabel.loc[obs, :].argmax()
        return act

    def choose_action_val(self, obs):
        return self.q_tabel.loc[obs, :].argmax()

    def check_state(self, obs):
        if obs not in self.q_tabel.index:
            self.q_tabel = self.q_tabel.append(
                pd.Series([0] * self.action_n, name=obs, index=self.q_tabel.columns))


class Sarsa:
    def __init__(self, action_n, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9, total_step=100, greedy_decay=2):
        self.action_n = action_n
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.greedy_decay = greedy_decay
        self.total_step = total_step
        self.q_tabel = pd.DataFrame(columns=list(range(self.action_n)), dtype=np.float64)

    def learn(self, s, a, r, s_, a_):
        s = str(s)
        s_ = str(s_)
        self.check_state(s_)
        # if self.q_tabel.loc[s_, :].max() != 0:
        q_target = r + self.reward_decay * self.q_tabel.loc[s_, a_] - self.q_tabel.loc[s, a]
        self.q_tabel.loc[s, a] = self.lr * (q_target + self.q_tabel.loc[s, a])

    def check_e_greedy(self, n_step):
        return self.e_greedy * (1 - pow((n_step / self.total_step), self.greedy_decay))

    def choose_action(self, obs, n_step):
        obs = str(obs)
        self.check_state(obs)
        greedy = self.check_e_greedy(n_step)
        if np.random.uniform(0, 1) < greedy:
            act = np.random.randint(self.action_n)
        else:
            act = self.q_tabel.loc[obs, :].argmax()
        return act

    def choose_action_val(self, obs):
        return self.q_tabel.loc[obs, :].argmax()

    def check_state(self, obs):
        if obs not in self.q_tabel.index:
            self.q_tabel = self.q_tabel.append(
                pd.Series([0] * self.action_n, name=obs, index=self.q_tabel.columns))


class DQN:
    import tensorflow as tf

    def __init__(self, n_actions, n_features, learning_rate=0.02, reward_decay=0.95, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32):
        self.n_action = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

    def store_transition(self):
        pass
    def learn(self):
        pass
    def choose_action(self, obs):
        pass
