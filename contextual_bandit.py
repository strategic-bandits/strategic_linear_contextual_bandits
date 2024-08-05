import numpy as np


class ContextualBandit:
    """
    Strategic Linear Contextual Bandit

    """

    def __init__(
            self,
            K=None,
            T=None,
            true_arm_features=None,
            reported_arm_features=None,
            user_contexts=None,
            theta=None,
            d=None
    ):
        self.K = K
        self.T = T
        self.true_arm_features = true_arm_features
        self.user_contexts = user_contexts
        self.theta = theta
        self.d = d
        self.arm_utilities = np.zeros(K)
        self.reported_arm_features = reported_arm_features  # Kxd entries

        # Define sequence of arm contexts and mean rewards
        self.arm_contexts = np.zeros((T, K, d))  # arm_contexts[time, arm, context element]
        self.reported_arm_contexts = np.zeros((T, K, d))
        self.mean_rewards = np.zeros((T, K))  # mean_rewards[time, arm]
        self.reported_mean_rewards = np.zeros((T, K))
        for t in range(self.T):
            for i in range(self.K):
                self.arm_contexts[t, i, :] = np.multiply(self.user_contexts[t], self.true_arm_features[i])
                self.reported_arm_contexts[t, i, :] = np.multiply(self.user_contexts[t], self.reported_arm_features[i])
                self.mean_rewards[t, i] = np.dot(self.arm_contexts[t, i, :], self.theta)
                self.reported_mean_rewards[t, i] = np.dot(self.reported_arm_contexts[t, i, :], self.theta)
        self.optimal_return = []
        for t in range(self.T):
            self.optimal_return.append(np.sum(np.amax(self.mean_rewards[0:t], axis=1)))
        # self.optimal_return = np.sum(np.amax(self.mean_rewards[0:t], axis=1))

        # report manipulation in terms of feature and reward corruption over the course of iterations
        self.total_feature_corruption = []
        self.total_reward_corruption = []

    def round(self, t, a):
        self.arm_utilities[a] += 1
        reward = self.mean_rewards[t, a] + np.random.normal(0, .5)
        return self.mean_rewards[t, a], reward, np.max(self.mean_rewards[t, :]) - self.mean_rewards[t, a]

    # set the arm features of arm i and update arm_contexts
    def set_reported_features(self, i, features):
        self.reported_arm_features[i] = features
        # update reported contexts
        for t in range(self.T):
            for i in range(self.K):
                self.reported_arm_contexts[t, i, :] = np.multiply(self.user_contexts[t], self.reported_arm_features[i])
                self.reported_mean_rewards[t, i] = np.dot(self.reported_arm_contexts[t, i, :], self.theta)

    def reset_arm_utilities(self):
        self.arm_utilities = np.zeros(self.K)

    def feature_gradient_step(self, grad):
        self.reported_arm_features = self.reported_arm_features + grad
        # update reported contexts and reported means
        for t in range(self.T):
            for i in range(self.K):
                self.reported_arm_contexts[t, i, :] = np.multiply(self.user_contexts[t], self.reported_arm_features[i])
                self.reported_mean_rewards[t, i] = np.dot(self.reported_arm_contexts[t, i, :], self.theta)

    # return the current total feature corruption
    def get_total_feature_corruption(self):
        return np.linalg.norm(self.arm_contexts - self.reported_arm_contexts)

    # return the current total corruption in terms of rewards w.r.t. reported features
    def get_total_reward_corruption(self):
        return np.linalg.norm(self.mean_rewards - self.reported_mean_rewards)

    def get_arm_utilities(self):
        return self.arm_utilities