import numpy as np
from scipy import linalg

class LinUCB:
    """
    Vanilla LinUCB

    """

    def __init__(
            self,
            K=None,
            T=None,
            reported_contexts=None,
            d=None
    ):
        self.K = K
        self.T = T
        self.reported_contexts = reported_contexts
        self.d = d
        self.lam = 1
        self.total_return = 0
        self.cumulative_return = [0]

        # initialize matrices for estimation
        self.V = self.lam * np.identity(self.d)
        self.theta = np.zeros(self.d)
        self.B = np.zeros(self.d)
        self.beta = np.sqrt(self.lam) * self.d + np.sqrt(2 * np.log(self.T) + self.d * np.log((self.d+self.T*self.d**2) / self.d * self.lam))

    # update estimates given time step, action, reported context and observed reward
    def update(self, t, a, reward, mean):
        self.V = self.V + np.outer(self.reported_contexts[t, a], self.reported_contexts[t, a])
        self.B = self.B + self.reported_contexts[t, a] * reward
        self.theta = np.dot(linalg.inv(self.V), self.B)
        self.total_return += mean
        self.cumulative_return.append(self.cumulative_return[t] + mean)

    def play(self, t):
        if t < 10:
            return np.random.choice(self.K)
        ucb = np.zeros(self.K)
        for i in range(self.K):
            ucb[i] = np.dot(self.theta, self.reported_contexts[t, i]) + self.beta * self.norm_bonus(self.reported_contexts[t, i])
        return np.argmax(ucb)

    # outputs the norm of context vector w.r.t. V^-1
    def norm_bonus(self, context):
        squared = np.dot(np.dot(context, linalg.inv(self.V)), context)
        return np.sqrt(squared)

    def set_reported_contexts(self, contexts):
        self.reported_contexts = contexts

    def reset_return(self):
        self.total_return = 0
        self.cumulative_return = [0]

    def reset_estimates(self):
        self.V = self.lam * np.identity(self.d)
        self.theta = np.zeros(self.d)
        self.B = np.zeros(self.d)

    def get_theta_estimate(self):
        return self.theta

    def get_theta_error(self, true_theta):
        return np.linalg.norm(self.theta - true_theta)