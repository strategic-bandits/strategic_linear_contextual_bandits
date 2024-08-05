import numpy as np
from scipy import linalg


class OptGTM:
    """
    The Optimistic Grim Trigger Mechanism

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

        # active set of arms and quantities used for elimination
        self.active_arms = [i for i in range(self.K)]
        self.arm_cumulative_reward = np.zeros(self.K)
        self.arm_selection_n = np.zeros(self.K)
        self.total_lcb = np.zeros(self.K)

        # initialize matrices for estimation
        # we need K-many matrices
        self.V = []
        self.theta = []
        self.B = []
        for i in range(self.K):
            self.V.append(self.lam * np.identity(self.d))
            self.theta.append(np.zeros(self.d))
            self.B.append(np.zeros(self.d))
        self.beta = np.sqrt(self.lam) * self.d + np.sqrt(2 * np.log(self.T) + self.d * np.log((self.d + self.T * self.d ** 2) / self.d * self.lam))

    # update estimates given time step, action, reported context and observed reward
    def update(self, t, a, reward, mean):
        if a == self.K + 1:
            return
        self.V[a] = self.V[a] + np.outer(self.reported_contexts[t, a], self.reported_contexts[t, a])
        self.B[a] = self.B[a] + self.reported_contexts[t, a] * reward
        self.theta[a] = np.dot(linalg.inv(self.V[a]), self.B[a])
        self.total_return += mean
        self.cumulative_return.append(self.cumulative_return[t] + mean)

        # check for elimination
        self.arm_cumulative_reward[a] += reward
        self.arm_selection_n[a] += 1
        ubc_reward = self.arm_cumulative_reward[a] + 2 * np.sqrt(np.log(self.T) * self.arm_selection_n[a])
        self.total_lcb[a] += np.dot(self.theta[a], self.reported_contexts[t, a]) - self.beta * self.norm_bonus(a, self.reported_contexts[t, a])
        if self.total_lcb[a] > ubc_reward:
            self.active_arms.remove(a)
            print("Eliminated Arm", a)
        if not self.active_arms:
            print("All Arms Eliminated in Round:", t)

    def play(self, t):
        # if active set of arms is empty, play no arm, i.e., arm K+1
        if not self.active_arms:
            return int(self.K + 1)
        # LinUCB for active arms
        ucb = np.zeros(self.K)
        for i in self.active_arms:
            ucb[i] = np.dot(self.theta[i], self.reported_contexts[t, i]) + self.beta * self.norm_bonus(i, self.reported_contexts[t, i])
        a = np.argmax(ucb)
        if a not in self.active_arms:
            print("ERROR. non-active arm has been selected")
        return int(a)

    # outputs the norm of context vector w.r.t. V^-1
    def norm_bonus(self, i, context):
        squared = np.dot(np.dot(context, linalg.inv(self.V[i])), context)
        return np.sqrt(squared)

    def set_reported_contexts(self, contexts):
        self.reported_contexts = contexts

    def reset_return(self):
        self.total_return = 0
        self.cumulative_return = [0]

    def reset_estimates(self):
        self.V = []
        self.theta = []
        self.B = []
        self.active_arms = [i for i in range(self.K)]
        for i in range(self.K):
            self.V.append(self.lam * np.identity(self.d))
            self.theta.append(np.zeros(self.d))
            self.B.append(np.zeros(self.d))

    def get_theta_estimate(self):
        return self.theta

    def get_theta_error(self, true_theta):
       avg_theta = np.sum(self.theta, axis=0) / self.K
       return np.linalg.norm(avg_theta - true_theta)