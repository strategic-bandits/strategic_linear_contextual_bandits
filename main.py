import numpy as np
from contextual_bandit import ContextualBandit
from LinUCB import LinUCB
from optgtm import OptGTM
import matplotlib.pyplot as plt
import time

""" 
- K arms with features y_1, ..., y_k dimension d2
- T contexts c_1, ..., c_T dimension d1 
- feature mapping phi dimension d
- theta dimension d
"""

# Dimension
d = 5

# Unknown parameter theta
theta = [1, 1, .8, 1, 1]
true_theta = np.array([.2, .2, .2, .2, .2])
# theta = [.5, .5]


# Define the arm features
K = 5

""" Design a problem where arm 1 dominates along dimension 1, arm dominates along dimension 2 etc. """
# PUT NO ZEROS INTO THE FEATURES; ONLY NEGATIVE AND POSITIVE VALUES
y1 = [.1, -.1, -.1, -.1, -.1]
y2 = [-.1, .1, -.1, -.1, -.1]
y3 = [-.1, -.1, .1, -.1, -.1]
y4 = [-.1, -.1, -.1, .1, -.1]
y5 = [-.1, -.2, -.3, 0.2, -.2]
arm_features = [y1, y2, y3, y4]

# Define sequence of user contexts
T = 10000
prob = [0.5, 0.5, 0.5, 0.5, 0.5]  # context distribution
# prob = [.5, .5]
user_contexts = np.zeros((T, d))  # user_contexts[time, context element]
for t in range(T):
    # user_context = np.random.binomial(1, prob)
    user_context = np.random.uniform(-1, 1, d)
    user_contexts[t] = user_context

# initialize bandit environment
bandit = ContextualBandit(K=K, T=T, true_arm_features=arm_features.copy(), reported_arm_features=arm_features.copy(), user_contexts=user_contexts, theta=theta, d=d)

# initialize LinUCB
lin_ucb = LinUCB(K=K, T=T, reported_contexts=bandit.reported_arm_contexts.copy(), d=d)

# initialize OptGTM
opt_gtm = OptGTM(K=K, T=T, reported_contexts=bandit.reported_arm_contexts.copy(), d=d)


suffix = '10'


opt_gtm.reset_estimates()
for t in range(T):
    a = opt_gtm.play(t)
    mean, reward, regret = bandit.round(t, a)
    opt_gtm.update(t, a, reward, mean)
regret = [a - b for a, b in zip(bandit.optimal_return, opt_gtm.cumulative_return)]
np.savetxt(r"results\T_optgtm_truthful_" + str(suffix) + ".txt", regret)
print(bandit.arm_utilities)
print("Regret OptGTM", regret[-1])
print("Thetas", opt_gtm.theta)
bandit.reset_arm_utilities()


lin_ucb.reset_estimates()
for t in range(T):
    a = lin_ucb.play(t)
    mean, reward, regret = bandit.round(t, a)
    lin_ucb.update(t, a, reward, mean)
regret = [a - b for a, b in zip(bandit.optimal_return, lin_ucb.cumulative_return)]
np.savetxt(r"results\T_linucb_truthful_" + str(suffix) + ".txt", regret)
print(bandit.arm_utilities)
bandit.reset_arm_utilities()
print("Regret LinUCB", regret[-1])


uniform_return = 0
uniform_list = [0]
lin_ucb.reset_estimates()
for t in range(T):
    a = np.random.choice([i for i in range(K)])
    mean, reward, regret = bandit.round(t, a)
    uniform_return += mean
    uniform_list.append(uniform_return)
regret = [a - b for a, b in zip(bandit.optimal_return, uniform_list)]
bandit.reset_arm_utilities()
print("Regret Uniform", bandit.optimal_return[-1] - uniform_return)
np.savetxt(r"results\T_uniform_truthful_" + str(suffix) + ".txt", regret)
# plt.plot(regret)
# plt.show()

""" Arms Greedily Update Their Strategies """

print("")
print("")

iter = 20
avg = 1  # how many roll-outs we do to approximate the gradient
eps = 0.2
learning_rate = .2


linucb_true = True
optgtm_true = True

if linucb_true:
    regret_lin_ucb = []
    feature_corruption_lin_ucb = []
    reward_corruption_lin_ucb = []
    theta_error_lin_ucb = []
    arm_utilities_lin_ucb = []
    # gradient step for each agent
    for e in range(iter):
        grad_steps = np.zeros((K, d))
        # get current arm utilities
        bandit.reset_arm_utilities()
        lin_ucb.reset_return()
        for _ in range(avg):
            lin_ucb.reset_estimates()
            for t in range(T):
                a = lin_ucb.play(t)
                mean, reward, regret = bandit.round(t, a)
                lin_ucb.update(t, a, reward, mean)
        print("Theta Estimate LinUCB", lin_ucb.theta, "iteration", e)
        regret_lin_ucb.append(bandit.optimal_return[-1] - lin_ucb.total_return / avg)
        arm_utilities_lin_ucb.append(bandit.get_arm_utilities() / avg)
        print("Regret", regret_lin_ucb, "iteration", e)

        theta_error_lin_ucb.append(lin_ucb.get_theta_error(np.array(theta)))
        feature_corruption_lin_ucb.append(bandit.get_total_feature_corruption())
        reward_corruption_lin_ucb.append(bandit.get_total_reward_corruption())
        print("LinUCB Feature Corruption:", feature_corruption_lin_ucb[-1], "Reward Corruption:", reward_corruption_lin_ucb[-1], "Theta Error", theta_error_lin_ucb[-1], "iteration", e)

        current_arm_utility = bandit.arm_utilities / avg
        print("Arm Utility", current_arm_utility, "iteration", e)
        bandit.reset_arm_utilities()
        eps_arm_utility = np.zeros(K)
        current_features = bandit.reported_arm_features.copy()
        for i in range(K):
            # print("Original Arm Features", current_features)
            # each dimension separately
            for x in range(d):
                new_features = current_features[i].copy()
                if new_features[x] > .99:
                    new_features[x] -= eps
                    sign = -1
                else:
                    new_features[x] += eps
                    sign = 1
                new_features[x] = max(-1, min(1, new_features[x]))
                bandit.set_reported_features(i, new_features)
                bandit.reset_arm_utilities()
                lin_ucb.set_reported_contexts(bandit.reported_arm_contexts)
                for _ in range(avg):
                    lin_ucb.reset_estimates()
                    for t in range(T):
                        a = lin_ucb.play(t)
                        mean, reward, regret = bandit.round(t, a)
                        lin_ucb.update(t, a, reward, mean)
                eps_arm_utility[i] = bandit.arm_utilities[i] / avg
                # print("Utilities", current_arm_utility[i], eps_arm_utility[i])

                # calculate gradient step in direction x for arm i
                grad_steps[i, x] = min(0.1, max(-0.1, learning_rate * 10 ** (-4) * sign * (eps_arm_utility[i] - current_arm_utility[i]) / eps * (1 / (e + 1))))
                bandit.set_reported_features(i, current_features[i])
        print("Gradient Steps:", grad_steps)
        # perform gradient step
        bandit.feature_gradient_step(grad_steps)
        print("Reported Features:", bandit.reported_arm_features, "iteration", e + 1)

    # last round evaluation
    bandit.reset_arm_utilities()
    lin_ucb.reset_return()
    for _ in range(avg):
        lin_ucb.reset_estimates()
        for t in range(T):
            a = lin_ucb.play(t)
            mean, reward, regret = bandit.round(t, a)
            lin_ucb.update(t, a, reward, mean)
    regret_lin_ucb.append(bandit.optimal_return[-1] - lin_ucb.total_return / avg)
    regret = [a - b for a, b in zip(bandit.optimal_return, lin_ucb.cumulative_return)]
    np.savetxt(r"results\T_linucb_last_" + str(suffix) + ".txt", regret)

    arm_utilities_lin_ucb.append(bandit.get_arm_utilities() / avg)
    np.savetxt(r"results\linucb_arm_utility_" + str(suffix) + ".txt", arm_utilities_lin_ucb)

    print("Theta Estimate LinUCB", lin_ucb.theta, "Final Iteration")
    print("Regret", regret_lin_ucb, "Final Iteration")
    theta_error_lin_ucb.append(lin_ucb.get_theta_error(np.array(theta)))
    feature_corruption_lin_ucb.append(bandit.get_total_feature_corruption())
    reward_corruption_lin_ucb.append(bandit.get_total_reward_corruption())
    print("LinUCB Feature Corruption:", feature_corruption_lin_ucb[-1], "Reward Corruption:", reward_corruption_lin_ucb[-1], "Theta Error", theta_error_lin_ucb[-1], "Final Iteration")

    np.savetxt(r"results\linucb_regret_" + suffix + ".txt", regret_lin_ucb)
    np.savetxt(r"results\linucb_feature_" + suffix + ".txt", feature_corruption_lin_ucb)
    np.savetxt(r"results\linucb_reward_" + suffix + ".txt", reward_corruption_lin_ucb)
    np.savetxt(r"results\linucb_theta_" + suffix + ".txt", theta_error_lin_ucb)

print("")
print("")

""" OptGTM """

if optgtm_true:
    # reset bandit environment
    for i in range(K):
        bandit.set_reported_features(i, arm_features[i])

    regret_optgtm = []
    feature_corruption_optgtm = []
    reward_corruption_optgtm = []
    theta_error_optgtm = []
    arm_utilities_optgtm = []
    # gradient step for each agent
    for e in range(iter):
        grad_steps = np.zeros((K, d))
        # get current arm utilities
        bandit.reset_arm_utilities()
        opt_gtm.reset_return()
        for _ in range(avg):
            opt_gtm.reset_estimates()
            for t in range(T):
                a = opt_gtm.play(t)
                mean, reward, regret = bandit.round(t, a)
                opt_gtm.update(t, a, reward, mean)
        print("Theta Estimate OptGTM", opt_gtm.theta, "iteration", e)
        regret_optgtm.append(bandit.optimal_return[-1] - opt_gtm.total_return / avg)  # removed / avg
        print("Regret", regret_optgtm, "iteration", e)
        arm_utilities_optgtm.append(bandit.get_arm_utilities() / avg)

        # report feature corruption and reward corruption
        theta_error_optgtm.append(opt_gtm.get_theta_error(np.array(theta)))
        feature_corruption_optgtm.append(bandit.get_total_feature_corruption())
        reward_corruption_optgtm.append(bandit.get_total_reward_corruption())
        print("OptGTM Feature Corruption:", feature_corruption_optgtm[-1], "Reward Corruption:", reward_corruption_optgtm[-1], "Theta Error", theta_error_optgtm[-1], "iteration", e)

        current_arm_utility = bandit.arm_utilities / avg
        print("Arm Utility", current_arm_utility, "iteration", e)
        bandit.reset_arm_utilities()
        eps_arm_utility = np.zeros(K)
        current_features = bandit.reported_arm_features.copy()
        for i in range(K):
            # print("Original Arm Features", current_features)
            # each dimension separately
            for x in range(d):
                new_features = current_features[i].copy()
                if new_features[x] > .99:
                    new_features[x] -= eps
                    sign = -1
                else:
                    new_features[x] += eps
                    sign = 1
                new_features[x] = max(-1, min(1, new_features[x]))
                bandit.set_reported_features(i, new_features)
                bandit.reset_arm_utilities()
                opt_gtm.set_reported_contexts(bandit.reported_arm_contexts)
                for _ in range(avg):
                    opt_gtm.reset_estimates()
                    for t in range(T):
                        a = opt_gtm.play(t)
                        mean, reward, regret = bandit.round(t, a)
                        opt_gtm.update(t, a, reward, mean)
                eps_arm_utility[i] = bandit.arm_utilities[i] / avg
                # print("Utilities", current_arm_utility[i], eps_arm_utility[i])
                # print("Active Arms Epsilon Step OptGTM:", opt_gtm.active_arms, "iteration", e)

                # calculate gradient step in direction x for arm i
                grad_steps[i, x] = min(0.1, max(-0.1, learning_rate * 10 ** (-4) * sign * (eps_arm_utility[i] - current_arm_utility[i]) / eps * (1 / (e + 1))))
                bandit.set_reported_features(i, current_features[i])
        print("Gradient Steps:", grad_steps)
        # perform gradient step
        bandit.feature_gradient_step(grad_steps)
        print("Reported Features:", bandit.reported_arm_features, "iteration", e + 1)
        print("Active Arms OptGTM:", opt_gtm.active_arms, "iteration", e + 1)

    # Last evaluation
    bandit.reset_arm_utilities()
    opt_gtm.reset_return()
    for _ in range(avg):
        opt_gtm.reset_estimates()
        for t in range(T):
            a = opt_gtm.play(t)
            mean, reward, regret = bandit.round(t, a)
            opt_gtm.update(t, a, reward, mean)
    regret_optgtm.append(bandit.optimal_return[-1] - opt_gtm.total_return / avg)
    regret = [a - b for a, b in zip(bandit.optimal_return, opt_gtm.cumulative_return)]
    np.savetxt(r"results\T_optgtm_last_" + str(suffix) + ".txt", regret)

    arm_utilities_optgtm.append(bandit.get_arm_utilities() / avg)
    np.savetxt(r"results\optgtm_arm_utility_" + str(suffix) + ".txt", arm_utilities_optgtm)

    print("Theta Estimate OptGTM", opt_gtm.theta, "Final Iteration")
    print("Regret", regret_optgtm, "Final Iteration", )
    theta_error_optgtm.append(opt_gtm.get_theta_error(np.array(theta)))
    feature_corruption_optgtm.append(bandit.get_total_feature_corruption())
    reward_corruption_optgtm.append(bandit.get_total_reward_corruption())
    print("OptGTM Feature Corruption:", feature_corruption_optgtm[-1], "Reward Corruption:", reward_corruption_optgtm[-1], "Theta Error:", theta_error_optgtm[-1], "Final Iteration")

    np.savetxt(r"results\optgtm_regret_" + suffix + ".txt", regret_optgtm)
    np.savetxt(r"results\optgtm_feature_" + suffix + ".txt", feature_corruption_optgtm)
    np.savetxt(r"results\optgtm_reward_" + suffix + ".txt", reward_corruption_optgtm)
    np.savetxt(r"results\optgtm_theta_" + suffix + ".txt", theta_error_optgtm)

if linucb_true and optgtm_true:
    plt.plot(range(iter + 1), regret_lin_ucb)
    plt.plot(range(iter + 1), regret_optgtm)
    plt.show()
