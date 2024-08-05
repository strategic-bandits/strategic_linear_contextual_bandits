import matplotlib.pyplot as plt
import numpy as np
import math
plt.style.use('ggplot')

iter = 20
runs_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
runs = len(runs_list)

include_uniform = True

xy_size = 18

name_epoch = "Epoch"

""" Load Uniform Regret """
# Note that uniform performs the same in truthful and untruthful settings
all_uniform = []
for x in runs_list:
    all_uniform.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\T_uniform_truthful_" + str(x) + ".txt"))
average_uniform = np.sum(all_uniform, axis=0) / runs
std_uniform = np.std(all_uniform, axis=0)
top_uniform = average_uniform + std_uniform
bot_uniform = average_uniform - std_uniform

ylimit = max(top_uniform) + 75
xlimit = -20

"""Regret Plots"""

all_regret_optgtm = []
all_regret_linucb = []
for x in runs_list:
    regret_optgtm = np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\optgtm_regret_" + str(x) + ".txt")
    regret_linucb = np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\linucb_regret_" + str(x) + ".txt")
    all_regret_optgtm.append(regret_optgtm)
    all_regret_linucb.append(regret_linucb)

all_regret = [all_regret_optgtm, all_regret_linucb]
average_regret = np.sum(all_regret, axis=1) / runs
std_regret = np.std(all_regret, axis=1)
# ste_regret = std_regret / np.sqrt(runs)
top_err = average_regret + std_regret
bot_err = average_regret - std_regret

plt.plot(average_regret[0], label='OptGTM', color='C1')
plt.fill_between(range(len(average_regret[0])), top_err[0], bot_err[0], alpha=0.2, color='C1')
plt.plot(average_regret[1], label='LinUCB', color='C0')
plt.fill_between(range(len(average_regret[1])), top_err[1], bot_err[1], alpha=0.2, color='C0')
# if include_uniform:
plt.plot([average_uniform[-1] for _ in range(iter+1)], label='Uniform', color='C3')
plt.fill_between(range(len(average_regret[0])), top_uniform[-1], bot_uniform[-1], alpha=0.2, color='C3')

plt.ylim(xlimit, ylimit)
plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(name_epoch, fontsize=xy_size)
plt.ylabel("Total Regret", fontsize=xy_size)
plt.savefig(r"plots\regret_plot.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()

print(average_regret[0][-1])

""" T step regret: After 20 Iterations """

all_regret_optgtm = []
all_regret_linucb = []
for x in runs_list:
    regret_optgtm = np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\T_optgtm_last_" + str(x) + ".txt")
    regret_linucb = np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\T_linucb_last_" + str(x) + ".txt")
    all_regret_optgtm.append(regret_optgtm)
    all_regret_linucb.append(regret_linucb)

all_regret = [all_regret_optgtm, all_regret_linucb]
average_regret = np.sum(all_regret, axis=1) / runs
std_regret = np.std(all_regret, axis=1)
# ste_regret = std_regret / np.sqrt(runs)
top_err = average_regret + std_regret
bot_err = average_regret - std_regret

print(average_regret[0][-1])

plt.plot(average_regret[0], label='OptGTM', color='C1')
plt.fill_between(range(len(average_regret[0])), top_err[0], bot_err[0], alpha=0.2, color='C1')
plt.plot(average_regret[1], label='LinUCB', color='C0')
plt.fill_between(range(len(average_regret[1])), top_err[1], bot_err[1], alpha=0.2, color='C0')
if include_uniform:
    plt.plot(average_uniform, label='Uniform', color='C3')
    plt.fill_between(range(len(average_regret[0])), top_uniform, bot_uniform, alpha=0.2, color='C3')

# max_regret = np.max([np.max(top_err), np.max(average_regret)]) + 20
max_regret = ylimit
plt.ylim(xlimit, max_regret)
plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Rounds in Epoch 20", fontsize=xy_size)
plt.ylabel("Regret", fontsize=xy_size)
plt.savefig(r"plots\last_plot.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()



""" T step regret: Truthful """

all_regret_optgtm = []
all_regret_linucb = []
for x in runs_list:
    regret_optgtm = np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\T_optgtm_truthful_" + str(x) + ".txt")
    regret_linucb = np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\T_linucb_truthful_" + str(x) + ".txt")
    all_regret_optgtm.append(regret_optgtm)
    all_regret_linucb.append(regret_linucb)

all_regret = [all_regret_optgtm, all_regret_linucb]
average_regret = np.sum(all_regret, axis=1) / runs
std_regret = np.std(all_regret, axis=1)
# ste_regret = std_regret / np.sqrt(runs)
top_err = average_regret + std_regret
bot_err = average_regret - std_regret


plt.plot(average_regret[0], label='OptGTM', color='C1')
plt.fill_between(range(len(average_regret[0])), top_err[0], bot_err[0], alpha=0.2, color='C1')
plt.plot(average_regret[1], label='LinUCB', color='C0')
plt.fill_between(range(len(average_regret[1])), top_err[1], bot_err[1], alpha=0.2, color='C0')
if include_uniform:
    plt.plot(average_uniform, label='Uniform', color='C3')
    plt.fill_between(range(len(average_regret[0])), top_uniform, bot_uniform, alpha=0.2, color='C3')

plt.ylim(xlimit, max_regret)
plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Rounds in Epoch 0", fontsize=xy_size)
plt.ylabel("Regret", fontsize=xy_size)
plt.savefig(r"plots\truthful_plot.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()


""" Feature Manipulation Plot """

all_feature_optgtm = []
all_feature_linucb = []
for x in runs_list:
    all_feature_optgtm.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\optgtm_feature_" + str(x) + ".txt"))
    all_feature_linucb.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\linucb_feature_" + str(x) + ".txt"))

all_feature = [all_feature_optgtm, all_feature_linucb]
average_feature = np.sum(all_feature, axis=1) / runs
std_feature = np.std(all_feature, axis=1) / runs
# ste_regret = std_regret / np.sqrt(runs)
top_err = average_feature + std_feature
bot_err = average_feature - std_feature

plt.plot(average_feature[0], label='OptGTM', color='C1')
plt.fill_between(range(iter + 1), top_err[0], bot_err[0], alpha=0.2, color='C1')
plt.plot(average_feature[1], label='LinUCB', color='C0')
plt.fill_between(range(iter + 1), top_err[1], bot_err[1], alpha=0.2, color='C0')

plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(name_epoch, fontsize=xy_size)
plt.ylabel("Total Context Manipulation", fontsize=xy_size)
plt.savefig(r"plots\feature_plot.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()


""" Reward Manipulation Plot """

all_feature_optgtm = []
all_feature_linucb = []
for x in runs_list:
    all_feature_optgtm.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\optgtm_reward_" + str(x) + ".txt"))
    all_feature_linucb.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\linucb_reward_" + str(x) + ".txt"))

all_feature = [all_feature_optgtm, all_feature_linucb]
average_feature = np.sum(all_feature, axis=1) / runs
std_feature = np.std(all_feature, axis=1) / runs
# ste_regret = std_regret / np.sqrt(runs)
top_err = average_feature + std_feature
bot_err = average_feature - std_feature

plt.plot(average_feature[0], label='OptGTM', color='C1')
plt.fill_between(range(iter + 1), top_err[0], bot_err[0], alpha=0.2, color='C1')
plt.plot(average_feature[1], label='LinUCB', color='C0')
plt.fill_between(range(iter + 1), top_err[1], bot_err[1], alpha=0.2, color='C0')

plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(name_epoch, fontsize=xy_size)
plt.ylabel("Total Reward Manipulation", fontsize=xy_size)
plt.savefig(r"plots\reward_plot.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()



""" Theta Error Plot """

all_theta_optgtm = []
all_theta_linucb = []
for x in runs_list:
    all_theta_optgtm.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\optgtm_theta_" + str(x) + ".txt"))
    all_theta_linucb.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\linucb_theta_" + str(x) + ".txt"))

all_theta = [all_theta_optgtm, all_theta_linucb]
average_theta = np.sum(all_theta, axis=1) / runs
std_theta = np.std(all_theta, axis=1) / runs
# ste_regret = std_regret / np.sqrt(runs)
top_err = average_theta + std_theta
bot_err = average_theta - std_theta

plt.plot(average_theta[0], label='OptGTM', color='C1')
plt.fill_between(range(iter + 1), top_err[0], bot_err[0], alpha=0.2, color='C1')
plt.plot(average_theta[1], label='LinUCB', color='C0')
plt.fill_between(range(iter + 1), top_err[1], bot_err[1], alpha=0.2, color='C0')

# plt.ylim(0.5, 1.05)
plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(name_epoch, fontsize=xy_size)
plt.ylabel(r"$\theta^*$ Estimation Error", fontsize=xy_size)
plt.savefig(r"plots\theta_plot.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()


""" Plot Utility """

all_util_optgtm = []
all_util_linucb = []
for x in runs_list:
    all_util_optgtm.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\optgtm_arm_utility_" + str(x) + ".txt"))
    all_util_linucb.append(np.loadtxt(r"C:\Users\thoma\Github\strategic_bandits\results\linucb_arm_utility_" + str(x) + ".txt"))

""" OptGTM """
colors = ['C3', 'C2', 'C4', 'C5']
for r in range(len(all_util_optgtm)):
    transposed_list = [list(i) for i in zip(*all_util_optgtm[r])]
    for i in range(len(colors)):
        plt.plot(range(iter + 1), transposed_list[i], color=colors[i], label=f"Arm {i + 1}" if r == 0 else "")
plt.ylim(-300, np.max(all_util_linucb) + 300)
plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(name_epoch, fontsize=xy_size)
plt.ylabel(r"Arm Utility", fontsize=xy_size)
plt.savefig(r"plots\arm_util_optgtm.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()

""" LinUCB """
for r in range(len(all_util_linucb)):
    transposed_list = [list(i) for i in zip(*all_util_linucb[r])]
    for i in range(len(colors)):
        plt.plot(range(iter + 1), transposed_list[i], color=colors[i], label=f"Arm {i + 1}" if r == 0 else "")
plt.ylim(-300, np.max(all_util_linucb) + 300)
plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(name_epoch, fontsize=xy_size)
plt.ylabel(r"Arm Utility", fontsize=xy_size)
plt.savefig(r"plots\arm_util_linucb.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()


""" Joint Utility Plot """
for r in range(len(all_util_optgtm)):
        transposed_list = [list(i) for i in zip(*all_util_optgtm[r])]
        transposed_list2 = [list(i) for i in zip(*all_util_linucb[r])]
        for i in range(len(colors)):
            plt.plot(range(iter + 1), transposed_list[i], color='C1', alpha=.5, label=f"Utility of the Arms under OptGTM" if r == 0 and i == 0 else "")
            plt.plot(range(iter + 1), transposed_list2[i], color='C0', alpha=.5, label=f"Utility of the Arms under LinUCB" if r == 0 and i == 0 else "")
plt.ylim(-300, np.max(all_util_linucb) + 300)
plt.tight_layout()
plt.locator_params(axis='x', nbins=iter)
plt.legend(fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(name_epoch, fontsize=xy_size)
plt.ylabel(r"Arm Utility", fontsize=xy_size)
plt.savefig(r"plots\arm_util_joint.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()


