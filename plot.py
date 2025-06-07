import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

plt.rcParams['text.usetex'] = True
sns.set_theme()

TOTAL_COST = 10000
QUERY_COST = 1
X = np.arange(TOTAL_COST)

with open(f"attack_CGBA_{TOTAL_COST}.0_1.0_None_0.0.log", "rb") as fp:
    cgba_result = pickle.load(fp)
with open(f"attack_HSJA_{TOTAL_COST}.0_1.0_None_0.0.log", "rb") as fp:
    hsja_result = pickle.load(fp)
# with open(f"attack_OPT_{TOTAL_COST}.0_1.0_None_0.0.log", "rb") as fp:
#     opt_result = pickle.load(fp)
with open(f"attack_SURFREE_{TOTAL_COST}.0_1.0_None_0.0.log", "rb") as fp:
    surfree_result = pickle.load(fp)
with open(f"attack_GEODA_{TOTAL_COST}.0_1.0_None_0.0.log", "rb") as fp:
    geoda_result = pickle.load(fp)

def get_per_iteration(logs: dict):
    costs = [0]
    norms = []
    for log in logs:
        for k, v in log.items():
            if len(k.split(',')) == 2 and k.split(',')[0] == 'norm':
                norms.append(v)
            if len(k.split(',')) == 2 and k.split(',')[0] == 'cost':
                costs.append(v)
    f = interp1d(costs, norms)
    return f(X)

def get_attack_plot(logs):
    ans = []
    for log in logs:
        ans.append(get_per_iteration(log[1]))
    return np.median(ans, axis=0)

# plt.plot(X, np.log(get_attack_plot(cgba_result)), linestyle='-', label='CGBA')
# plt.plot(X, np.log(get_attack_plot(hsja_result)), linestyle='-.', label='HSJA')
# plt.plot(X, np.log(get_attack_plot(surfree_result)), linestyle='--', label='SURFREE')
# plt.plot(X, np.log(get_attack_plot(geoda_result)), linestyle='--', label='GEODA', dashes=(10, 5, 20, 5))
# plt.plot(X, np.log(get_attack_plot(opt_result)), linestyle=':', label='OPT')
plt.plot(X, get_attack_plot(cgba_result), linestyle='-', label='CGBA')
plt.plot(X, get_attack_plot(hsja_result), linestyle='-.', label='HSJA')
plt.plot(X, get_attack_plot(surfree_result), linestyle='--', label='SURFREE')
plt.plot(X, get_attack_plot(geoda_result), linestyle='--', label='GEODA', dashes=(10, 5, 20, 5))
# plt.plot(X, get_attack_plot(opt_result), linestyle=':', label='OPT')

plt.title(r"$C_{total}=1000$ $C_{flagged}=1$")
plt.xlabel('Cost')
plt.ylabel(r'Median $l_2$ perturbation')
plt.legend()
# plt.ylim(0, 50)
plt.show()

# print("TABLE RESULTS: CGBA", get_attack_plot(cgba_result)[-1])
# # print("TABLE RESULTS: OPT", get_attack_plot(opt_result)[-1])
# print("TABLE RESULTS: HSJA", get_attack_plot(hsja_result)[-1])
# print("TABLE RESULTS: SURFREE", get_attack_plot(surfree_result)[-1])
# print("TABLE RESULTS: GEODA", get_attack_plot(geoda_result)[-1])
print("TABLE RESULTS: CGBA", get_attack_plot(cgba_result)[1000])
# print("TABLE RESULTS: OPT", get_attack_plot(opt_result)[-1])
print("TABLE RESULTS: HSJA", get_attack_plot(hsja_result)[1000])
print("TABLE RESULTS: SURFREE", get_attack_plot(surfree_result)[1000])
print("TABLE RESULTS: GEODA", get_attack_plot(geoda_result)[1000])