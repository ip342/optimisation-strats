from es import evo_strat
from pso import pso
import numpy as np
import matplotlib.pyplot as plt


def obj_f(x):
    """Rana function
    Domain: -512 < xi < 512
    """
    n = len(x)
    f = 0
    for i in range(n-1):
        a = np.abs(x[i+1]+1-x[i])**0.5
        b = np.abs(x[i+1]+1+x[i])**0.5
        f += x[i]*np.sin(a)*np.cos(b) + (x[i+1]+1)*np.cos(a)*np.sin(b)
    return f


# """PSO"""

lims = [-500, 500]
p_num = 200
v_max = 200
n = 5
max_it = 100

f_opts = []
dis_solns = []
for i in range(100):

    f_opt, g_opt, p_hist, f_hist, f_opt_hist, pso_iters = pso(obj_f, p_num=p_num, v_max=v_max, n=n, lims=lims, max_it=max_it, runtime=0.5, top='g')

    # get rid of zero values for iterations not reached
    f_opt_hist = f_opt_hist[f_opt_hist != 0]

    # find 15 dissimilar solutions
    f_hist = f_hist.flatten()
    p_hist = np.reshape(p_hist, (p_num*max_it, n))
    f_x = {i: f_hist[i] for i in range(p_num*max_it)}
    f_sort = sorted(f_x.items(), key=lambda x: x[1])  # sort particles in ascending order wrt obj function value
    dis_archive = []
    dis_archive.append([f_sort[0][1], p_hist[f_sort[0][0]]])  # first solution is best obj function value reached

    for f in f_sort:
        for g in dis_archive:
            if len(dis_archive) >= 15:
                break
            if np.linalg.norm(p_hist[f[0]] - g[1]) > 200:  # D_min=100
                dis_archive.append([f[1], p_hist[f[0]]])
                dis_solns.append(f[1])
                break
            break

    f_opts.append(f_opt)

print('Min f mean = ', np.mean(f_opts))
print('Min f stdv = ', np.std(f_opts))
print('Dis solns mean = ', np.mean(dis_solns))
# fig = plt.figure(figsize=(6,6))
# ax1 = fig.add_subplot(111, facecolor='w')
# plt.plot(pso_iters, f_opt_hist)
# plt.plot(es_iters, f_hist)
# plt.show()

"""EVO STRAT"""

lims = [-500, 500]
lamb = 200
mu = 10
n = 5
max_it = 100

f_opts = []
dis_solns = []
for i in range(100):

    f_opt, x_opt, f_opt_hist, x_hist, f_hist, iters = evo_strat(obj_f, lamb=lamb, mu=mu, n=n, lims=lims, max_it=max_it, runtime=0.5, scheme='comma')

    # get rid of zero values for iterations not reached
    f_opt_hist = f_opt_hist[f_opt_hist != 0]
    # find 15 dissimilar solutions
    f_hist = f_hist.flatten()
    x_hist = np.reshape(x_hist, (lamb*max_it, n))
    f_x = {i: f_hist[i] for i in range(lamb*max_it)}
    f_sort = sorted(f_x.items(), key=lambda x: x[1])  # sort particles in ascending order wrt obj function value
    dis_archive = []
    dis_archive.append([f_sort[0][1], x_hist[f_sort[0][0]]])  # first solution is best obj function value reached

    for f in f_sort:
        for g in dis_archive:
            if len(dis_archive) >= 15:
                break
            if np.linalg.norm(x_hist[f[0]] - g[1]) > 200:  # D_min=100
                dis_archive.append([f[1], x_hist[f[0]]])
                dis_solns.append(f[1])
                break

    f_opts.append(f_opt)

print('Min f mean = ', np.mean(f_opts))
print('Min f stdv = ', np.std(f_opts))
print('Dis solns mean = ', np.mean(dis_solns))