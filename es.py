import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib import animation


def obj_f(x):
    """Rosenbrock function
    Domain: -5 < xi < 5
    Global minimum: f_min(1,..,1)=0
    """
    return 100*(x[1] - x[0]**2)**2 + (x[0]-1)**2


def obj_f2(x):
    """Rana function
    Domain: -512 < xi < 512
    Global minimum (2D): f_min(-488.6326, 512)=0
    """
    a = np.abs(x[1]+1-x[0])**0.5
    b = np.abs(x[1]+1+x[0])**0.5
    return x[0]*np.sin(a)*np.cos(b) + (x[1]+1)*np.cos(a)*np.sin(b)


def mutate(x, sig, alph, lamb, n, lims):
    """
    ARGUMENTS
        x    : generation locations
        sig  : generation standard deviations
        alph : generation rotation angles
        lamb : number of offspring created in each iteration
        n    : number of control variables
        lims : bounds of the optimisation function in form [min, max]
    OUTPUTS
        x_mut    : mutated generation locations
        sig_mut  : mutated generation standard deviations
        alph_mut : mutated generation rotation angles
    """

    # initialise control parameters as per schwefel 1995
    tau = 1/(2*(n)**0.5)**0.5
    tau_dash = 1/(2*n)**0.5
    beta = 0.0873

    # mutation of strategy variables
    sig_mut = sig * np.exp(tau_dash*np.random.randn(lamb, 1) +
                           tau*np.random.randn(lamb, n)) # different sized random 
                                                         # variable arrays are added by broadcasting
    alph_mut = alph + beta*np.random.randn(lamb, n, n)

    # full covariance matrix is created from mutated alph and sig
    cov = np.zeros((lamb, n, n))
    for i in range(lamb):
        for j in range(n):
            for k in range(n):
                if j == k:
                    a = sig_mut[i][j]**2
                else:
                    a = (np.tan(2*alph_mut[i][j][k]) * (sig_mut[i][j]**2 - sig_mut[i][k]**2)) / 2
                cov[i][j][k] = a

    # diagonal covariance matrix created from mutated sig
    # cov = np.zeros((lamb, n, n))
    # for i in range(lamb):
    #     cov[i] = np.diag(sig_mut[i]**2)

    # create mutated offspring
    n = np.array([np.random.multivariate_normal(np.zeros(n), cov[i]) for i in range(lamb)])

    x_mut = np.clip(x+n, lims[0], lims[1])  # clip to be within feasible region
                                   # (-2, 2) chosen for rosenbrook

    return x_mut, sig_mut, alph_mut


def recomb(x, sig, alph, lamb, mu, n):
    """
    ARGUMENTS
        x    : generation locations
        sig  : generation standard deviations
        alph : generation rotation angles
        lamb : number of offspring created in each iteration
        mu   : number of parents
        n    : number of control variables
    OUTPUTS
        x_re    : recombined generation locations
        sig_re  : recombined generation standard deviations
        alph_re : recombined generation rotation angles
    """

    # discrete recombination for control variables
    x_re = np.zeros((lamb, n))
    for i in range(lamb):
        # for each offspring choose two parents to inherit from
        parents = np.random.randint(mu, size=2)
        for j in range(n):
            x_re[i][j] = x[random.choice(parents)][j]

    # intermediate recombination for strategy variables
    sig_re = np.zeros((lamb, n))
    alph_re = np.zeros((lamb, n, n))
    for i in range(lamb):
        parents = np.random.randint(mu, size=2)
        sig_re[i] = 0.5*(sig[parents[0]] + sig[parents[1]])
        alph_re[i] = 0.5*(alph[parents[0]] + alph[parents[1]])

    return x_re, sig_re, alph_re


def selection(f, x, sig, alph, mu, x_p=None, sig_p=None, alph_p=None, scheme='comma'):
    """
    ARGUMENTS
        f      : objective function
        x      : generation locations
        sig    : generation standard deviations
        alph   : generation rotation angles
        mu     : number of parents
        scheme : selection scheme, choose from:
                    comma - selection is from newly generated offsprint
                    plus - selection is from offspring and parent population
    OUTPUTS
        x_sel    : selected locations for next generation parents
        sig_sel  : selected generation standard deviations
        alph_sel : selected generation rotation angles
        f_sel    : objective function evaluated for x_sel
    """

    if scheme == 'plus':
        # join offspring and previous parents into one array
        x = np.concatenate((x, x_p), axis=0)
        sig = np.concatenate((sig, sig_p), axis=0)
        alph = np.concatenate((alph, alph_p), axis=0)

    # evaluate fitness function of all offspring
    f_x = {i: f(x[i]) for i in range(len(x))}
    # select best μ contenders for next generation
    f_sort = dict(sorted(f_x.items(), key=lambda x: x[1])[0:mu])

    x_sel = np.array([x[i] for i in f_sort])
    sig_sel = np.array([sig[i] for i in f_sort])
    alph_sel = np.array([alph[i] for i in f_sort])
    f_sel = np.array([i for i in f_sort.values()])

    return x_sel, sig_sel, alph_sel, f_sel


def evo_strat(f, lamb, mu, n, lims, max_it, runtime, scheme):
    """
    ARGUMENTS
        f      : objective function
        lamb   : number of offspring created in each iteration
        mu     : number of parents selected for each following generation
        n      : dimension of objective function
        lims   : bounds of the optimisation function in form [min, max]
        max_it : max number of iterations
        runtime : value to limit function runtime to in seconds
        scheme : selection scheme, choose from:
                    comma - selection is from newly generated offsprint
                    plus - selection is from offspring and parent population
    OUTPUTS
        f_opt      : minimum objective function value reached
        x_opt      : location of minimum objective function
        f_opt_hist : best objective function obtained at every generation
        x_hist     : generation history of parent/offspring locations
        iters      : number of total generations evaluated in runtime
    """

    # start timing function
    start = time.time()

    # create empty arrays for history of each variable over max iterations
    x_hist = np.zeros((max_it, lamb, n))
    f_hist = np.zeros((max_it, lamb))
    f_opt_hist = np.zeros(max_it)
    sig_hist = np.zeros((max_it, lamb, n))
    alph_hist = np.zeros((max_it, lamb, n, n))

    # initialise first generation randomly
    x_hist[0] = np.random.uniform(lims[0], lims[1], size=(lamb, n))  # within feasible region
    f_hist[0] = [f(x) for x in x_hist[0]]
    f_opt_hist[0] = np.min(f_hist[0])
    sig_hist[0] = np.random.random(size=(lamb, n))  # within [0, 1]
    alph_hist[0] = np.random.uniform(-np.pi*4, np.pi*4, size=(lamb, n, n))  # within +- pi/4
    iters = [0]  # empty array for iteration numbers

    # perform first found of selection and find optimal obj f and x
    x_sel, sig_sel, alph_sel, f_sel = selection(f, x_hist[0], sig_hist[0], alph_hist[0], mu)
    x_opt = x_sel[0]
    f_opt = f_sel[0]

    for i in range(1, max_it):  # start at 1 as first gen already initialised

        # run for specified time
        if time.time() - start > runtime:
            break

        # recombination of parameters
        x_re, sig_re, alph_re = recomb(x_sel, sig_sel, alph_sel, lamb, mu, n)

        # create next gen by mutation
        x_hist[i], sig_hist[i], alph_hist[i] = mutate(x_re, sig_re, alph_re, lamb, n, lims)
        f_hist[i] = [f(x) for x in x_hist[i]]

        # select members for next gen
        x_sel, sig_sel, alph_sel, f_sel = selection(f, x_hist[i], sig_hist[i], alph_hist[i], mu, x_sel, sig_sel, alph_sel, scheme)

        # record best achieved value for generation
        f_opt_hist[i] = f_sel[0]

        if f_sel[0] < f_opt:
            f_opt = f_sel[0]
            x_opt = x_sel[0]

        iters.append(i)

    return f_opt, x_opt, f_opt_hist, x_hist, f_hist, iters


def visualize(func, history, lims, minima):
    """Visualize the process of optimizing
    ARGUMENTS
        func    : object function
        history : particle history object returned from evo_strat above
        lims    : bounds of objective function
        minima  : minima to display on plot
    """

    # define meshgrid according to given boundaries
    x = np.linspace(lims[0], lims[1], 50)
    y = np.linspace(lims[0], lims[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func([x, y]) for x, y in zip(X, Y)])

    # initialize figure
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, facecolor='w')
    ax2 = fig.add_subplot(122, facecolor='w', projection="3d")

    surf = ax2.plot_surface(X, Y, Z, cmap="inferno", linewidth=0, antialiased=True)

    # animation callback function
    def animate(frame, history):
        ax1.cla()
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')

        ax1.set_xlim(lims[0], lims[1])
        ax1.set_ylim(lims[0], lims[1])

        # data to be plot
        data = history[frame]

        # contour and global minimum
        contour = ax1.contour(X, Y, Z, levels=50, cmap="magma")
        ax1.plot(minima[0], minima[1], marker='o', color='black')

        # plot particles
        ax1.scatter(data[:, 0], data[:, 1], marker='x', color='black')

        for i in range(data.shape[0]):
            ax1.plot(data[i][0], data[i][1], marker='x', color='black')

    ani = animation.FuncAnimation(fig, animate, fargs=(history,),
                                  frames=history.shape[0],
                                  interval=250)

    plt.show()


def screenshot(f, history, lims, minima, f_best_hist, i):
    """Visualize the process of optimizing through evenly spaced screenshots of iterations
    ARGUMENTS
        f           : objective function
        history     : history of particle locations for each iteration
        lims        : bounds of objective function
        minima      : minima to display on plot
        f_best_hist : history of minimum obj func value for each iteration
        i           : array of total iterations
    """
    # define meshgrid according to given boundaries
    x = np.linspace(lims[0], lims[1], 50)
    y = np.linspace(lims[0], lims[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([x, y]) for x, y in zip(X, Y)])

    # initialize figure
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221, facecolor='w')
    ax2 = fig.add_subplot(222, facecolor='w')
    ax3 = fig.add_subplot(223, facecolor='w')
    ax4 = fig.add_subplot(224, facecolor='w')

    # contour and global minimum
    contour1 = ax1.contourf(X, Y, Z, levels=7, cmap="inferno")
    contour2 = ax2.contourf(X, Y, Z, levels=7, cmap="inferno")
    contour3 = ax3.contourf(X, Y, Z, levels=7, cmap="inferno")
    contour4 = ax4.contourf(X, Y, Z, levels=7, cmap="inferno")
    ax1.plot(minima[0], minima[1], marker='o', color='black')
    ax2.plot(minima[0], minima[1], marker='o', color='black')
    ax3.plot(minima[0], minima[1], marker='o', color='black')
    ax4.plot(minima[0], minima[1], marker='o', color='black')

    # evenly space the iterations to show
    a, b, c, d = 0, (i[-1])//3, 2*(i[-1])//3, i[-1]
    # graph titles
    ax1.set_title('iteration={} | f_min=({:.3f})'.format(a+1, f_best_hist[a]))
    ax2.set_title('iteration={} | f_min=({:.3f})'.format(b+1, f_best_hist[b]))
    ax3.set_title('iteration={} | f_min=({:.3f})'.format(c+1, f_best_hist[c]))
    ax4.set_title('iteration={} | f_min=({:.3f})'.format(d+1, f_best_hist[d]))

    # axes limits
    ax1.set_xlim(lims[0]-50, lims[1]+50), ax1.set_ylim(lims[0]-50, lims[1]+50)
    ax2.set_xlim(lims[0]-50, lims[1]+50), ax2.set_ylim(lims[0]-50, lims[1]+50)
    ax3.set_xlim(lims[0]-50, lims[1]+50), ax3.set_ylim(lims[0]-50, lims[1]+50)
    ax4.set_xlim(lims[0]-50, lims[1]+50), ax4.set_ylim(lims[0]-50, lims[1]+50)

    # data to be plot
    data1 = history[a]
    # data to be plot
    data2 = history[b]
    # data to be plot
    data3 = history[c]
    # data to be plot
    data4 = history[d]

    # plot particles
    ax1.scatter(data1[:, 0], data1[:, 1], marker='x', color='black')
    for i in range(data1.shape[0]):
        ax1.plot(data1[i][0], data1[i][1], marker='x', color='black')

    # plot particles
    ax2.scatter(data2[:, 0], data2[:, 1], marker='x', color='black')
    for i in range(data2.shape[0]):
        ax2.plot(data2[i][0], data2[i][1], marker='x', color='black')

    # plot particles
    ax3.scatter(data3[:, 0], data3[:, 1], marker='x', color='black')
    for i in range(data3.shape[0]):
        ax3.plot(data3[i][0], data3[i][1], marker='x', color='black')

    # plot particles
    ax4.scatter(data4[:, 0], data4[:, 1], marker='x', color='black')
    for i in range(data4.shape[0]):
        ax4.plot(data4[i][0], data4[i][1], marker='x', color='black')

    plt.savefig('2D_es.png')
    plt.show()


lims = [-500, 500]
minima = [-300.3376,  500]
# f_opt, x_opt, f_opt_hist, x_hist, iters = evo_strat(obj_f2, lamb=100, mu=15, n=2, lims=lims, max_it=100, runtime=0.2, scheme='plus')
# screenshot(obj_f2, x_hist, lims, minima, f_opt_hist, iters)

#visualize(obj_f, x_hist, lims, minima)