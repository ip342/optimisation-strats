import numpy as np
import random
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
        x    :
        sig  :
        alph :
        lamb : number of offspring created in each iteration
        n    : number of control variables
        lims : bounds of the optimisation function in form [min, max]
    OUTPUTS
        x_mut    :
        sig_mut  :
        alph_mut :
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
        x    :
        sig  :
        alph :
        lamb : number of offspring created in each iteration
        mu   : number of parents
        n    : number of control variables
    OUTPUTS
        x_re    :
        sig_re  :
        alph_re :
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


def selection(f, x, sig, alph, mu):
    """ assume (λ,μ) selection for now """

    # evaluate fitness function of all offspring
    f_x = {i: f(x[i]) for i in range(len(x))}
    # select best μ contenders for next generation
    f_sort = dict(sorted(f_x.items(), key=lambda x: x[1])[0:mu])

    x_sel = np.array([x[i] for i in f_sort])
    sig_sel = np.array([sig[i] for i in f_sort])
    alph_sel = np.array([alph[i] for i in f_sort])
    f_sel = np.array([i for i in f_sort.values()])

    return x_sel, sig_sel, alph_sel, f_sel


def evo_strat(f, lamb, mu, n, lims, max_it):
    """
    ARGUMENTS
        f      : objective function
        lamb   : number of offspring created in each iteration
        mu     : number of parents selected for each following generation
        n      : dimension of objective function
        lims   : bounds of the optimisation function in form [min, max]
        max_it : max number of iterations
    OUTPUTS
        x_opt  :
        f_opt  :
        x_hist :
        f_hist :
    """
    # create empty arrays for history of each variable over max iterations
    x_hist = np.zeros((max_it, lamb, n))
    f_hist = np.zeros((max_it, lamb))
    sig_hist = np.zeros((max_it, lamb, n))
    alph_hist = np.zeros((max_it, lamb, n, n))

    # initialise first generation randomly
    x_hist[0] = np.random.uniform(lims[0], lims[1], size=(lamb, n))  # within feasible region
    f_hist[0] = [f(x) for x in x_hist[0]]
    sig_hist[0] = np.random.random(size=(lamb, n))  # within [0, 1]
    alph_hist[0] = np.random.uniform(-np.pi*4, np.pi*4, size=(lamb, n, n))  # within +- pi/4

    # perform first found of selection and find optimal obj f and x
    x_sel, sig_sel, alph_sel, f_sel = selection(f, x_hist[0], sig_hist[0], alph_hist[0], mu)
    x_opt = x_sel[0]
    f_opt = f_sel[0]

    for i in range(1, max_it):  # start at 1 as first gen already initialised

        # recombination of parameters
        x_re, sig_re, alph_re = recomb(x_sel, sig_sel, alph_sel, lamb, mu, n)

        # create next gen by mutation
        x_hist[i], sig_hist[i], alph_hist[i] = mutate(x_re, sig_re, alph_re, lamb, n, lims)

        # select members for next gen
        x_sel, sig_sel, alph_sel, f_sel = selection(f, x_hist[i], sig_hist[i], alph_hist[i], mu)

        if f_sel[0] < f_opt:
            f_opt = f_sel[0]
            x_opt = x_sel[0]

    return f_opt, x_opt, x_hist, f_hist


def visualize(func, history, lims, minima):
    """Visualize the process of optimizing
    ARGUMENTS
        func    : object function
        history : object returned from evo_strat above
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


lims = [-2, 2]
minima = [1, 1]
a, b, x_hist, f_hist = evo_strat(obj_f, lamb=100, mu=15, n=4, lims=lims, max_it=100)

visualize(obj_f, x_hist, lims, minima)
