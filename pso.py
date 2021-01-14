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


def update(f, p_cur, v_cur, gl_best, p_bests, v_max, lims, top):
    """
    ARGUMENTS
        f      : objective function
        p_cur  : current distribution of particles
        v_cur  : current velocities of particles
        gl_best : global or local best location depending on topology scheme
        p_bests : each particles best position
        v_max  : maximum velocity of an individual
        top     : neighbourhood topology, choose from:
                    g - global
                    r - ring
                    w - wheel
    OUTPUTS
        p_cur  : updated particle positions
    """

    # constants
    w = 0.5
    c1 = 1.3
    c2 = 2.8

    for i in range(len(p_cur)):
        # find accelerations
        local_acc = np.random.rand() * (p_bests[i] - p_cur[i])
        if top == 'g':
            global_acc = np.random.rand() * (gl_best - p_cur[i])
        elif top == 'r' or 'w':
            global_acc = np.random.rand() * (gl_best[i] - p_cur[i])

        # update velocities
        v_cur[i] = (w * v_cur[i]) + (c1 * local_acc) + (c2 * global_acc)

        # restrict by max velocity
        if np.linalg.norm(v_cur[i]) <= v_max:
            v_cur[i] = v_cur[i]
        else:
            v_cur[i] = v_cur[i] * (v_max / np.linalg.norm(v_cur[i]))

        # update positions
        p_cur[i] = p_cur[i] + v_cur[i]
        p_cur[i] = np.clip(p_cur[i], lims[0], lims[1])

    return p_cur


def evaluate(f, p_cur, p_bests, top):
    """
    ARGUMENTS
        f       : objective function
        p_cur   : current distribution of particles
        p_bests : current personal best location
        top     : neighbourhood topology, choose from:
                    g - global
                    r - ring
                    w - wheel
    OUTPUTS
        p_bests : updated personal best locations
        l_bests : best location out of self and neighbours
        g_best  : location of new global best
        fg_best : value of new global best
    """

    p_num = len(p_cur)  # swarm size
    n = len(p_cur[0])  # dimension

    # update personal best locations
    for i in range(p_num):
        if f(p_cur[i]) < f(p_bests[i]):
            p_bests[i] = p_cur[i]

    # GLOBAL TOPOLOGY
    if top == 'g':

        # find global best
        fp_cur = [f(p_cur[i]) for i in range(p_num)]
        fg_best = np.min(fp_cur)
        g_best = p_cur[np.argmin(fp_cur)]

        return p_bests, g_best, fg_best

    # RING TOPOLOGY
    elif top == 'r':

        # find best location from particles either side
        l_bests = np.zeros((p_num, n))
        for i in range(p_num):
            local_vals = np.zeros(3)
            local_vals[0] = f(p_cur[i-2])
            local_vals[1] = f(p_cur[i-1])
            local_vals[2] = f(p_cur[i])
            l_bests[i-1] = p_cur[np.argmin(local_vals)+i-2]

        # also find global best
        fp_cur = [f(p_cur[i]) for i in range(p_num)]
        fg_best = np.min(fp_cur)
        g_best = p_cur[np.argmin(fp_cur)]

        return p_bests, l_bests, g_best, fg_best

    # WHEEL TOPOLOGY
    elif top == 'w':

        # find global best
        fp_cur = [f(p_cur[i]) for i in range(p_num)]
        fg_best = np.min(fp_cur)
        g_best = p_cur[np.argmin(fp_cur)]

        # find best location between self and leader particle
        l_bests = np.zeros((p_num, n))
        l_bests[0] = g_best
        for i in range(p_num-1):
            local_vals = np.zeros(2)
            local_vals[0] = f(p_cur[0])
            local_vals[1] = f(p_cur[i+1])
            if np.argmin(local_vals) == 0:
                l_bests[i+1] = p_cur[0]
            else:
                l_bests[i+1] = p_cur[i+1]

        return p_bests, l_bests, g_best, fg_best


def pso(f, p_num, v_max, n, lims, max_it, runtime, top='g'):
    """
    ARGUMENTS
        f       : objective function
        p_num   : swarm size
        v_max   : maximum velocity of an individual
        n       : dimension of function
        lims    : bounds of the optimisation function in form [min, max]
        max_it  : max number of iterations
        runtime : value to limit function runtime to in seconds
        top     : neighbourhood topology, choose from:
                     g - global
                     r - ring
                     w - wheel
    OUTPUT
        f_opt        : minimum objective function value reached
        g_opt        : location of global minimum found
        p_hist       : particle locations for each iteration
        f_hist       : objective function evaluated for every particle 
        fg_best_hist : best objective function obtained at every iteration 
        iters        : array of total iterations completed in runtime
    """

    # begin timing function
    start = time.time()

    p_hist = np.zeros((max_it, p_num, n))  # particle positions
    v_hist = np.zeros((max_it, p_num, n))  # particle velocities
    f_hist = np.zeros((max_it, p_num))  # objective function value for every particle
    l_bests_hist = np.zeros((max_it, p_num, n))  # local best position for each particle
    g_best_hist = np.zeros((max_it, n))  # global best particles
    fg_best_hist = np.zeros(max_it)  # global best values
    iters = [0]  # empty array for iteration numbers

    # initialise particles randomly
    p_hist[0] = np.random.uniform(lims[0], lims[1], size=(p_num, n))  # within feasible region
    v_hist[0] = np.random.uniform(lims[0], lims[1], size=(p_num, n))
    f_hist[0] = np.array([f(x) for x in p_hist[0]])

    # evaluate initial particles
    p_bests = p_hist[0]  # personal best locations
    if top == 'g':
        p_bests, g_best_hist[0], fg_best_hist[0] = evaluate(f, p_hist[0], p_bests, top)
    elif top == 'r' or 'w':
        p_bests, l_bests_hist[0], g_best_hist[0], fg_best_hist[0] = evaluate(f, p_hist[0], p_bests, top)


    # limit to maximum number of iterations
    for i in range(max_it-1):

        # run for specified time
        if time.time() - start > runtime:
            break

        if top == 'g':
            p_hist[i+1] = update(f, p_hist[i], v_hist[i], g_best_hist[i], p_bests, v_max, lims, top)
            p_bests, g_best_hist[i+1], fg_best_hist[i+1] = evaluate(f, p_hist[i+1], p_bests, top)

        elif top == 'r' or 'w':
            p_hist[i+1] = update(f, p_hist[i], v_hist[i], l_bests_hist[i], p_bests, v_max, lims, top)
            p_bests, l_bests_hist[i+1], g_best_hist[i+1], fg_best_hist[i+1] = evaluate(f, p_hist[i+1], p_bests, top)

        f_hist[i+1] = np.array([f(x) for x in p_hist[i+1]])
        iters.append(i+1)

    f_opt = np.min(fg_best_hist)
    g_opt = g_best_hist[np.argmin(fg_best_hist)]

    return f_opt, g_opt, p_hist, f_hist, fg_best_hist, iters


def visualize(f, history, lims, minima):
    """Visualize the process of optimizing
    ARGUMENTS
        func    : objective function
        history : object returned from pso above
        lims    : bounds of objective function
        minima  : minima to display on plot
    """

    # define meshgrid according to given boundaries
    x = np.linspace(lims[0], lims[1], 50)
    y = np.linspace(lims[0], lims[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([x, y]) for x, y in zip(X, Y)])

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

        ax1.set_xlim(lims[0]-50, lims[1]+50)
        ax1.set_ylim(lims[0]-50, lims[1]+50)

        # data to be plot
        data = history[frame]

        # contour and global minimum
        contour = ax1.contour(X, Y, Z, levels=20, cmap="magma")
        ax1.plot(minima[0], minima[1], marker='o', color='black')

        # plot particles
        ax1.scatter(data[:, 0], data[:, 1], marker='x', color='black')

        for i in range(data.shape[0]):
            ax1.plot(data[i][0], data[i][1], marker='x', color='black')

    ani = animation.FuncAnimation(fig, animate, fargs=(history,),
                                  frames=history.shape[0],
                                  interval=250)

    plt.show()


lims = [-500, 500]
minima = [-300.3376,  500]
#v_max generally should be 10% of lims[1] - lims[0]
# f_opt, g_opt, history, iters = pso(obj_f2, p_num=200, v_max=200, n=2, lims=lims, max_it=100, runtime=0.5, top='g')
# print(f_opt, g_opt, iters)
#visualize(obj_f2, history, lims, minima)


# # define meshgrid according to given boundaries
# x = np.linspace(lims[0], lims[1], 50)
# y = np.linspace(lims[0], lims[1], 50)
# X, Y = np.meshgrid(x, y)
# Z = np.array([obj_f2([x, y]) for x, y in zip(X, Y)])

# # initialize figure
# fig = plt.figure(figsize=(13, 6))
# ax1 = fig.add_subplot(122, facecolor='w', projection="3d")
# ax2 = fig.add_subplot(121, facecolor='w')

# # plot contour and 3d surface
# surf = ax1.plot_surface(X, Y, Z, cmap="inferno", linewidth=0, antialiased=True)
# contour = ax2.contour(X, Y, Z, levels=10, cmap="magma")

# ax2.set_xlim(lims[0]-50, lims[1]+50)
# ax2.set_ylim(lims[0]-50, lims[1]+50)
# ax2.plot(minima[0], minima[1], marker='o', color='black')

# ax1.set_xlabel('X1')
# ax1.set_ylabel('X2')
# ax1.set_zlabel('f(X1, X2)')

# ax2.set_xlabel('X1')
# ax2.set_ylabel('X2')

# plt.savefig('2D_Rana.png')
# plt.show()