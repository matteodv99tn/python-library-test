import casadi as cas
import numpy as np
import matplotlib.pyplot as plt

from mdv.data import generate_reference_data, Demonstration
from mdv.dmp import Dmp
from mdv.pprint import heading

plot_dmp_data = False
plot_solutions = True

# A test
heading("DMP", "Load data", "DMP learning")

print("Loading reference data")
t, pos, vel, acc = generate_reference_data(5, 1000)
dem = Demonstration(t, pos, vel, acc)

print("Creating and learning DMP")
alpha = 48.0
dmp = Dmp(alpha, alpha / 4.0, 2, 15)
dmp.assign_demonstration(dem)
fd = dmp.eval_forcing_term()
w = dmp.learn_weights()

# dmp.integrate(0.01, dem.t[-1]).plot(title="Integrated DMP")
# dem.plot(title="Reference trajectory")
if plot_dmp_data:
    exec_tr = dmp.integrate(0.01, dem.t[-1])
    exec_tr.plot_compare(dem)
    dmp.plot_learned_forcing()
    dmp.plot_basis()
    plt.show()

h = dmp.h
c = dmp.c
nb = dmp.n_basis

alpha = dmp.alpha
beta = dmp.beta
gamma = dmp.gamma
g = dmp.g
y0 = dmp.y0

print("Number of basis:", nb)

heading("CASADI", "Problem", "Formulation")
print("Initial condition:", y0)
print("Goal:", g)
print("alpha:", alpha)
print("beta:", beta)
print("gamma:", gamma)

vmax = 1.0
print("Maximum velocity:", vmax)

v_delta = 0.3
g_delta = 0.2
print("Max goal reaching error:", g_delta)
print("Max velocity error at end:", v_delta)

w_max = 1.2 * dmp.w
w_min = 0.8 * dmp.w
use_bounds = True

if use_bounds:
    print("Boundaries for the weights:")
    print(" > Min:", w_min)
    print(" > Max:", w_max)

time_horiz_mult = 1
N = 200  # number of control intervals
print("Splitting time horizon in", N, "steps")

# Optimisation problem

opti = cas.Opti()

# Decision variables
X = opti.variable(2, time_horiz_mult * N + 1)  # state trajectory
z = X[0, :]
y = X[1, :]
u = opti.variable(dmp.n_basis)  # weights
tau = opti.variable()

# Basis functions and forcing term
base = lambda c, h, xi: \
    cas.exp( -h * (cas.exp(-gamma * xi) - c)**2)
basis_sum = lambda xi: sum(base(c[i], h[i], xi) for i in range(dmp.n_basis))
ft = lambda w, xi: sum(
    [w[i] * base(c[i], h[i], xi) for i in range(dmp.n_basis)]) / basis_sum(xi)

# Objective function
opti.minimize(tau)

f = lambda x, w, zeta: cas.vertcat(
    alpha * (beta *
             (g - x[1]) - x[0]) + ft(w, zeta) * cas.exp(-gamma * zeta), x[0])

# RK Integration
dt = 1 / N
for k in range(time_horiz_mult * N):
    k1 = f(X[:, k], u, k * dt)
    k2 = f(X[:, k] + dt / 2 * k1, u, k * dt + dt / 2)
    k3 = f(X[:, k] + dt / 2 * k2, u, k * dt + dt / 2)
    k4 = f(X[:, k] + dt * k3, u, k * dt + dt)
    x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

# Path constraints
opti.subject_to(z <= tau * vmax)
opti.subject_to(z >= -tau * vmax)

# Initial condition constraints
opti.subject_to(z[0] == 0)
opti.subject_to(y[0] == y0)

# Final condition constraints
opti.subject_to(z[-1] / tau <= v_delta)
opti.subject_to(z[-1] / tau >= -v_delta)
opti.subject_to(y[-1] - g <= g_delta)
opti.subject_to(y[-1] - g >= -g_delta)

# Optimisation variables domain
opti.subject_to(tau >= 0.0)
if use_bounds:
    opti.subject_to(u <= w_max)
    opti.subject_to(u >= w_min)

# Initial guess for the solver
opti.set_initial(z, 0)
opti.set_initial(y, y0)
opti.set_initial(tau, 2 * dmp.tau)
opti.set_initial(u, dmp.w)

# NLP Solver
heading("IPOPT", "Solving the problem")
opti.solver("ipopt")  # set numerical backend

try:
    sol = opti.solve()  # actual solve
except:
    print("Solver failed!")
    exit(1)

print("Optimal weights:", sol.value(u))
print("Solution found with optimal tau:", sol.value(tau))

# Plotting

t = np.linspace(0, sol.value(tau), time_horiz_mult * N + 1)
ys = sol.value(y)
zs = sol.value(z)
taus = sol.value(tau)

if plot_solutions:
    plt.plot(t, ys, label="$y(t)$")
    plt.plot(0, y0, "bo", label="y0")
    plt.plot(taus, g, "go", label="g")
    plt.vlines(taus, g - g_delta, g + g_delta, "g", "solid")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()

    plt.figure()
    plt.plot(t, zs / taus, label=r"$\dot y(t)$")
    plt.hlines(vmax, 0, taus, "r", "dashed", label="$v_{lim}$")
    plt.hlines(-vmax, 0, taus, "r", "dashed")
    plt.vlines(taus, -v_delta, v_delta, "g", "solid")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()

    plt.figure()
    plt.plot(w_min, "ro")
    plt.plot(w_max, "ro")
    plt.plot(sol.value(u), "go")

opt_dmp = dmp.clone()
opt_dmp.w = sol.value(u)
opt_dmp.tau = sol.value(tau)
_, axs = opt_dmp.integrate(0.01, dmp.tau).plot_compare(dem)
for ax in axs:
    ymin, ymax = ax.get_ylim()
    ax.vlines(taus, ymin, ymax, "g", "dashed",  label=r"$\tau_{opt}$")

plt.show()
