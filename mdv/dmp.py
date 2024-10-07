import numpy as np
import matplotlib.pyplot as plt

from .data import Demonstration


def eval_gaussian_basis(x: float | np.ndarray, c: float | np.ndarray,
                        h: float | np.ndarray) -> float | np.ndarray:
    from .concepts import is_floating, is_array

    def eval_single_basis(x: float, c: float, h: float) -> float:
        return np.exp(-h * (x - c)**2)

    def eval_whole_basis(x: float, c: np.ndarray, h: np.ndarray) -> np.ndarray:
        row = np.array([np.exp(-h[i] * (x - c[i])**2) for i in range(len(c))])
        return row / np.sum(row)

    assert (type(c) == type(h))
    if (is_floating(x) and is_floating(c)):
        return eval_single_basis(x, c, h)
    if (is_floating(x) and is_array(c)):
        return eval_whole_basis(x, c, h)
    if (is_array(x) and is_array(c)):
        Phi = np.zeros((len(x), len(c)))
        for i in range(len(x)):
            Phi[i, :] = eval_whole_basis(x[i], c, h)
        return Phi

    raise ValueError('Invalid input types')


class Dmp:

    def __init__(self,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 n_basis: int = 15):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_basis = n_basis

        self.tau = 1.0
        self.g = 0.0

        # Weight centers
        self.c = np.array([
            np.exp(-self.gamma * i / (self.n_basis+1)) for i in range(self.n_basis)
        ])
        self.h = np.zeros(n_basis)
        self.w = np.zeros(n_basis)
        for i in range(n_basis - 1):
            self.h[i] = 1.2 / (self.c[i + 1] - self.c[i])**2
        self.h[-1] = self.h[-2]
        assert (self.h.shape == (n_basis, ))
        assert (self.c.shape == (n_basis, ))
        assert (self.w.shape == (n_basis, ))

        self.dem: Demonstration | None = None
        self.fd: np.ndarray | None = None

    def clone(self):
        dmp = Dmp(self.alpha, self.beta, self.gamma, self.n_basis)
        dmp.tau = self.tau
        dmp.g = self.g
        dmp.y0 = self.y0
        dmp.c = self.c.copy()
        dmp.h = self.h.copy()
        dmp.w = self.w.copy()
        return dmp

    def assign_demonstration(self, dem: Demonstration | None):
        if dem is not None:
            self.dem = dem
            self.tau = dem.t[-1]
            self.g = dem.x[-1]
            self.y0 = dem.x[0]

    def eval_forcing_term(self) -> np.ndarray:
        assert (self.dem is not None)
        dem = self.dem
        self.fd = self.tau**2 * dem.a - self.alpha * (self.beta *
                                                      (self.g - dem.x) - dem.v)
        return self.fd

    def learn_weights(self) -> np.ndarray:
        if self.fd is None:
            self.eval_forcing_term()
        assert (self.fd is not None)
        assert (self.dem is not None)

        s = np.exp(-self.gamma * self.dem.t / self.tau)
        assert (s.shape == self.fd.shape)
        f = self.fd / s
        Phi = eval_gaussian_basis(s, self.c, self.h)
        assert (Phi.shape == (len(s), self.n_basis))
        self.w = np.linalg.lstsq(Phi, f)[0]
        assert (self.w.shape == (self.n_basis, ))
        return self.w

    def integrate(self, dt: float, T: float) -> Demonstration:
        assert (self.w is not None)
        assert (self.tau > 0)
        assert (self.alpha > 0)
        assert (self.beta > 0)
        assert (self.gamma > 0)
        assert (self.g is not None)
        assert (self.y0 is not None)
        assert (len(self.w) == self.n_basis)
        assert (len(self.c) == self.n_basis)
        assert (len(self.h) == self.n_basis)
        t = np.arange(0, T, dt)
        s = np.exp(-self.gamma * t / self.tau)

        y = np.zeros_like(t)
        z = np.zeros_like(t)
        a = np.zeros_like(t)
        y[0] = self.y0
        z[0] = 0.0

        for i in range(len(t) - 1):
            Phi = eval_gaussian_basis(s[i], self.c, self.h)
            f = Phi @ self.w * s[i]
            dz_dt = self.alpha * (self.beta * (self.g - y[i]) - z[i]) + f
            a[i] = dz_dt / self.tau
            z[i + 1] = z[i] + dt * dz_dt / self.tau
            y[i + 1] = y[i] + dt * z[i] / self.tau
        a[-1] = a[-2]

        return Demonstration(t, y, z / self.tau, a / self.tau)

    def plot_learned_forcing(self, show_plot: bool = False):
        assert (self.dem is not None)
        assert (self.fd is not None)
        assert (self.w is not None)

        s = np.exp(-self.gamma * self.dem.t / self.tau)
        Phi: np.ndarray = eval_gaussian_basis(s, self.c, self.h)
        assert (Phi.shape == (len(s), self.n_basis))
        f = Phi @ self.w * s
        plt.figure(figsize=(6, 2))
        plt.plot(self.dem.t, self.fd, label='Desired (demonstration)')
        plt.plot(self.dem.t, f, label='Learned')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.title("Forcing term")
        if show_plot:
            plt.show()

    def plot_basis(self, show_plot: bool = False):
        assert (self.dem is not None)
        s = np.exp(-self.gamma * self.dem.t / self.tau)

        Phi = eval_gaussian_basis(s, self.c, self.h)
        fif, axs = plt.subplots(3, 1, figsize=(8, 6))

        for i in range(self.n_basis):
            axs[0].plot(s, Phi[:, i], label=f'Basis {i}')
            axs[1].plot(self.dem.t, Phi[:, i], label=f'Basis {i}')

        axs[2].plot(self.dem.t, s, label='Phase variable')
        axs[0].set_xlabel('coordinate system')
        axs[1].set_xlabel('time [s]')
        axs[2].set_xlabel('time [s]')
        axs[2].set_ylabel('Phase variable')

        axs[0].set_title("Basis VS Phase variable")
        axs[1].set_title("Basis VS Time")
        axs[2].set_title("Phase variable VS Time")

        if show_plot:
            plt.show()
