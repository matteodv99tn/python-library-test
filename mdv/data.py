import numpy as np

import matplotlib.pyplot as plt


class Demonstration:

    def __init__(self, t: np.ndarray, x: np.ndarray, v: np.ndarray,
                 a: np.ndarray):
        self.t = t
        self.x = x
        self.v = v
        self.a = a

    def plot(self, show_plot: bool = False, title: str = ''):
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axs[0].plot(self.t, self.x, label='Position')
        axs[1].plot(self.t, self.v, label='Velocity')
        axs[2].plot(self.t, self.a, label='Acceleration')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        plt.xlabel('Time [s]')

        if title != '':
            axs[0].set_title(title)
        if show_plot:
            plt.show()

    def plot_compare(self, other, show_plot: bool = False):

        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axs[0].plot(self.t, self.x, label='Current')
        axs[0].plot(other.t, other.x, "--", label='Reference')
        axs[1].plot(self.t, self.v, label='Current')
        axs[1].plot(other.t, other.v, label='Reference')
        axs[2].plot(self.t, self.a, label='Current')
        axs[2].plot(other.t, other.a, label='Reference')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[0].set_title('Position')
        axs[1].set_title('Velocity')
        axs[2].set_title('Acceleration')
        plt.xlabel('Time [s]')

        if show_plot:
            plt.show()

        return fig, axs


def generate_reference_data(
    t_dem: float = 10.0,
    n_samples: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega = 0.5 * 2 * np.pi / t_dem
    t = np.linspace(0, t_dem, n_samples)
    x = np.cos(omega * t)
    v = -omega * np.sin(omega * t)
    a = -omega**2 * np.cos(omega * t)
    return t, x, v, a
