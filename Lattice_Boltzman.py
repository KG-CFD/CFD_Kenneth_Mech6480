

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

def main():
    # Lattice Boltzmann Simulation Parameters
    Nx, Ny = 400, 100
    rho0, tau = 100, 0.6
    Nt = 4000
    plotRealTime = True

    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    F = np.ones((Ny, Nx, NL))
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    X, Y = np.meshgrid(range(Nx), range(Ny))
    cylinder = (X - Nx / 4)**2 + (Y - Ny / 2)**2 < (Ny / 4)**2

    # Time series storage for velocity at a probe point
    probe_y = Ny // 2
    probe_x = int(3 * Nx / 4)
    velocity_probe = []

    for it in range(Nt):
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        bndryF = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) +
                                      9 * (cx * ux + cy * uy)**2 / 2 -
                                      3 * (ux**2 + uy**2) / 2)

        F += -(1.0 / tau) * (F - Feq)
        F[cylinder, :] = bndryF

        # Store the x-velocity at the probe point
        velocity_probe.append(ux[probe_y, probe_x])

        if (plotRealTime and (it % 10) == 0) or (it == Nt - 1):
            plt.cla()
            ux[cylinder] = 0
            uy[cylinder] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - \
                        (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[cylinder] = np.nan
            cmap = plt.cm.bwr
            cmap.set_bad('black')
            vorticity = np.ma.array(vorticity, mask=cylinder)
            plt.imshow(vorticity, cmap='bwr')
            plt.imshow(~cylinder, cmap='gray', alpha=0.3)
            plt.clim(-.1, .1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)

    # FFT Analysis
    velocity_probe = np.array(velocity_probe)
    fft_result = fft(velocity_probe)
    freq = fftfreq(len(velocity_probe), d=1)  # Assuming dt=1

    # Plot the Frequency Spectrum
    plt.figure()
    plt.plot(freq[:len(freq)//2], np.abs(fft_result[:len(freq)//2]))
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()

if __name__ == "__main__":
    main()