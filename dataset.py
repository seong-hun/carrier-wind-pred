import os
import numpy as np
from scipy import interpolate
import pickle as pkl

import tmc
from tmc.terrain import Terrain

import args


class WindShear:
    def __init__(self, vx, vy):
        self.v = np.vstack((vx, vy, 0))

    def get(self, pos):
        return self.v


class WindUpdraft:
    """
    Refereces:
        [1] https://doi.org/10.1109/AIS.2010.5547048
        [2] https://doi.org/10.2514/6.2006-1510

    wast and zi are selected from Table.2 in [2], in July and maximum wast.
        wast: 6.3, zi: 3962
    """
    ktable = np.array([
        [1.5352, 2.5826, -0.0113, 0.0008],
        [1.5265, 3.6054, -0.0176, 0.0005],
        [1.4866, 4.8356, -0.0320, 0.0001],
        [1.2042, 7.7904, 0.0848, 0.0001],
        [0.8816, 13.9720, 0.3404, 0.0001],
        [0.7067, 23.9940, 0.5689, 0.0002],
        [0.6189, 42.7965, 0.7157, 0.0001],
    ])
    rratiotable = np.array([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80])
    kfun = interpolate.interp1d(rratiotable, ktable, axis=0)

    def __init__(self, A, center=(0, 0), zi=3962, wast=6.3,
                 basewind=None, terrain=None):
        self.A = A
        self.zi = zi
        self.wast = wast
        self.basewind = basewind
        self.terrain = terrain
        self.center = np.asarray(center)

    def get(self, pos):
        pos = pos.squeeze()
        x, y = pos[:2]
        h = -pos[2] - self._get_base_height(x, y)
        xc, yc = self._get_center(h)
        w = self._get_w(x, y, h, xc, yc, zi=self.zi, wast=self.wast, A=self.A)
        return np.vstack((0, 0, -w))

    def _get_base_height(self, x, y):
        h = 0
        if self.terrain:
            h = h + self.terrain.get_height(x, y)
        return h

    def _get_center(self, z):
        """z is altitude"""
        ratio = 50 * np.tanh(12 * z / self.zi)
        center = self.center
        if self.basewind:
            pos = np.vstack((0, 0, -z))
            basevel = self.basewind.get(pos).squeeze()[:2]
            center = center + basevel * ratio
        return center

    def _get_w(self, x, y, z, xc, yc, A, zi, wast):
        """z is altitude"""
        zratio = z / zi
        r2 = max(10, 0.102 * zratio**(1/3) * (1 - 0.25 * zratio) * zi)
        r1 = (0.0011 * r2 + 0.14 if r2 < 600 else 0.8) * r2
        r = np.sqrt((x - xc)**2 + (y - yc)**2)
        k1, k2, k3, k4 = self.kfun(r1 / r2).squeeze()
        N = 1

        wbar = wast * zratio**(1/3) * (1 - 1.1 * zratio)
        wpeak = wbar * 3 * r2**2 * (r2 - r1) / (r2**3 - r1**3)

        is_zratio_in = zratio > 0.5 and zratio < 0.9

        if r > r1 and r < 2 * r2 and is_zratio_in:
            wD = wbar * 5 * np.pi / 12 * (zratio - 0.5) * np.sin(np.pi * r / r2)
        else:
            wD = 0

        if is_zratio_in:
            wemult = 1 - 2.5 * (zratio - 0.5)
        else:
            wemult = 1

        we = - wbar * N * np.pi * r2**2 / (A - N * np.pi * r2**2) * wemult
        w = (
            1 / (1 + np.abs(k1 * r / r2 + k3)**k2)
            + k4 * r / r2 + wD
        ) * (wpeak - we) + we
        return w


def preprocess_dataset():
    for i in range(args.FILENUM):
        # Terrain
        terrain = Terrain()

        # Map
        maploc = np.random.rand(2) * 2 * args.MAPSIZE
        xlim = maploc[0] + np.array((-args.MAPSIZE, args.MAPSIZE))
        ylim = maploc[1] + np.array((-args.MAPSIZE, args.MAPSIZE))
        xgrid = np.linspace(*xlim, args.GRIDSIZE)
        ygrid = np.linspace(*ylim, args.GRIDSIZE)
        xmap, ymap = np.meshgrid(xgrid, ygrid)
        A = float(np.diff(xlim) * np.diff(ylim))

        # Wind
        basewindvel = np.random.randn(2) * 9
        basewind = WindShear(*basewindvel)
        wind = WindUpdraft(A=A, basewind=basewind, terrain=terrain)

        hlist = [200, 250, 300, 350, 400]
        hmap = np.zeros_like(xmap)
        wmap = np.zeros((len(hlist),) + xmap.shape)
        for i, j in np.ndindex(xmap.shape):
            x = xmap[i, j]
            y = ymap[i, j]
            terrain_h = terrain.get_height(x, y)
            hmap[i, j] = terrain_h

            for k, h in enumerate(hlist):
                if h < terrain_h:
                    w = 0
                else:
                    pos = np.vstack((x, y, -h))
                    w = -wind.get(pos)[-1]
                wmap[k, i, j] = w

        quad_wmap, wmap = wmap[0], wmap[1:]

        # Save
        path = os.path.join("data", "video")
        video_id = f"VID_{i:05d}"
        data = []
        filename = f"{video_id}.vid"
        pkl.dump(data, open(os.path.join(path, filename), "wb"))


def plot(xmap, ymap, hmap, hlist, wmap):
    # Plotting
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.matshow(hmap, origin="lower")

    # 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    levels = np.linspace(np.nanmin(wmap), np.nanmax(wmap), 20)
    for h, w in zip(hlist, wmap):
        w[w < 0.01] = np.nan
        w = h + w
        ax.contourf(xmap, ymap, w, levels=h + levels)

    ax.set_xlim3d(xmap.min(), xmap.max())
    ax.set_ylim3d(ymap.min(), ymap.max())
    ax.set_zlim3d(0, max(hlist))

    plt.show()


if __name__ == "__main__":
    preprocess_dataset()
