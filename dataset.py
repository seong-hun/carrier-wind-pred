import os
import random
import numpy as np
from scipy import interpolate
import pickle as pkl
from tqdm import tqdm, trange
import click

import torch
from torch.utils.data import Dataset

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
        basevel = self.basewind.get(pos)
        return np.vstack((0, 0, -w)) + basevel

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


class UpdraftDataset(Dataset):
    def __init__(self, root, extension=".vid", shuffle=False,
                 transform=None, shuffle_frames=False):
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(extension)
        ]
        self.files.sort()
        self.transform = transform
        self.length = len(self.files)
        self.indices = np.arange(self.length)

        if shuffle:
            random.shuffle(self.indices)

        self.shuffle_frames = shuffle_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path = self.files[real_idx]
        data = pkl.load(open(path, "rb"))
        if self.shuffle_frames:
            random.shuffle(data)

        data_array = []
        for d in data:
            x = d["frame"]
            y = d["landmarks"]
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
            assert torch.is_tensor(x)
            data_array.append(torch.stack((x, y)))
        data_array = torch.stack(data_array)

        return real_idx, data_array


class ToTensor:
    """Change a numpy array of an image [C, W, H] to a tensor"""
    def __call__(self, image):
        return torch.from_numpy(image).float()


@click.group()
def main():
    pass


@main.command()
def generate():
    # Terrain
    terrain = Terrain()

    PIXSIZE = 2 * args.MAPSIZE / (args.GRIDSIZE - 1)
    hlist = np.linspace(200, 500, args.FRAMENUM)  # frame

    for file_id in trange(args.VIDEONUM):
        # Map (random location)
        maploc = np.random.rand(2) * 2 * args.MAPSIZE
        xlim = maploc[0] + np.array((-args.MAPSIZE, args.MAPSIZE))
        ylim = maploc[1] + np.array((-args.MAPSIZE, args.MAPSIZE))
        xgrid = np.linspace(*xlim, args.GRIDSIZE)
        ygrid = np.linspace(*ylim, args.GRIDSIZE)
        xmap, ymap = np.meshgrid(xgrid, ygrid)
        A = float(np.diff(xlim) * np.diff(ylim))

        # Wind (random base velocity, and random updraft center)
        basewindvel = np.random.randn(2) * 9
        basewind = WindShear(*basewindvel)
        center = maploc + 0.2 * np.random.randn(2) * args.MAPSIZE
        wind = WindUpdraft(A=A, basewind=basewind, terrain=terrain)

        frames = np.zeros(hlist.shape + (4, ) + xmap.shape)  # [K, C, W, H]
        landmarks = np.zeros(hlist.shape + (4, ) + xmap.shape)  # [K, C, W, H]

        for k, h in enumerate(hlist):
            r = np.random.rand()
            d = (1 - r) * center[0] + r * center[1]
            for i, j in np.ndindex(xmap.shape):
                x = xmap[i, j]
                y = ymap[i, j]
                terrain_h = terrain.get_height(x, y)

                if h < terrain_h:
                    wvel = np.zeros((3, 1))
                else:
                    pos = np.vstack((x, y, -h))
                    wvel = wind.get(pos)

                frames[k, :, i, j] = np.vstack((h, wvel)).squeeze()

                if abs((1 - r) * x + r * y - d) < PIXSIZE * np.sqrt(2) / 2:
                    lmvel = wvel
                else:
                    lmvel = np.zeros((3, 1))

                landmarks[k, :, i, j] = np.vstack((h, lmvel)).squeeze()

        # Save
        path = os.path.join("data", "video")
        if not os.path.isdir(path):
            os.makedirs(path)

        video_id = f"VID_{file_id:05d}"
        data = []
        for frame, landmark in zip(frames, landmarks):
            data.append({
                "frame": frame,
                "landmarks": landmark,
            })
        filename = f"{video_id}.vid"
        pkl.dump(data, open(os.path.join(path, filename), "wb"))


@main.command()
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
    main()
    # preprocess_dataset()
    # pass
