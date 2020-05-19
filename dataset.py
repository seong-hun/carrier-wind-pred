import os
import sys
import random
import logging
import functools
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import count
import numpy as np
from scipy import interpolate
import pickle as pkl
from tqdm import tqdm, trange
from numba import njit

import torch
from torch.utils.data import Dataset

import tmc
from tmc.terrain import Terrain

import args


class WindShear:
    def __init__(self, vx, vy):
        self.v = np.array([vx, vy, 0])

    def get(self, x, y, z):
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
        # Height of the updraft center
        self.hc = self.get_terrain_height(*center)

    def get(self, x, y, z):
        basewindvel = self.basewind.get(x, y, z)
        h = -z
        hr = h - self.get_terrain_height(x, y).squeeze()
        # hr = 2 * hr
        hw = h - self.hc
        xc, yc = self.get_center(hw)
        assert hr >= 0 and hw >= 0

        wx, wy = basewindvel[:2]

        zratio, r, r1, r2 = _get_params(x, y, hr, xc, yc, wx, wy, zi=self.zi)
        ks = self.kfun(r1 / r2).squeeze()
        w = _get_updraft(
            zratio, r, r1, r2, ks=ks, wast=self.wast, A=self.A)
        return np.array([0, 0, -w]) + basewindvel

    def is_valid_height(self, x, y, h):
        return h >= self.get_terrain_height(x, y) and h >= self.hc

    def get_terrain_height(self, x, y):
        h = 0
        if self.terrain:
            h = h + self.terrain.get_height(x, y)
        return h

    def get_center(self, hw):
        """z is altitude"""
        # Expected maximum horizontal wind speed: 6 m/s
        # Expected maximum shifted center of the updraft: 150 m
        ratio = 300 / 6 * np.tanh(hw / 300)
        center = self.center
        if self.basewind:
            h = hw + self.hc
            basewindvel = self.basewind.get(0, 0, -h)[:2]
            center = center + basewindvel * ratio
        return center


@njit
def _get_params(x, y, z, xc, yc, wx, wy, zi):
    ws = np.sqrt(wx ** 2 + wy ** 2)  # Wind speed
    wa = np.arctan2(wy, wx)  # Wind angle
    xr, yr = x - xc, y - yc
    xrr = np.cos(wa) * xr + np.sin(wa) * yr
    yrr = -np.sin(wa) * xr + np.cos(wa) * yr
    a, b = 1 + 0.03 * ws, 1
    c = (xrr / a)**2 + (yrr / b)**2
    x = np.sqrt(c) * a + xc
    y = yc

    zratio = z / zi
    r2 = max(10., 0.102 * zratio**(1/3) * (1 - 0.25 * zratio) * zi)
    r1 = (0.0011 * r2 + 0.14 if r2 < 600 else 0.8) * r2
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    return zratio, r, r1, r2


@njit
def _get_updraft(zratio, r, r1, r2, ks, A, wast):
    """z is altitude"""
    k1, k2, k3, k4 = ks
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
    def __init__(self, root, extension=".vid", transform=None):
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(extension)
        ]
        self.files.sort()
        self.transform = transform
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.files[idx]
        data = pkl.load(open(path, "rb"))
        data = random.sample(data, args.K + 1)

        data_array = []
        for d in data:
            x = d["frame"]
            y = d["landmark"]
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
            assert torch.is_tensor(x)
            data_array.append(torch.stack((x, y)))
        data_array = torch.stack(data_array)

        return idx, data_array


class NormHeight:
    def __init__(self):
        self.factor = np.array([7, 7, 7, 600, 1]).reshape([-1, 1, 1])

    def __call__(self, image):
        return image / self.factor


class ToTensor:
    """Change a numpy array of an image [C, W, H] to a tensor"""
    def __call__(self, image):
        return torch.from_numpy(image).float()


def generate():
    max_workers = os.cpu_count()
    logging.info(
        f"Worker: {max_workers}"
    )

    with ProcessPoolExecutor(max_workers) as p:
        list(tqdm(
            p.map(gen_video, range(args.VIDEONUM)),
            total=args.VIDEONUM
        ))


def gen_video(video_id):
    np.random.seed(video_id)
    t0 = datetime.now()

    # Terrain
    terrain = Terrain()

    MAPSIZE = args.MAPSIZE

    hrlist = np.linspace(0, 300, args.FRAMENUM)  # relative heights

    MAPSIZE = args.MAPSIZE
    # Define an identity
    for cnt in count(1):
        # Map (random location)
        mapcenter = np.random.uniform(-MAPSIZE, MAPSIZE, size=2)

        # mapcenter = np.random.rand(2) * 2 * MAPSIZE
        xlim = mapcenter[0] + np.array((-MAPSIZE, MAPSIZE)) / 2
        ylim = mapcenter[1] + np.array((-MAPSIZE, MAPSIZE)) / 2
        xgrid = np.linspace(*xlim, args.GRIDSIZE)
        ygrid = np.linspace(*ylim, args.GRIDSIZE)
        xmap, ymap = np.meshgrid(xgrid, ygrid)

        # Wind (random base velocity, and random updraft center)
        basewindvel = np.random.randn(2) * 9
        basewind = WindShear(*basewindvel)
        windcenter = (
            mapcenter
            + np.random.uniform(-MAPSIZE/5, MAPSIZE/5, size=2))
        wind = WindUpdraft(A=MAPSIZE**2, center=windcenter,
                           basewind=basewind, terrain=terrain)

        # Check validity of the wind center
        h = terrain.get_height(*windcenter)
        hw = h + hrlist[-1] - wind.hc
        center_at_h = wind.get_center(hw)
        validmap = np.vectorize(wind.is_valid_height)(xmap, ymap, h)
        lim = np.vstack((xlim, ylim)).T
        if np.sum(validmap) / validmap.size > 0.4:
            if (center_at_h < lim[1]).all() and (center_at_h > lim[0]).all():
                break

    logging.info(
        f"Video {video_id + 1:02d}/{args.VIDEONUM:02d} | "
        f"find a valid configuration in {cnt} tries..."
    )

    # C: vx, vy, vz, h, bool
    frames = np.zeros(hrlist.shape + (5, ) + xmap.shape)  # [K, C, W, H]
    landmarks = np.zeros(hrlist.shape + (5, ) + xmap.shape)  # [K, C, W, H]

    hlist = terrain.get_height(*windcenter) + hrlist
    for k, h in enumerate(hlist):
        r = np.random.rand()
        xc, yc = wind.get_center(h - wind.hc)
        c = (1 - r) * xc + r * yc
        d = args.LANDMARK_RANGE**2 * ((1 - r)**2 + r**2)**2
        for i, j in np.ndindex(xmap.shape):
            x, y = xmap[i, j], ymap[i, j]

            is_valid = wind.is_valid_height(x, y, h)
            if is_valid:
                wvel = wind.get(x, y, -h)
            else:
                wvel = args.NANVALUE

            frames[k, :3, i, j] = wvel
            frames[k, 3, i, j] = h
            frames[k, 4, i, j] = is_valid

            # Landmark
            is_landmark = ((1 - r) * x + r * y - c)**2 < d
            if is_landmark and is_valid:
                lmvel = wvel
                is_landmark = True
            else:
                lmvel = args.NANVALUE
                is_landmark = False

            landmarks[k, :3, i, j] = lmvel
            landmarks[k, 3, i, j] = h
            landmarks[k, 4, i, j] = is_landmark

        if (k + 1) % (int(args.FRAMENUM / 10) or 1) == 0 or k == 0:
            logging.info(
                f"Video {video_id + 1:02d}/{args.VIDEONUM:02d} "
                f"[{k + 1:03d}/{args.FRAMENUM:03d}] | "
                f"Time: {datetime.now() - t0}"
            )

    # plot(terrain, xmap, ymap, frames, landmarks)

    # Save
    path = os.path.join("data", "video")
    if not os.path.isdir(path):
        os.makedirs(path)

    data = []
    for frame, landmark in zip(frames, landmarks):
        data.append({
            "frame": frame,
            "landmark": landmark,
        })

    filename = f"VID_{video_id:02d}.vid"
    path = os.path.join(path, filename)
    pkl.dump(data, open(path, "wb"))

    logging.info(
        f"Video {video_id + 1:02d}/{args.VIDEONUM:02d} [Finished] | "
        f"Time: {datetime.now() - t0} | "
        f"Saved: {path}"
    )


def plot(terrain, xmap, ymap, frames, landmarks):
    import plotly.colors as colors
    import plotly.graph_objects as go

    data = []

    # Terrain
    zmap = np.vectorize(terrain.get_height)(xmap, ymap)
    terrain_trace = go.Surface(
        x=xmap, y=ymap, z=zmap,
        colorscale="Darkmint", showscale=False, opacity=1,
    )
    data.append(terrain_trace)

    # Wind map & landmarks
    for k in np.linspace(0, len(frames) - 1, 5, dtype="int"):
        # Wind map
        windmap = -frames[k, 2] + frames[k, 3]
        # windmap[windmap <= frames[k, 3]] = np.nan
        windmap[frames[k, 4] == 0] = np.nan
        windmap_trace = go.Surface(
            x=xmap, y=ymap, z=windmap,
            surfacecolor=-frames[k, 2],
            coloraxis="coloraxis",
        )
        if k > 0:
            windmap_trace.opacityscale = [
                [0, 0], [0.2, 0.7], [0.5, 0.9], [1, 1]]
        else:
            windmap_trace.opacity = 0.5
        data.append(windmap_trace)

        # Landmark
        landmarkmap = -landmarks[k, 2] + landmarks[k, 3]
        landmarkmap[landmarks[k, 4] == 0] = np.nan
        landmark_trace = go.Surface(
            x=xmap, y=ymap, z=landmarkmap,
            surfacecolor=-landmarks[k, 2],
            coloraxis="coloraxis",
        )
        if k > 0:
            landmark_trace.opacityscale = [
                [0, 0.2], [0.1, 0.9], [1, 1]]
        else:
            landmark_trace.opacity = 0

        data.append(landmark_trace)

    fig = go.Figure(data=data)
    fig.update_layout(
        title="Title",
        autosize=False, width=800, height=800,
        margin={"l": 65, "r": 50, "b": 65, "t": 90},
        template="none",
        coloraxis={"colorscale": "Jet", "cmin": 0, "cmax": 6},
    )
    fig.show()


def _plot(xmap, ymap, hmap, hlist, wmap):
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


def main():
    if not os.path.isdir(args.LOG_DIR):
        os.makedirs(args.LOG_DIR)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(args.LOG_DIR, f"{datetime.now():%Y%m%d}.log"),
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("===== DATA-GENERATION =====")
    logging.info(
        f"Videos: {args.VIDEONUM}, "
        f"Frames: {args.FRAMENUM}, "
        f"Image Size: {args.GRIDSIZE}x{args.GRIDSIZE}"
    )

    generate()
    # gen_video(1)


if __name__ == "__main__":
    main()
