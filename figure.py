import torch
import pickle as pkl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import train
import network
import args
from dataset import UpdraftDataset, ToTensor


E = network.Embedder()
G = network.Generator()

DATANUM = "20200514_2113"
train.load_model(E, DATANUM)
train.load_model(G, DATANUM)

E.eval()
G.eval()

raw_dataset = UpdraftDataset(
    root=args.DATASET_PATH,
    transform=ToTensor(),
)
dataset = DataLoader(raw_dataset, batch_size=len(raw_dataset))

for batch_num, (i, video) in enumerate(dataset):
    t = video[:, -1, ...]  # [B, 2, C, W, H]
    video = video[:, :-1, ...]  # [B, K, 2, C, W, H]
    dims = video.shape

    # Calculate average encoding vector for video
    e_in = video.reshape(dims[0] * dims[1], dims[2], dims[3], dims[4], dims[5])  # [BxK, 2, C, W, H]
    x, y = e_in[:, 0, ...], e_in[:, 1, ...]
    e_vectors = E(x, y).reshape(dims[0], dims[1], -1)  # B, K, len(e)
    e_hat = e_vectors.mean(dim=1)

    # Generate frame using landmarks from frame t
    x_t, y_t = t[:, 0, ...], t[:, 1, ...]
    x_hat = G(y_t, e_hat)

    # goodlist = range(len(x_hat))
    goodlist = [2, 3, 10, 18]

    fig, axes = plt.subplots(
        len(goodlist), 4, figsize=(5, 5),
        squeeze=False, subplot_kw={"xticks": [], "yticks": []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i, idx in enumerate(goodlist):
        base_img = -video[idx, 0, 0, -1, ...].detach().numpy()
        real_img = -x_t[idx, -1, ...].detach().numpy()
        fake_img = -x_hat[idx, -1, ...].detach().numpy()
        landmark = -y_t[idx, -1, ...].detach().numpy()

        vmin = real_img.min()
        vmax = real_img.max()
        kwargs = dict(vmin=vmin, vmax=vmax, cmap="jet")
        axes[i, 0].imshow(base_img, **kwargs)
        axes[i, 1].imshow(landmark, **kwargs)
        axes[i, 2].imshow(fake_img, **kwargs)
        axes[i, 3].imshow(real_img, **kwargs)

    axes[0, 0].set_title("Base")
    axes[0, 1].set_title("Landmark")
    axes[0, 2].set_title("Estimated")
    axes[0, 3].set_title("Real")

    plt.show()
