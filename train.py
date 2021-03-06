import os
import sys
import logging
import pickle as pkl
from datetime import datetime
import click

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import args
import network
from network.vgg import vgg_face
from dataset import UpdraftDataset, ToTensor, NormHeight


@click.group()
def main():
    if not os.path.isdir(args.LOG_DIR):
        os.makedirs(args.LOG_DIR)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(args.LOG_DIR, f"{datetime.now():%Y%m%d}.log"),
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


@main.command()
def meta():
    # Meta Learning
    """https://github.com/grey-eye/talking-heads"""

    run_start = datetime.now()
    logging.info("===== META-TRAINING =====")
    logging.info(f"Training using dataset located in {args.DATASET_PATH}")
    raw_dataset = UpdraftDataset(
        root=args.DATASET_PATH,
        transform=transforms.Compose([
            NormHeight(),
            ToTensor(),
        ])
    )
    dataset = DataLoader(raw_dataset, batch_size=args.BATCHSIZE, shuffle=True)

    E = network.Embedder()
    G = network.Generator()
    D = network.Discriminator(len(raw_dataset))
    criterion_E_G = network.LossEG(args.FEED_FORWARD)
    criterion_D = network.LossD()

    optimizer_E_G = optim.Adam(
        params=list(E.parameters()) + list(G.parameters()),
        lr=args.LEARNING_RATE_E_G
    )
    optimizer_D = optim.Adam(
        params=D.parameters(),
        lr=args.LEARNING_RATE_D
    )

    logging.info(
        f"Epochs: {args.EPOCHS} "
        f"Batches: {len(dataset)} "
        f"Batch Size: {args.BATCHSIZE}")

    for epoch in range(args.EPOCHS):
        epoch_start = datetime.now()

        E.train()
        G.train()
        D.train()

        for batch_num, (i, video) in enumerate(dataset):
            # batch_start = datetime.now()

            # video [B, K+1, 2, C, W, H]
            # Remove index channel from all video (CHANNEL = 4)
            video = video[..., :args.CHANNEL, :, :]

            # t: Target
            # video: for Encoder only
            t = video[:, -1, ...]  # [B, 2, C, W, H]
            video = video[:, :-1, ...]  # [B, K, 2, C, W, H]
            dims = video.shape

            # Calculate average encoding vector for video
            e_in = video.reshape(dims[0] * dims[1], *dims[2:])  # [BxK, 2, C, W, H]
            x, y = e_in[:, 0, ...], e_in[:, 1, ...]  # [BxK, C, W, H]
            e_vectors = E(x, y).reshape(dims[0], dims[1], -1)  # [B, K, len(e)]
            e_hat = e_vectors.mean(dim=1)  # [B, len(e)]

            # Target frame and landmark
            # x_t, y_t: [B, C, W, H]
            x_t, y_t = t[:, 0, ...], t[:, 1, ...]

            """
            Following:
            https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models/blob/e461da8dc54ed76ae9f5087c77f93f6e12a83bf0/train.py#L52
            """
            # Train E and G
            optimizer_E_G.zero_grad()
            optimizer_D.zero_grad()

            x_hat = G(y_t, e_hat)  # [B, C, W, H]
            r_x_hat, D_hat_res_list = D(x_hat, y_t, i)

            with torch.no_grad():
                r_x, D_res_list = D(x_t, y_t, i)

            loss_E_G = criterion_E_G(
                x_t, x_hat, r_x_hat,
                D_res_list, D_hat_res_list, e_hat, D.W[:, i].transpose(1, 0))
            loss_E_G.backward(retain_graph=False)

            # for k, v in E.named_parameters():
            #     if torch.isnan(v.grad).any():
            #         breakpoint()

            # for k, v in G.named_parameters():
            #     if torch.isnan(v.grad).any():
            #         breakpoint()

            optimizer_E_G.step()

            # Train D
            optimizer_E_G.zero_grad()
            optimizer_D.zero_grad()

            x_hat = x_hat.detach()
            r_x_hat, D_hat_res_list = D(x_hat, y_t, i)
            r_x, D_res_list = D(x_t, y_t, i)

            loss_D = criterion_D(r_x, r_x_hat)
            loss_D.backward(retain_graph=False)

            # for k, v in D.named_parameters():
            #     if torch.isnan(v.grad).any():
            #         breakpoint()

            optimizer_D.step()

            # Train D once again
            optimizer_D.zero_grad()

            x_hat.detach().requires_grad_()
            r_x_hat, D_hat_res_list = D(x_hat, y_t, i)
            r_x, D_res_list = D(x_t, y_t, i)

            loss_D2 = criterion_D(r_x, r_x_hat)
            loss_D2.backward(retain_graph=False)

            loss_D += loss_D2

            # for k, v in D.named_parameters():
            #     if torch.isnan(v.grad).any():
            #         breakpoint()

            optimizer_D.step()

            # Show progress
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{args.EPOCHS}: "
                    f"[{batch_num + 1}/{len(dataset)}] | "
                    f"Time: {datetime.now() - run_start} | "
                    f"Loss_E_G = {loss_E_G.item():.4f} "
                    f"Loss_D = {loss_D.item():.4f}"
                )
                logging.debug(
                    f"D(x) = {r_x.mean().item():.4f} "
                    f"D(x_hat) = {r_x_hat.mean().item():.4f}")

        save_model(E, run_start)
        save_model(G, run_start)
        save_model(D, run_start)
        epoch_end = datetime.now()
        logging.info(
            f"Epoch {epoch + 1} finished in {epoch_end - epoch_start}. ")


@main.command()
def pre():
    """
    Pre-training the wind classifier to construct a loss function
    """
    run_start = datetime.now()
    logging.info("===== PRE-TRANING =====")
    logging.info(f"Traning using dataset located in {args.DATASET_PATH}")
    raw_dataset = UpdraftDataset(
        root=args.DATASET_PATH,
        transform=transforms.Compose([
            NormHeight(),
            ToTensor(),
        ])
    )
    dataset = DataLoader(
        raw_dataset, batch_size=args.PRE_BATCHSIZE, shuffle=True)

    model = vgg_face(pretrained=False, in_channels=4)
    criterion = nn.CrossEntropyLoss()

    LR = 5e-4
    params = [
        {"params": model.features.parameters(), "lr": LR / 10},
        {"params": model.classifier.parameters()}
    ]
    optimizer = optim.Adam(params, lr=LR)

    breakpoint()
    logging.info(
        f"Epochs: {args.PRE_EPOCHS} "
        f"Batches: {len(dataset)} "
        f"Batch Size: {args.PRE_BATCHSIZE}")

    for epoch in range(args.PRE_EPOCHS):
        epoch_start = datetime.now()

        model.train()

        for batch_num, (index, video) in enumerate(dataset):
            # video: [B, K, 2, C+1, W, H]
            video = video[..., 0, :args.CHANNEL, :, :]  # [B, K, C, W, H]
            video = video.transpose(0, 1)

            y = index

            loss_val = 0
            for vid in video:
                y_pred = model(vid)

                optimizer.zero_grad()
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                loss_val += loss.item()

            # Show progress
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{args.PRE_EPOCHS}: "
                    f"[{batch_num + 1}/{len(dataset)}] | "
                    f"Time: {datetime.now() - run_start} | "
                    f"Loss = {loss_val/len(video):.4f} "
                )

        save_model(model, run_start)
        epoch_end = datetime.now()
        logging.info(
            f"Epoch {epoch + 1} finished in {epoch_end - epoch_start}."
        )


def save_model(model, time_for_name=None):
    if time_for_name is None:
        time_for_name = datetime.now()

    model.eval()

    if not os.path.exists(args.MODELS_DIR):
        os.makedirs(args.MODELS_DIR)
    filename = f'{type(model).__name__}_{time_for_name:%Y%m%d_%H%M}.pth'
    torch.save(
        model.state_dict(),
        os.path.join(args.MODELS_DIR, filename)
    )

    model.train()

    logging.info(f'Model saved: {filename}')


def load_model(model, continue_id):
    filename = f'{type(model).__name__}_{continue_id}.pth'
    state_dict = torch.load(
        os.path.join(args.MODELS_DIR, filename),
        map_location=model.device
    )
    model.load_state_dict(state_dict)
    return model


def save_image(filename, data):
    if not os.path.isdir(args.GENERATED_DIR):
        os.makedirs(args.GENERATED_DIR)

    data = data.clone().detach()
    pkl.dump(data, open(filename, "wb"))


if __name__ == "__main__":
    main()
