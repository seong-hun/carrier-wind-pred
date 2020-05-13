from dataset import preprocess_dataset

# Meta Learning
"""https://github.com/grey-eye/talking-heads"""
raw_dataset = UpdraftDataset()
dataset = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    for batch_num, (i, video) in enumerate(dataset):
        # video [B, K+1, 2, C, W, H]

        t = video[:, -1, ...]  # [B, 2, C, W, H]
        video = video[:, :-1, ...]  # [B, K, 2, C, W, H]
        dims = video.shape

        # Calculate average encoding vector for video
        e_in = video.reshape(dims[0] * dims[1], *dims[2:])  # [BxK, 2, C, W, H]
        x, y = e_in[:, 0, ...], e_in[:, 1, ...]  # [BxK, C, W, H]
        e_vectors = E(x, y).reshape(dims[0], dims[1], -1)  # [B, K, len(e)]
        e_hat = e_vectors.mean(dim=1)
