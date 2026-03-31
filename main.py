"""
# CIFAR Patch Puzzle with a Transformer

Below, you will build a small PyTorch prototype for a jigsaw-like vision task on CIFAR-10.([docs.wandb.ai](https://docs.wandb.ai/models/integrations/pytorch?utm_source=chatgpt.com))the patches, and train a transformer to predict where each patch originally belonged.

This version is intentionally compact and fully working, so you can run it once, verify that the pipeline works, and then remove selected parts later for students to fill in.

---

## 1. What the task is

In the following, you will formulate the puzzle as a **64-way?** No — here it is only a **16-way patch-position classification** problem, because a `32x32` image split into `8x8` patches gives a `4x4` grid, hence **16 patches** in total.

For each scrambled patch, the model should predict its **original position index** in `{0, ..., 15}`.

That means:

* **input**: 16 scrambled image patches
* **output**: 16 class predictions, one per patch
* **target**: the original location of each patch before shuffling

---

## 2. What you need to install

In the following, you will need PyTorch, torchvision, and Weights & Biases.

```bash
pip install torch torchvision wandb matplotlib
```

You will also need to log in to W&B once:

```bash
wandb login
```

W&B runs are initialized with `wandb.init()`, metrics and media are logged with `run.log(...)`, and gradient/parameter tracking can be enabled with `run.watch(...)`. ([docs.wandb.ai](https://docs.wandb.ai/models/integrations/pytorch?utm_source=chatgpt.com))

---

## 3. Full prototype

In the following, you will implement the entire training pipeline in one file.
"""

import json
import math
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import wandb


# ============================================================
# 1) Configuration
# ============================================================
@dataclass
class Config:
    project: str = "jigsaw-transformer"
    entity: str = "nihermann"
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4

    image_size: int = 32
    patch_size: int = 8
    num_channels: int = 3

    embed_dim: int = 256
    num_heads: int = 2
    depth: int = 2
    mlp_ratio: float = 1.0
    dropout: float = 0.1

    log_every: int = 100
    num_visualizations: int = 8
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2) Helpers for patching and visualization
# ============================================================
def images_to_patches(images: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    images: (B, C, H, W)
    returns patches: (B, N, C*P*P)
    where N = number of patches
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0
    gh = H // patch_size
    gw = W // patch_size
    num_patches = gh * gw

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # (B, C, gh, gw, P, P)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # (B, gh, gw, C, P, P)
    patches = patches.view(B, num_patches, C * patch_size * patch_size)
    # (B, gh*gw, C*P*P)
    return patches


def patches_to_images(patches: torch.Tensor, patch_size: int = 8, image_size: int = 32, num_channels: int = 3) -> torch.Tensor:
    """
    patches: (B, N, C*P*P)
    returns images: (B, C, H, W)
    """
    B, N, _ = patches.shape
    gh = image_size // patch_size
    gw = image_size // patch_size
    assert N == gh * gw

    patches = patches.view(B, gh, gw, num_channels, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    images = patches.view(B, num_channels, image_size, image_size)
    return images


def scramble_patches(patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    patches: (B, N, D)

    Returns:
        scrambled_patches: (B, N, D)
        targets: (B, N) where targets[b, i] = original position of scrambled patch i
    """
    B, N, _ = patches.shape
    scrambled = torch.empty_like(patches)
    targets = torch.empty(B, N, dtype=torch.long, device=patches.device)

    for b in range(B):
        perm = torch.randperm(N, device=patches.device)
        scrambled[b] = patches[b, perm]
        targets[b] = perm

    return scrambled, targets


def reorder_patches_from_predictions(scrambled_patches: torch.Tensor, pred_positions: torch.Tensor) -> torch.Tensor:
    """
    scrambled_patches: (B, N, D)
    pred_positions: (B, N) predicted original position for each scrambled patch

    Builds a reconstructed image by placing each scrambled patch into its predicted slot.
    If multiple patches claim the same slot, the last one wins. This is acceptable for a toy prototype.
    """
    B, N, _ = scrambled_patches.shape
    reordered = torch.zeros_like(scrambled_patches)
    for b in range(B):
        for i in range(N):
            pos = pred_positions[b, i].item()
            reordered[b, pos] = scrambled_patches[b, i]
    return reordered


def make_image_grid(original: torch.Tensor, scrambled: torch.Tensor, reconstructed: torch.Tensor, max_items: int = 8) -> plt.Figure:
    """Create a matplotlib figure with original / scrambled / reconstructed images."""
    n = min(max_items, original.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        triplet = [original[i], scrambled[i], reconstructed[i]]
        titles = ["original", "scrambled", "reconstructed"]
        for j, (img, title) in enumerate(zip(triplet, titles)):
            img = img.detach().cpu().permute(1, 2, 0)
            img = img.clamp(0, 1)
            axes[i, j].imshow(img)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")

    plt.tight_layout()
    return fig


# ============================================================
# 3) Dataset
# ============================================================
def prepare_dataloader(cfg: Config, train: bool = True) -> DataLoader:
    transform = T.Compose([
        T.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
    return loader


# ============================================================
# 4) Transformer model
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(
            h, h, h,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            return x, attn_weights
        return x


class PatchPuzzleTransformer(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 8, in_chans: int = 3,
                 embed_dim: int = 128, depth: int = 4, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_patches)

    def forward(self, patches: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        x = self.patch_embed(patches)
        x = x + self.pos_embed

        attention_maps = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x = block(x)

        x = self.norm(x)
        logits = self.head(x)  # (B, N, N)

        if return_attention:
            return logits, attention_maps
        return logits


# ============================================================
# 5) Loss and metrics
# ============================================================
def compute_loss_and_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, float, float, torch.Tensor]:
    """
    logits:  (B, N, N)
    targets: (B, N)
    Each scrambled patch predicts one of N original positions.
    """
    B, N, _ = logits.shape
    loss = F.cross_entropy(logits.view(B * N, N), targets.view(B * N))

    preds = logits.argmax(dim=-1)
    patch_acc = (preds == targets).float().mean()

    full_puzzle_acc = (preds == targets).all(dim=1).float().mean()
    return loss, patch_acc, full_puzzle_acc, preds


# ============================================================
# 6) One training / validation step
# ============================================================
def run_epoch(
        model: nn.Module, loader: DataLoader, cfg: Config, optimizer: torch.optim.Optimizer | None = None, 
        epoch: int = 0, run: wandb.Run | None = None, split: str = "train", device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_patch_acc = 0.0
    total_full_acc = 0.0
    total_items = 0

    for step, (images, _) in enumerate(loader):
        images = images.to(device)

        patches = images_to_patches(images, patch_size=cfg.patch_size)
        scrambled_patches, targets = scramble_patches(patches)

        logits = model(scrambled_patches)
        loss, patch_acc, full_acc, preds = compute_loss_and_accuracy(logits, targets)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_patch_acc += patch_acc.item() * bs
        total_full_acc += full_acc.item() * bs
        total_items += bs

        global_step = epoch * len(loader) + step
        if run is not None and global_step % cfg.log_every == 0:
            run.log({
                f"{split}/loss_step": loss.item(),
                f"{split}/patch_acc_step": patch_acc.item(),
                f"{split}/full_puzzle_acc_step": full_acc.item(),
                "epoch": epoch,
                "global_step": global_step,
            })

    metrics = {
        f"{split}/loss": total_loss / total_items,
        f"{split}/patch_acc": total_patch_acc / total_items,
        f"{split}/full_puzzle_acc": total_full_acc / total_items,
    }
    return metrics


# ============================================================
# 7) Attention logging for one mini-batch
# ============================================================
@torch.no_grad()
def log_attention_example(model: nn.Module, loader: DataLoader, cfg: Config, run: wandb.Run, epoch: int, device: torch.device) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:8].to(device)

    patches = images_to_patches(images, patch_size=cfg.patch_size)
    scrambled_patches, targets = scramble_patches(patches)

    logits, attention_maps = model(scrambled_patches, return_attention=True)
    preds = logits.argmax(dim=-1)

    reconstructed_patches = reorder_patches_from_predictions(scrambled_patches, preds)

    original_img = images
    scrambled_img = patches_to_images(
        scrambled_patches,
        patch_size=cfg.patch_size,
        image_size=cfg.image_size,
        num_channels=cfg.num_channels,
    )
    reconstructed_img = patches_to_images(
        reconstructed_patches,
        patch_size=cfg.patch_size,
        image_size=cfg.image_size,
        num_channels=cfg.num_channels,
    )
    
    fig_triplet = make_image_grid(original_img, scrambled_img, reconstructed_img, max_items=8)
    run.log({"examples/triplet": wandb.Image(fig_triplet), "epoch": epoch})
    plt.close(fig_triplet)

    # Log one attention map from the last layer, averaged over heads.
    # attn shape: (B, num_heads, N, N)
    last_attn = attention_maps[-1][0].mean(dim=0).detach().cpu()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(last_attn)
    ax.set_title("Last-layer attention (mean over heads)")
    ax.set_xlabel("key patch index")
    ax.set_ylabel("query patch index")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    run.log({"attention/last_layer_mean": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


# ============================================================
# 8) Training loop with W&B
# ============================================================
def main() -> None:
    cfg = Config()
    with wandb.init(project=cfg.project, entity=cfg.entity, config=vars(cfg)) as run:
        cfg = Config(**run.config)

        set_seed(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = prepare_dataloader(cfg, train=True)

        val_loader = prepare_dataloader(cfg, train=False)

        model = PatchPuzzleTransformer(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.num_channels,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        run.watch(model, log="gradients", log_freq=200)

        for epoch in range(cfg.epochs):
            train_metrics = run_epoch(model, train_loader, cfg, optimizer=optimizer, epoch=epoch, run=run, split="train", device=device)
            val_metrics = run_epoch(model, val_loader, cfg, optimizer=None, epoch=epoch, run=run, split="val", device=device)

            run.log({
                **train_metrics,
                **val_metrics,
                "epoch": epoch,
            })

            log_attention_example(model, val_loader, cfg, run, epoch, device)

            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_metrics['train/loss']:.4f} | "
                f"train patch acc {train_metrics['train/patch_acc']:.2%} | "
                f"val patch acc {val_metrics['val/patch_acc']:.2%} | "
                f"val full puzzle acc {val_metrics['val/full_puzzle_acc']:.2%}"
            )
        torch.save(model.state_dict(), f"{run.dir}/patch_puzzle_transformer.pth")
        json.dump(vars(cfg), open(f"{run.dir}/config.json", "w"), indent=4)


if __name__ == "__main__":
    main()

"""

## 4. What is implemented in the prototype

In the following, you will notice that the code is split into a few small building blocks.

### 4.1 Configuration

In the first section, you will define all important hyperparameters in one place:

* image size
* patch size
* transformer depth
* embedding dimension
* optimizer settings
* W&B logging frequency

This makes later ablations easier.

### 4.2 Patch extraction

In the next section, you will convert each CIFAR image into a sequence of 16 patch tokens.
Each patch is flattened and later projected into the transformer embedding space.

### 4.3 Patch scrambling

In the following helper, you will randomly permute the 16 patches per image.
The permutation also defines the label:
for each scrambled patch, the target is the patch's original position.

### 4.4 Transformer model

In the model section, you will implement a very small transformer encoder:

* linear patch embedding
* learnable positional embeddings
* several self-attention blocks
* one classification head per token

The head predicts one of 16 possible original patch locations.

### 4.5 Loss and metrics

In the loss section, you will train the model with token-wise cross-entropy.
You will also compute two useful metrics:

* **patch accuracy**: how many single patch positions are correct
* **full puzzle accuracy**: how many whole images are solved perfectly

### 4.6 W&B logging

In the experiment section, you will log:

* training and validation loss
* patch accuracy
* full puzzle accuracy
* gradient statistics via `run.watch(...)`
* example image triplets: original, scrambled, reconstructed
* one average attention map from the last transformer layer

This is enough to make the run visually interesting without adding too much complexity. W&B recommends initializing runs with `wandb.init()`, using `run.log(...)` for metrics and media, and `run.watch(...)` to track gradients and parameters. ([docs.wandb.ai](https://docs.wandb.ai/models/integrations/pytorch?utm_source=chatgpt.com))

---

## 5. What students should understand from this setup

In the following, students should focus on three central ideas.

### 5.1 The task is not image generation

They are not asked to predict pixels.
Instead, they are asked to predict a **permutation-related label**: the original position of each patch.

### 5.2 The transformer sees a sequence of visual tokens

The image is turned into a sequence of patches, so the vision task becomes a sequence modeling task.
That is the central transformer idea in a minimal setting.

### 5.3 Attention can be inspected

Because the model processes patch tokens with self-attention, students can inspect whether some patches attend more strongly to others in later layers.
This is one of the main reasons this task is a good teaching example.

---

## 6. Good places to remove code later

In the following, you could delete a few carefully chosen pieces and ask students to complete them.

### Option A: remove the patch extraction helper

Students implement:

* `images_to_patches(...)`
* maybe also `patches_to_images(...)`

### Option B: remove the scrambling helper

Students implement:

* random permutation of patch tokens
* target generation from the permutation

### Option C: remove the transformer block

Students implement:

* LayerNorm
* MultiheadAttention
* MLP with residual connections

### Option D: remove the metric computation

Students implement:

* cross-entropy over patch tokens
* patch accuracy
* full puzzle accuracy

### Option E: remove the attention logging function

Students implement:

* extracting one attention matrix
* averaging over heads
* logging it as a W&B image

A good teaching sequence is often:

1. give the full prototype,
2. let students run it,
3. remove one or two helper sections,
4. keep the rest fixed so they can debug locally.

---

## 7. Simple extensions

In the following, students could extend the prototype in small steps:

* compare 8x8 patches with 4x4 patches
* compare shallow vs deeper transformers
* check whether attention becomes sharper over epochs
* compare CIFAR-10 classes separately
* enforce valid permutations during reconstruction

---

## 8. Expected behavior

In the following, you should expect patch accuracy to increase well before full-puzzle accuracy becomes high.
That is normal:
solving all 16 patch positions correctly at once is much harder than getting many individual positions right.

For a toy prototype, it is enough if:

* loss decreases clearly,
* patch accuracy rises above chance,
* reconstructed images begin to look more structured,
* attention maps show non-uniform patterns.

---

## 9. One small note on correctness

In the following prototype, the reconstruction function does not enforce a perfect one-to-one assignment between patches and output slots.
This keeps the code simple.
For teaching, that is usually the right tradeoff.
Later, students could improve this with a matching-based decoding step.
"""
