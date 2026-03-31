"""Student skeleton for the CIFAR-10 patch puzzle transformer assignment.

Implement the TODO functions/classes marked with NotImplementedError.
"""

import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import wandb


@dataclass
class Config:
    project: str = "jigsaw-transformer"
    entity: str = "<your_wandb_username>"

    num_workers: int = 4
    image_size: int = 32
    patch_size: int = 8
    num_channels: int = 3

    batch_size: int = 16
    epochs: int = 200
    lr: float = 0.000128
    weight_decay: float = 0.000002

    embed_dim: int = 512
    num_heads: int = 8
    depth: int = 8
    mlp_ratio: float = 1.0
    dropout: float = 0.07

    log_every: int = 100
    num_visualizations: int = 8
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def images_to_patches(images: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """Convert images to a sequence of flattened patches.

    Args:
        images: Tensor of shape (B, C, H, W).
        patch_size: Patch edge length P.

    Returns:
        Tensor of shape (B, N, C * P * P), where N = (H / P) * (W / P).

    Notes:
        - Preserve patch order in row-major grid order.
        - You can implement this with tensor operations (e.g. view, permute, unfold) without explicit Python loops.
    """
    raise NotImplementedError("TODO: implement images_to_patches")


def patches_to_images(
    patches: torch.Tensor,
    patch_size: int = 8,
    image_size: int = 32,
    num_channels: int = 3,
) -> torch.Tensor:
    """Inverse operation of images_to_patches.

    Args:
        patches: Tensor of shape (B, N, C * P * P).
        patch_size: Patch edge length P.
        image_size: Output image size H = W.
        num_channels: Number of channels C.

    Returns:
        Tensor of shape (B, C, H, W).
    """
    raise NotImplementedError("TODO: implement patches_to_images")



def scramble_patches(patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scramble patch tokens and produce permutation labels.

    Args:
        patches: Tensor of shape (B, N, D).

    Returns:
        scrambled_patches: Tensor of shape (B, N, D).
        targets: Long tensor of shape (B, N) where targets[b, i] is the
            original position index of scrambled patch i.
    """
    raise NotImplementedError("TODO: implement scramble_patches")


def reorder_patches_from_predictions(scrambled_patches: torch.Tensor, pred_positions: torch.Tensor) -> torch.Tensor:
    """Reorder scrambled patches according to predicted slot indices.

    This helper is intentionally kept in the skeleton so students can inspect
    reconstruction quality during training.
    """
    bsz, num_patches, _ = scrambled_patches.shape
    reordered = torch.zeros_like(scrambled_patches)
    for b in range(bsz):
        for i in range(num_patches):
            pos = int(pred_positions[b, i].item())
            reordered[b, pos] = scrambled_patches[b, i]
    return reordered


def assert_patch_roundtrip(images: torch.Tensor, patch_size: int = 8) -> None:
    """Sanity checks for patch conversion.

    Call this after implementing images_to_patches and patches_to_images.
    """
    bsz, channels, height, width = images.shape
    assert height % patch_size == 0 and width % patch_size == 0

    patches = images_to_patches(images, patch_size)
    n_patches = (height // patch_size) * (width // patch_size)
    assert patches.shape == (bsz, n_patches, channels * patch_size * patch_size)

    reconstructed = patches_to_images(
        patches,
        patch_size=patch_size,
        image_size=height,
        num_channels=channels,
    )
    assert reconstructed.shape == images.shape
    assert torch.allclose(reconstructed, images)


def assert_scramble_and_reorder(patches: torch.Tensor) -> None:
    """Sanity checks for scramble_patches and reorder_patches_from_predictions."""
    scrambled, targets = scramble_patches(patches)
    assert scrambled.shape == patches.shape
    assert targets.shape == (patches.size(0), patches.size(1))

    # Check that each patch appears exactly once in the scrambled output.
    for b in range(patches.size(0)):
        for i in range(patches.size(1)):
            pos = (targets[b] == i).nonzero(as_tuple=True)[0]
            assert len(pos) == 1, f"Patch {i} appears {len(pos)} times in scrambled output"
            assert torch.allclose(scrambled[b, pos], patches[b, i]), "Scrambled patch does not match original"

    # Check that reordering recovers the original patches.
    reordered = reorder_patches_from_predictions(scrambled, targets)
    assert torch.allclose(reordered, patches), "Reordered patches do not match original"


def make_image_grid(original: torch.Tensor, scrambled: torch.Tensor, reconstructed: torch.Tensor, max_items: int = 8) -> Figure:
    """Create a matplotlib figure with original/scrambled/reconstructed images."""
    n = min(max_items, original.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        triplet = [original[i], scrambled[i], reconstructed[i]]
        titles = ["original", "scrambled", "reconstructed"]
        for j, (img, title) in enumerate(zip(triplet, titles)):
            img = img.detach().cpu().permute(1, 2, 0).clamp(0, 1)
            axes[i, j].imshow(img)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")

    plt.tight_layout()
    return fig


def prepare_dataloader(cfg: Config, train: bool = True) -> DataLoader:
    transform = T.Compose([
        T.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )


class TransformerBlock(nn.Module):
    """Single transformer encoder block for patch tokens.

    Required structure:
        1) LayerNorm
        2) MultiheadAttention (batch_first=True)
        3) Residual add
        4) LayerNorm
        5) MLP: Linear -> GELU -> Dropout -> Linear -> Dropout
        6) Residual add
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        # define 1) LayerNorm

        # define 2) MultiheadAttention

        # define 4) LayerNorm

        # define 5) MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            # todo
        )
        raise NotImplementedError("TODO: implement TransformerBlock.__init__")

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass.

        a) Apply LayerNorm and MultiheadAttention with self-attention
        b) Add residual connection (x + attention_output)
        c) Apply LayerNorm and MLP
        d) Add residual connection (x + mlp_output)
        e) Return final output, and optionally attention weights.

        Args:
            x: Token tensor of shape (B, N, D).
            return_attention: If True, also return attention weights
                with shape (B, num_heads, N, N). (Note: PyTorch's MultiheadAttention returns attn_weights in shape (B, N, N) when batch_first=True.)
        """
        raise NotImplementedError("TODO: implement TransformerBlock.forward")


class PatchPuzzleTransformer(nn.Module):
    """Transformer model for patch-position classification.

    Input:
        scrambled patches of shape (B, N, patch_dim)

    Output:
        logits of shape (B, N, N), where each token predicts one of N
        original patch positions.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
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
        """Forward pass.

        a) Embed patches
        b) Add positional embeddings (patch_embed + self.pos_embed)
        c) Pass through transformer blocks, optionally collecting attention maps per layer
        d) LayerNorm
        e) Linear head to produce logits of shape (B, N, N)
        f) Return logits, and optionally attention maps.

        Args:
            patches: (B, N, patch_dim)
            return_attention: whether to collect attention maps per block.
        """
        raise NotImplementedError("TODO: implement PatchPuzzleTransformer.forward")


def compute_loss_and_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute token-level CE loss and accuracy metrics.

    Args:
        logits: (B, N, N)
        targets: (B, N)

    Returns:
        loss: scalar tensor
        patch_acc: scalar tensor
        full_puzzle_acc: scalar tensor
        preds: (B, N)
    """
    # implement cross entropy loss

    # compute patch-level accuracy (percentage of correctly predicted patches)

    # compute full puzzle accuracy (percentage of samples in the batch where all patches are correct)

    raise NotImplementedError("TODO: implement compute_loss_and_accuracy")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    run: wandb.Run | None = None,
    split: str = "train",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict[str, float]:
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
        loss, patch_acc, full_acc, _ = compute_loss_and_accuracy(logits, targets)

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
            run.log(
                {
                    f"{split}/loss_step": loss.item(),
                    f"{split}/patch_acc_step": patch_acc.item(),
                    f"{split}/full_puzzle_acc_step": full_acc.item(),
                    "epoch": epoch,
                    "global_step": global_step,
                }
            )

    return {
        f"{split}/loss": total_loss / total_items,
        f"{split}/patch_acc": total_patch_acc / total_items,
        f"{split}/full_puzzle_acc": total_full_acc / total_items,
    }


@torch.no_grad()
def log_attention_example(model: nn.Module, loader: DataLoader, cfg: Config, run: wandb.Run, epoch: int, device: torch.device) -> None:
    """Log one qualitative attention/reconstruction batch to W&B."""
    model.eval()
    images, _ = next(iter(loader))
    images = images[:8].to(device)

    patches = images_to_patches(images, patch_size=cfg.patch_size)
    scrambled_patches, _ = scramble_patches(patches)

    logits, attention_maps = model(scrambled_patches, return_attention=True)
    preds = logits.argmax(dim=-1)
    reconstructed_patches = reorder_patches_from_predictions(scrambled_patches, preds)

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

    fig_triplet = make_image_grid(images, scrambled_img, reconstructed_img, max_items=8)
    run.log({"examples/triplet": wandb.Image(fig_triplet), "epoch": epoch})
    plt.close(fig_triplet)

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


def main() -> None:
    cfg = Config()
    if cfg.entity == "<your_wandb_username>":
        raise ValueError("Please set your W&B username in the Config.entity field before running.")
    with wandb.init(project=cfg.project, entity=cfg.entity, config=vars(cfg)) as run:
        cfg = Config(**run.config)

        set_seed(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = prepare_dataloader(cfg, train=True)
        val_loader = prepare_dataloader(cfg, train=False)

        # Sanity check for patch conversion before training.
        test_image = next(iter(train_loader))[0][:4]
        assert_patch_roundtrip(test_image, patch_size=cfg.patch_size)
        assert_scramble_and_reorder(images_to_patches(test_image, patch_size=cfg.patch_size))

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

            run.log({**train_metrics, **val_metrics, "epoch": epoch})
            log_attention_example(model, val_loader, cfg, run, epoch, device)

            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_metrics['train/loss']:.4f} | "
                f"train patch acc {train_metrics['train/patch_acc']:.2%} | "
                f"val patch acc {val_metrics['val/patch_acc']:.2%} | "
                f"val full puzzle acc {val_metrics['val/full_puzzle_acc']:.2%}"
            )

        torch.save(model.state_dict(), f"{run.dir}/patch_puzzle_transformer.pth")
        with open(f"{run.dir}/config.json", "w", encoding="utf-8") as f:
            json.dump(vars(cfg), f, indent=4)


if __name__ == "__main__":
    main()