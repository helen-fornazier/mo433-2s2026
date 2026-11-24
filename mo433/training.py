"""Training components shared by rows B and C (spec 3.4, 3.5, 3.9; report Secs. 5.1, 5.3).

The throughput measurement (step E14) and the training loop (step E15) use exactly these pieces,
so what is timed is what is trained.

* ``ViewBank`` holds the K pre-generated views of the training images (uint8) and, for row C,
  the teacher's cached features of the same views, on the training device. A batch is gathered
  by indexing: there is no data loader and no CPU work per step.
* ``epoch_batches`` draws, for one epoch, a permutation of the images and the views of each
  image (C: one view, uniform; B: two distinct views, uniform), from a generator seeded by the
  training seed. Both rows drop the last incomplete batch, so they take the same number of steps.
* ``RowB`` (student + VICReg head, lightly's VICRegLoss) and ``RowC`` (student + projector,
  1 - cos to the teacher's feature of the same view) both return the loss of a batch.
* ``train_step``: one AdamW step in fp16 autocast with gradient scaling, channels-last input.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mo433.student import create_projector, create_student, create_vicreg_head


class ViewBank:
    def __init__(self, views, targets=None, device="cpu"):
        """``views``: (N, K, 64, 64, 3) uint8 array; ``targets``: (N, K, 384) array or None."""
        # np.array copies (memmaps are read-only), then the copy moves to the device
        self.views = torch.from_numpy(np.array(views)).to(device)
        self.targets = None if targets is None else torch.from_numpy(np.array(targets)).to(device)
        self.n_images, self.k_views = self.views.shape[:2]
        self.device = self.views.device

    def images(self, idx, k) -> torch.Tensor:
        """Views k[j] of images idx[j], as (B, 3, 64, 64) uint8 in channels-last layout."""
        return self.views[idx, k].permute(0, 3, 1, 2)

    def teacher(self, idx, k) -> torch.Tensor:
        return self.targets[idx, k].float()


def epoch_batches(positions, k_views: int, batch_size: int, n_views: int,
                  generator: torch.Generator):
    """Yield (idx, k) per batch: idx are image positions, k is (B,) for one view per image or
    (B, 2) with two distinct views. The last incomplete batch is dropped."""
    positions = torch.as_tensor(positions)
    n = len(positions)
    order = positions[torch.randperm(n, generator=generator)]
    first = torch.randint(k_views, (n,), generator=generator)
    if n_views == 1:
        ks = first
    else:
        offset = torch.randint(1, k_views, (n,), generator=generator)
        ks = torch.stack([first, (first + offset) % k_views], dim=1)
    for start in range(0, n - batch_size + 1, batch_size):
        yield order[start:start + batch_size], ks[start:start + batch_size]


def steps_per_epoch(n_images: int, batch_size: int) -> int:
    return n_images // batch_size


class RowB(nn.Module):
    """VICReg: the two views go through the student and the head separately (lightly's example)."""
    n_views = 2

    def __init__(self, c: int, hidden: int, seed: int):
        super().__init__()
        self.student = create_student(c, seed)
        self.head = create_vicreg_head(c, hidden)
        from lightly.loss import VICRegLoss
        self.criterion = VICRegLoss()

    def loss(self, bank: ViewBank, idx, k) -> torch.Tensor:
        z_a = self.head(self.student(bank.images(idx, k[:, 0])))
        z_b = self.head(self.student(bank.images(idx, k[:, 1])))
        return self.criterion(z_a.float(), z_b.float())


class RowC(nn.Module):
    """Distillation: 1 - cos(g(f(x)), t(x)), t(x) the cached teacher feature of the same view."""
    n_views = 1

    def __init__(self, c: int, hidden: int, seed: int):
        super().__init__()
        self.student = create_student(c, seed)
        self.projector = create_projector(c, hidden)

    def loss(self, bank: ViewBank, idx, k) -> torch.Tensor:
        pred = self.projector(self.student(bank.images(idx, k)))
        return 1 - F.cosine_similarity(pred.float(), bank.teacher(idx, k), dim=1).mean()


def create_row(row: str, c: int, hidden: int, seed: int) -> nn.Module:
    model = {"B": RowB, "C": RowC}[row](c, hidden, seed)
    return model.to(memory_format=torch.channels_last)


def train_step(model, optimizer, scaler, bank, idx, k) -> torch.Tensor:
    """One optimizer step; returns the loss as a detached tensor (no GPU sync)."""
    with torch.autocast(device_type=bank.device.type, dtype=torch.float16,
                        enabled=bank.device.type == "cuda"):
        loss = model.loss(bank, idx.to(bank.device), k.to(bank.device))
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()
