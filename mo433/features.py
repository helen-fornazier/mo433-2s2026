"""Step E8: the single feature-extraction function, called identically on every encoder (spec 3.6).

Every encoder module (the teacher wrapper of mo433.teacher, the student wrapper of
mo433.student) takes raw uint8 images, (N, 3, 64, 64), and does its own deterministic
preprocessing inside: ImageNet normalization for the student, upsampling to 224 px plus
normalization for the teacher. ``extract`` therefore has no per-encoder branches and no
transform argument: there is no way to apply augmentation here.
"""

import numpy as np
import torch

from mo433.transforms import to_tensor


@torch.no_grad()
def extract(model, images, batch_size: int = 256, device=None) -> np.ndarray:
    """Features of ``images`` ((N, 64, 64, 3) uint8 array, memmap or tensor) as (N, d) float32.

    The model is put in eval mode (BatchNorm uses its running statistics, dropout is off) and
    run under no_grad, batch by batch, in the order of ``images``.
    """
    device = torch.device(device) if device else next(model.parameters()).device
    model.eval()
    feats = []
    for start in range(0, len(images), batch_size):
        x = to_tensor(np.ascontiguousarray(images[start:start + batch_size])).to(device)
        feats.append(model(x).float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32, copy=False)
