"""Step E7: the frozen teacher (row D), DINO ViT-S/16 from timm (spec 3.3; DECISOES 8-10).

``load_teacher`` returns a ``Teacher`` module that takes uint8 views at 64 px, (N, 3, 64, 64),
and returns the 384-d pooled feature: the [CLS] token after the final LayerNorm
(``num_classes=0``, ``global_pool="token"``). Inside it, the views are upsampled to 224 px
(bicubic) and normalized with the ImageNet statistics (mo433.transforms.teacher_preprocess),
so the teacher and the student receive the same view, each at its own resolution.

The teacher is never trained: all parameters have requires_grad=False, ``train()`` is a no-op
(it stays in eval mode, so dropout and drop-path are off), and ``forward`` runs under
no_grad, in fp16 autocast on CUDA.

``python -m mo433.teacher`` checks all of this and writes $MO433_DATA/teacher_info.json.
"""

import argparse
import hashlib

import torch
import torch.nn as nn

from mo433 import config
from mo433.io import banner, data_dir, save_json, write_meta
from mo433.transforms import teacher_preprocess, to_tensor


class Teacher(nn.Module):
    def __init__(self, vit: nn.Module, amp: bool = True):
        super().__init__()
        self.vit = vit
        self.amp = amp
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        """The teacher always stays in eval mode, whatever the caller asks."""
        return super().train(False)

    @property
    def feature_dim(self) -> int:
        return self.vit.num_features

    @torch.no_grad()
    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """(N, 3, 64, 64) uint8 -> (N, 384) float32."""
        x = teacher_preprocess(x_uint8)
        use_amp = self.amp and x.device.type == "cuda"
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_amp):
            return self.vit(x).float()


def load_teacher(device="cpu", pretrained: bool = True, amp: bool = True) -> Teacher:
    """The frozen DINO ViT-S/16. ``pretrained=False`` gives the same architecture with random
    weights, used as a negative control in the sanity check (step E10)."""
    import timm
    vit = timm.create_model(config.TEACHER_NAME, pretrained=pretrained, num_classes=0)
    return Teacher(vit, amp=amp).to(device)


def weights_sha256(model: nn.Module) -> str:
    """Hash of all parameters and buffers, in name order, to record exactly which weights ran."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def teacher_info(teacher: Teacher) -> dict:
    vit = teacher.vit
    cfg = vit.pretrained_cfg
    return {
        "name": config.TEACHER_NAME,
        "hf_hub_id": cfg.get("hf_hub_id"),
        "params": sum(p.numel() for p in vit.parameters()),
        "trainable_params": sum(p.numel() for p in vit.parameters() if p.requires_grad),
        "feature_dim": vit.num_features,
        "global_pool": vit.global_pool,
        "weights_sha256": weights_sha256(vit),
        # timm's own preprocessing for this checkpoint, to compare with ours (DECISOES 10-11)
        "pretrained_cfg": {k: cfg.get(k) for k in ("input_size", "interpolation", "mean", "std",
                                                    "crop_pct", "license")},
    }


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    out = data_dir() / "teacher_info.json"

    banner(f"E7: teacher {config.TEACHER_NAME}")
    teacher = load_teacher()
    info = teacher_info(teacher)

    # Determinism: eval mode (no dropout / drop-path) gives the same output twice,
    # and train() cannot switch it out of eval mode.
    x = to_tensor(torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8,
                                generator=torch.Generator().manual_seed(0)))
    teacher.train()
    a, b = teacher(x), teacher(x)
    info["checks"] = {
        "output_shape": list(a.shape),
        "deterministic": bool(torch.equal(a, b)),
        "stays_in_eval": not teacher.training and not teacher.vit.training,
        "preprocessing_matches_timm": (
            info["pretrained_cfg"]["interpolation"] == config.TEACHER_UPSAMPLE
            and tuple(info["pretrained_cfg"]["input_size"][1:]) == (config.TEACHER_RES,) * 2
            and tuple(info["pretrained_cfg"]["mean"]) == config.IMAGENET_MEAN
            and tuple(info["pretrained_cfg"]["std"]) == config.IMAGENET_STD),
    }
    save_json(info, out)
    write_meta(out)

    print(f"hub id        {info['hf_hub_id']}")
    print(f"params        {info['params'] / 1e6:.3f} M (trainable: {info['trainable_params']})")
    print(f"feature dim   {info['feature_dim']}  (global_pool={info['global_pool']!r})")
    print(f"timm cfg      {info['pretrained_cfg']}")
    print(f"weights hash  {info['weights_sha256'][:16]}...")
    for k, v in info["checks"].items():
        print(f"  {k:<28} {v}")
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
