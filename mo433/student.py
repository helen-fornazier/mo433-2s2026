"""Step E11: the student family convnet(c) and the heads used during training (spec 2.1;
DECISOES 1-7, 32).

Architecture for width c, input 3 x 64 x 64 (report, tab:student):
  stem     conv 3x3 stride 1, BN, ReLU (no max-pool)           64 x 64, c
  stage 1  2 basic residual blocks, stride 1                   64 x 64, c
  stage 2  2 basic residual blocks, first with stride 2        32 x 32, 2c
  stage 3  idem                                                16 x 16, 4c
  stage 4  idem                                                 8 x  8, 8c
  pool     global average pooling                              8c features, no head
Blocks are post-activation (conv-BN-ReLU-conv-BN, add, ReLU); when the shape changes the
shortcut is a 1x1 stride-2 conv + BN. Convolutions: Kaiming normal (fan_out, ReLU); BatchNorm:
gamma=1, beta=0, and no zero-init of the last BN of each block (it would make the untrained
row A artificially weak). The only scaling rule between sizes is the width c.

``Student`` takes uint8 images (N, 3, 64, 64) and normalizes them inside, like the teacher
wrapper, so the same ``extract`` serves every encoder.

Same initialization in rows A, B and C (DECISOES 32): ``create_student(c, seed)`` seeds torch and
builds only the backbone, before anything else draws random numbers; the VICReg head (B) and
the distillation projector (C) are built afterwards, by ``create_vicreg_head`` /
``create_projector``. So A-s, B-s and C-s start from identical backbone weights.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from mo433 import config
from mo433.io import banner, data_dir, save_json, write_meta
from mo433.transforms import normalize


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ConvNet(nn.Module):
    """Backbone convnet(c): normalized float images -> 8c pooled features."""

    def __init__(self, c: int, blocks_per_stage: int = 2):
        super().__init__()
        widths = [c, 2 * c, 4 * c, 8 * c]
        self.stem = nn.Sequential(nn.Conv2d(3, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c),
                                  nn.ReLU(inplace=True))
        stages, in_ch = [], c
        for i, width in enumerate(widths):
            blocks = []
            for b in range(blocks_per_stage):
                blocks.append(BasicBlock(in_ch, width, stride=2 if (i > 0 and b == 0) else 1))
                in_ch = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = widths[-1]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.pool(self.stages(self.stem(x))).flatten(1)


class Student(nn.Module):
    """uint8 images (N, 3, 64, 64) -> ImageNet normalization -> ConvNet -> (N, 8c)."""

    def __init__(self, c: int):
        super().__init__()
        self.c = c
        self.backbone = ConvNet(c)
        self.feature_dim = self.backbone.feature_dim

    def forward(self, x_uint8):
        return self.backbone(normalize(x_uint8))


def create_student(c: int, seed: int) -> Student:
    """The student of width c with the initialization of seed ``seed`` (rows A, B and C)."""
    torch.manual_seed(seed)
    return Student(c)


def create_vicreg_head(c: int, hidden: int) -> nn.Module:
    """Row B: lightly's VICReg projection head, 8c -> h -> h -> h (DECISOES 6). Built after the
    student, so it does not change the student's initialization."""
    from lightly.models.modules.heads import VICRegProjectionHead
    return VICRegProjectionHead(input_dim=8 * c, hidden_dim=hidden, output_dim=hidden, num_layers=3)


def create_projector(c: int, hidden: int) -> nn.Module:
    """Row C: Linear(8c, h) -> BN -> GELU -> Linear(h, 384) (spec 3.5; DECISOES 7)."""
    return nn.Sequential(nn.Linear(8 * c, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
                         nn.Linear(hidden, config.TEACHER_DIM))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flops_per_image(model: nn.Module, input_size: int) -> int:
    """FLOPs of one forward pass on one image (torch counts a multiply-add as 2 FLOPs)."""
    model.eval()
    x = torch.zeros(1, 3, input_size, input_size, dtype=torch.uint8)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(x)
    return int(counter.get_total_flops())


def stage_shapes(student: Student) -> list:
    """Output shape of the stem and of each stage for one 64 px image."""
    shapes, x = [], normalize(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
                                          dtype=torch.uint8))
    with torch.no_grad():
        x = student.backbone.stem(x)
        shapes.append(list(x.shape[1:]))
        for stage in student.backbone.stages:
            x = stage(x)
            shapes.append(list(x.shape[1:]))
    return shapes


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    out = data_dir() / "student_info.json"
    banner("E11: student family convnet(c)")

    from mo433.teacher import load_teacher
    teacher = load_teacher(pretrained=False)          # architecture only, for params and FLOPs
    info = {"teacher": {"params": count_params(teacher.vit),
                        "flops_per_image": flops_per_image(teacher, config.IMAGE_SIZE)}}
    for c in config.WIDTHS:
        s = create_student(c, seed=0)
        info[f"c{c}"] = {
            "params": count_params(s),
            "feature_dim": s.feature_dim,
            "flops_per_image": flops_per_image(s, config.IMAGE_SIZE),
            "stage_shapes": stage_shapes(s),
            "vicreg_head_params": {h: count_params(create_vicreg_head(c, h))
                                   for h in config.TUNING_HIDDEN},
            "projector_params": {h: count_params(create_projector(c, h))
                                 for h in config.TUNING_HIDDEN},
            "teacher_to_student_params": round(info["teacher"]["params"] / count_params(s), 1),
        }
    save_json(info, out)
    write_meta(out)

    t = info["teacher"]
    print(f"teacher  {t['params'] / 1e6:6.2f} M params  {t['flops_per_image'] / 1e9:6.2f} GFLOPs/img (224 px)")
    for c in config.WIDTHS:
        r = info[f"c{c}"]
        print(f"c={c:<3}   {r['params'] / 1e6:6.3f} M params  {r['flops_per_image'] / 1e9:6.3f} GFLOPs/img"
              f"  dim {r['feature_dim']}  teacher/student {r['teacher_to_student_params']}x")
        print(f"         stages {r['stage_shapes']}")
        print(f"         VICReg head {r['vicreg_head_params']}  projector {r['projector_params']}")
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
