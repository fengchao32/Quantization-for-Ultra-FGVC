from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn
from torchvision import models

from .asofmax import AngleLinear


@dataclass
class CLANetConfig:
    num_classes: int
    backbone: str = "resnet50"
    use_cdrm: bool = True
    swap_num: tuple[int, int] = (2, 2)
    cls_2: bool = True
    cls_2xmul: bool = False
    feature_enhance: bool = False
    use_asoftmax: bool = False
    feature_dim: int = 2048


def _load_torchvision_backbone(name: str, imagenet_checkpoint: Optional[str] = None):
    if name not in dir(models):
        raise ValueError(
            f"Backbone {name!r} is not in torchvision.models. "
            "Install pretrainedmodels and extend _load_backbone if you need SENet variants."
        )
    model = getattr(models, name)(weights=None)
    if imagenet_checkpoint:
        state_dict = torch.load(imagenet_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    return model


def _load_backbone(name: str, imagenet_checkpoint: Optional[str] = None):
    if name in dir(models):
        return _load_torchvision_backbone(name, imagenet_checkpoint)

    try:
        import pretrainedmodels
    except ImportError as exc:
        raise ImportError(
            f"Backbone {name!r} requires the optional 'pretrainedmodels' package."
        ) from exc

    model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=None)
    if imagenet_checkpoint:
        model.load_state_dict(torch.load(imagenet_checkpoint, map_location="cpu"))
    return model


class CLANet(nn.Module):
    """CLA-NET model ported from /home/fengchao/CLA-NET-main/models/LoadModel.py.

    The forward output order is intentionally kept compatible with the original:
    classification logits, optional swap logits, optional covariance logits, and the
    pooled feature vector.
    """

    def __init__(self, config: CLANetConfig, imagenet_checkpoint: Optional[str] = None):
        super().__init__()
        self.config = config
        self.use_cdrm = config.use_cdrm
        self.num_classes = config.num_classes
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_asoftmax

        if self.use_cdrm and not (config.cls_2 or config.cls_2xmul):
            raise ValueError("CLA-NET CDRM requires either cls_2 or cls_2xmul.")

        backbone = _load_backbone(config.backbone, imagenet_checkpoint)
        if config.backbone in {
            "resnet50",
            "se_resnet50",
            "se_resnet101",
            "se_resnext101_32x4d",
        }:
            self.model = nn.Sequential(*list(backbone.children())[:-2])
        elif config.backbone == "senet154":
            self.model = nn.Sequential(*list(backbone.children())[:-3])
        else:
            raise ValueError(
                f"Backbone {config.backbone!r} is not configured for CLA-NET feature extraction."
            )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(config.feature_dim, self.num_classes, bias=False)

        if self.use_cdrm:
            if config.cls_2:
                self.classifier_swap = nn.Linear(config.feature_dim, 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(
                    config.feature_dim, 2 * self.num_classes, bias=False
                )

            self.blockN = config.swap_num[0] * config.swap_num[1]
            cov_dim = 16 if config.feature_enhance else 9
            self.classifier_cova = nn.Linear(
                config.feature_dim, self.blockN * cov_dim, bias=False
            )

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(
                config.feature_dim, self.num_classes, bias=False
            )

    def forward(self, x, last_cont=None):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        out = [self.classifier(x)]

        if self.use_cdrm:
            out.append(self.classifier_swap(x))
            out.append(self.classifier_cova(x))

        out.append(x)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out


MainModel = CLANet


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_clanet_checkpoint(
    model: nn.Module,
    checkpoint: str,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
):
    """Load original CLA-NET checkpoints, including DataParallel 'module.' keys."""

    path = Path(checkpoint)
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = strip_module_prefix(state)

    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state.items() if key in model_state and model_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)

    if strict and (missing or unexpected or len(compatible) != len(state)):
        skipped = sorted(set(state) - set(compatible))
        raise RuntimeError(
            f"Failed strict CLA-NET checkpoint load. missing={missing}, "
            f"unexpected={unexpected}, skipped={skipped[:20]}"
        )

    return {
        "loaded": sorted(compatible),
        "missing": missing,
        "unexpected": unexpected,
        "skipped": sorted(set(state) - set(compatible)),
    }


def build_clanet(
    num_classes: int,
    backbone: str = "resnet50",
    swap_num: tuple[int, int] = (2, 2),
    use_cdrm: bool = True,
    cls_2: bool = True,
    cls_2xmul: bool = False,
    feature_enhance: bool = False,
    use_asoftmax: bool = False,
    imagenet_checkpoint: Optional[str] = None,
):
    config = CLANetConfig(
        num_classes=num_classes,
        backbone=backbone,
        use_cdrm=use_cdrm,
        swap_num=swap_num,
        cls_2=cls_2,
        cls_2xmul=cls_2xmul,
        feature_enhance=feature_enhance,
        use_asoftmax=use_asoftmax,
    )
    return CLANet(config, imagenet_checkpoint=imagenet_checkpoint)
