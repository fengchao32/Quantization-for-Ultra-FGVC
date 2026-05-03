from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_root: Path
    anno_root: Path
    num_classes: int


DATASET_LAYOUTS = {
    "CUB": ("CUB/images", "CUB/anno", 200),
    "disease": ("plant_disease/images", "plant_disease/anno", 4),
    "disease_ICLR": ("plant_disease_ICLR/images", "plant_disease_ICLR/anno", 3),
    "STCAR": ("STCAR/images", "STCAR/anno", 196),
    "COTTON": ("COTTON/images", "COTTON/anno", 80),
    "Soybean200": ("soybean200/images", "soybean200/anno", 200),
    "Soybean2000": ("soybean2000/images", "soybean2000/anno", 1938),
    "R1": ("R1/images", "R1/anno", 198),
    "R3": ("R3/images", "R3/anno", 198),
    "R4": ("R4/images", "R4/anno", 198),
    "R5": ("R5/images", "R5/anno", 198),
    "R6": ("R6/images", "R6/anno", 198),
    "soybean_gene": ("soybean_gene/images", "soybean_gene/anno", 1110),
}


def resolve_dataset_spec(
    name: str,
    data_root: str | Path = "data",
    image_root: Optional[str | Path] = None,
    anno_root: Optional[str | Path] = None,
    num_classes: Optional[int] = None,
) -> DatasetSpec:
    if name in DATASET_LAYOUTS:
        default_images, default_annos, default_classes = DATASET_LAYOUTS[name]
    else:
        if image_root is None or anno_root is None or num_classes is None:
            raise ValueError(
                f"Unknown dataset {name!r}. Provide image_root, anno_root, and num_classes."
            )
        default_images = default_annos = ""
        default_classes = num_classes

    root = Path(data_root)
    return DatasetSpec(
        name=name,
        image_root=Path(image_root) if image_root else root / default_images,
        anno_root=Path(anno_root) if anno_root else root / default_annos,
        num_classes=num_classes or default_classes,
    )


def read_annotation(anno_file: str | Path):
    records = []
    with Path(anno_file).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed annotation line {line_no} in {anno_file}: {line!r}")
            image_name = " ".join(parts[:-1])
            label = int(parts[-1])
            records.append((image_name, label))
    return records


def build_eval_transform(crop_resolution: int = 384):
    return transforms.Compose(
        [
            transforms.Resize((crop_resolution, crop_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CLANetImageDataset(Dataset):
    """Original CLA-NET val/test annotation reader.

    Annotation rows are `relative/image/path label`. CLA-NET labels are 1-based in
    the txt files, so `label_offset=1` preserves the original `label - 1` behavior.
    """

    def __init__(
        self,
        image_root: str | Path,
        annotations: Iterable[tuple[str, int]] | str | Path,
        transform=None,
        label_offset: int = 1,
    ):
        self.image_root = Path(image_root)
        if isinstance(annotations, (str, Path)):
            annotations = read_annotation(annotations)
        self.records = list(annotations)
        self.transform = transform or build_eval_transform()
        self.label_offset = label_offset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rel_path, label = self.records[index]
        image_path = self.image_root / rel_path
        with image_path.open("rb") as handle:
            image = Image.open(handle).convert("RGB")
        image = self.transform(image)
        return image, label - self.label_offset, rel_path


def collate_fn4test(batch):
    images, labels, image_names = zip(*batch)
    return torch.stack(list(images), 0), list(labels), list(image_names)


def build_clanet_dataloader(
    spec: DatasetSpec,
    split: str = "val",
    batch_size: int = 16,
    num_workers: int = 4,
    crop_resolution: int = 384,
    shuffle: bool = False,
    label_offset: int = 1,
):
    anno_file = spec.anno_root / f"{split}.txt"
    dataset = CLANetImageDataset(
        spec.image_root,
        anno_file,
        transform=build_eval_transform(crop_resolution),
        label_offset=label_offset,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn4test,
        pin_memory=torch.cuda.is_available(),
    )
    setattr(dataloader, "total_item_len", len(dataset))
    setattr(dataloader, "num_cls", spec.num_classes)
    return dataloader
