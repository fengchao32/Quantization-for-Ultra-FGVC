from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .data import build_clanet_dataloader, resolve_dataset_spec
from .model import build_clanet, load_clanet_checkpoint


@torch.no_grad()
def evaluate_clanet(model, dataloader, device, save_report: str | None = None):
    model.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    report = {}

    for images, labels, image_names in tqdm(dataloader, desc="Evaluating CLA-NET"):
        images = images.to(device)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)

        outputs = model(images)
        logits = outputs[0]
        k = min(3, logits.size(1))
        top_values, top_indices = torch.topk(logits, k)

        correct1 += torch.sum(top_indices[:, 0] == labels_tensor).item()
        if k >= 2:
            correct2 += torch.sum(
                (top_indices[:, :2] == labels_tensor.unsqueeze(1)).any(dim=1)
            ).item()
        else:
            correct2 += torch.sum(top_indices[:, 0] == labels_tensor).item()
        if k >= 3:
            correct3 += torch.sum(
                (top_indices[:, :3] == labels_tensor.unsqueeze(1)).any(dim=1)
            ).item()
        else:
            correct3 += torch.sum(
                (top_indices[:, :k] == labels_tensor.unsqueeze(1)).any(dim=1)
            ).item()

        total += labels_tensor.numel()

        if save_report is not None:
            top_values = top_values.detach().cpu().tolist()
            top_indices = top_indices.detach().cpu().tolist()
            for name, cats, vals, label in zip(image_names, top_indices, top_values, labels):
                padded_cats = cats + [-1] * (3 - len(cats))
                padded_vals = vals + [float("nan")] * (3 - len(vals))
                report[name] = {
                    "top1_cat": padded_cats[0],
                    "top2_cat": padded_cats[1],
                    "top3_cat": padded_cats[2],
                    "top1_val": padded_vals[0],
                    "top2_val": padded_vals[1],
                    "top3_val": padded_vals[2],
                    "label": label,
                }

    if total == 0:
        raise ValueError("Evaluation dataloader is empty.")

    metrics = {
        "acc1": correct1 / total,
        "acc2": correct2 / total,
        "acc3": correct3 / total,
    }

    if save_report is not None:
        Path(save_report).parent.mkdir(parents=True, exist_ok=True)
        torch.save(report, save_report)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a migrated CLA-NET model.")
    parser.add_argument("--data", dest="dataset", default="STCAR", type=str)
    parser.add_argument("--data-root", default="data", type=str)
    parser.add_argument("--image-root", default=None, type=str)
    parser.add_argument("--anno-root", default=None, type=str)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--backbone", default="resnet50", type=str)
    parser.add_argument("--imagenet-checkpoint", default=None, type=str)
    parser.add_argument("--batch-size", "-b", default=16, type=int)
    parser.add_argument("--num-workers", "-j", default=4, type=int)
    parser.add_argument("--crop", dest="crop_resolution", default=384, type=int)
    parser.add_argument("--swap-num", default=[2, 2], nargs=2, type=int)
    parser.add_argument("--label-offset", default=1, type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--acc-report", action="store_true")
    parser.add_argument("--report-path", default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = resolve_dataset_spec(
        args.dataset,
        data_root=args.data_root,
        image_root=args.image_root,
        anno_root=args.anno_root,
        num_classes=args.num_classes,
    )
    dataloader = build_clanet_dataloader(
        spec,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_resolution=args.crop_resolution,
        label_offset=args.label_offset,
    )
    model = build_clanet(
        num_classes=spec.num_classes,
        backbone=args.backbone,
        swap_num=tuple(args.swap_num),
        imagenet_checkpoint=args.imagenet_checkpoint,
    )
    load_info = load_clanet_checkpoint(model, args.checkpoint)
    if load_info["skipped"]:
        print(f"Skipped {len(load_info['skipped'])} incompatible checkpoint tensors.")

    device = torch.device(args.device)
    model.to(device)
    report_path = args.report_path
    if args.acc_report and report_path is None:
        stem = Path(args.checkpoint).stem
        report_path = f"result_gather_{stem}.pt"

    metrics = evaluate_clanet(model, dataloader, device, report_path)
    print(
        "--------acc1 %.6f--------\n--------acc2 %.6f--------\n--------acc3 %.6f--------"
        % (metrics["acc1"], metrics["acc2"], metrics["acc3"])
    )
    if report_path:
        print(f"Saved per-image report to {report_path}")


if __name__ == "__main__":
    main()
