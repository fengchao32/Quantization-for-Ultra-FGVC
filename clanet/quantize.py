from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from tqdm import tqdm

from awq.quantize.quantizer import pseudo_quantize_tensor
from gptq import Helper
from modelutils import find_layers
from quant import Quantizer, quantize

from .data import build_clanet_dataloader, resolve_dataset_spec
from .eval import evaluate_clanet
from .model import build_clanet, load_clanet_checkpoint


def _layer_module_map(model: nn.Module):
    return dict(model.named_modules())


def _calibration_batches(dataloader, device, max_images: int):
    seen = 0
    for images, _, _ in dataloader:
        remaining = max_images - seen
        if remaining <= 0:
            break
        if images.size(0) > remaining:
            images = images[:remaining]
        seen += images.size(0)
        yield images.to(device)


def _rtn_quantize_layer(layer: nn.Module, wbits: int, groupsize: int):
    weight = layer.weight.data
    orig_shape = weight.shape
    flat = weight.flatten(1).float() if isinstance(layer, nn.Conv2d) else weight.float()

    if groupsize > 0 and flat.shape[-1] % groupsize == 0:
        qweight = pseudo_quantize_tensor(flat, n_bit=wbits, q_group_size=groupsize)
    else:
        quantizer = Quantizer()
        quantizer.configure(wbits, perchannel=True, sym=False, mse=False)
        quantizer.find_params(flat, weight=True)
        qweight = quantize(flat, quantizer.scale, quantizer.zero, quantizer.maxq)

    layer.weight.data = qweight.reshape(orig_shape).to(dtype=weight.dtype, device=weight.device)


def _collect_statistics(
    model: nn.Module,
    true_model: nn.Module | None,
    layer_name: str,
    dataloader,
    device,
    nsamples: int,
    qep: bool,
):
    layer = _layer_module_map(model)[layer_name]
    true_layer = _layer_module_map(true_model)[layer_name] if qep else None
    helper = Helper(layer)

    current_inputs = []
    true_inputs = []

    def current_hook(_module, inputs):
        current_inputs.append(inputs[0].detach())

    def true_hook(_module, inputs):
        true_inputs.append(inputs[0].detach())

    handles = [layer.register_forward_pre_hook(current_hook)]
    if qep:
        handles.append(true_layer.register_forward_pre_hook(true_hook))

    try:
        for images in _calibration_batches(dataloader, device, nsamples):
            current_inputs.clear()
            true_inputs.clear()
            model(images)
            if qep:
                true_model(images)
                if not current_inputs or not true_inputs:
                    raise RuntimeError(f"Failed to capture QEP inputs for layer {layer_name}.")
                helper.add_batch_qep(current_inputs[0], true_inputs[0])
            else:
                if not current_inputs:
                    raise RuntimeError(f"Failed to capture calibration inputs for layer {layer_name}.")
                helper.add_batch(current_inputs[0])
    finally:
        for handle in handles:
            handle.remove()

    return helper


@torch.no_grad()
def quantize_clanet_model(
    model: nn.Module,
    dataloader,
    device,
    method: str = "gptq",
    wbits: int = 4,
    groupsize: int = -1,
    nsamples: int = 128,
    qep: bool = False,
    percdamp: float = 0.01,
    percdampqep: float = 1.0,
    perccorr: float = 0.5,
    skip_layers: Iterable[str] = (),
):
    if method not in {"gptq", "rtn"}:
        raise ValueError("CLA-NET quantization currently supports method='gptq' or 'rtn'.")

    model.to(device).eval()
    true_model = copy.deepcopy(model).to(device).eval() if qep else None
    layers = find_layers(model)
    skip_layers = set(skip_layers)

    for layer_name, layer in tqdm(layers.items(), desc=f"Quantizing CLA-NET ({method})"):
        if layer_name in skip_layers:
            continue
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue

        helper = _collect_statistics(
            model=model,
            true_model=true_model,
            layer_name=layer_name,
            dataloader=dataloader,
            device=device,
            nsamples=nsamples,
            qep=qep,
        )

        if qep:
            helper.run_weight_correct(layer, percdamp=percdampqep, perccorr=perccorr)

        if method == "gptq":
            helper.run_gptq(
                layer,
                percdamp=percdamp,
                wbits=wbits,
                groupsize=groupsize,
                actorder=False,
            )
        elif method == "rtn":
            _rtn_quantize_layer(layer, wbits=wbits, groupsize=groupsize)

        helper.free()

    if true_model is not None:
        true_model.cpu()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize CLA-NET with QEP/GPTQ/RTN.")
    parser.add_argument("--data", dest="dataset", default="STCAR", type=str)
    parser.add_argument("--data-root", default="data", type=str)
    parser.add_argument("--image-root", default=None, type=str)
    parser.add_argument("--anno-root", default=None, type=str)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--calib-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--eval-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--backbone", default="resnet50", type=str)
    parser.add_argument("--imagenet-checkpoint", default=None, type=str)
    parser.add_argument("--method", default="gptq", choices=["gptq", "rtn"])
    parser.add_argument("--qep", action="store_true")
    parser.add_argument("--wbits", default=4, type=int)
    parser.add_argument("--groupsize", default=-1, type=int)
    parser.add_argument("--nsamples", default=128, type=int)
    parser.add_argument("--percdamp", default=0.01, type=float)
    parser.add_argument("--percdampqep", default=1.0, type=float)
    parser.add_argument("--perccorr", default=0.5, type=float)
    parser.add_argument("--batch-size", "-b", default=8, type=int)
    parser.add_argument("--num-workers", "-j", default=4, type=int)
    parser.add_argument("--crop", dest="crop_resolution", default=384, type=int)
    parser.add_argument("--swap-num", default=[2, 2], nargs=2, type=int)
    parser.add_argument("--label-offset", default=1, type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--metrics-output",
        default=None,
        type=str,
        help="Optional JSON path for quantized acc1/acc3 and run metadata.",
    )
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
    calib_loader = build_clanet_dataloader(
        spec,
        split=args.calib_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_resolution=args.crop_resolution,
        shuffle=True,
        label_offset=args.label_offset,
    )
    model = build_clanet(
        num_classes=spec.num_classes,
        backbone=args.backbone,
        swap_num=tuple(args.swap_num),
        imagenet_checkpoint=args.imagenet_checkpoint,
    )
    load_clanet_checkpoint(model, args.checkpoint)

    device = torch.device(args.device)
    quantize_clanet_model(
        model,
        calib_loader,
        device=device,
        method=args.method,
        wbits=args.wbits,
        groupsize=args.groupsize,
        nsamples=args.nsamples,
        qep=args.qep,
        percdamp=args.percdamp,
        percdampqep=args.percdampqep,
        perccorr=args.perccorr,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), args.output)
    print(f"Saved quantized CLA-NET checkpoint to {args.output}")

    if args.eval or args.metrics_output:
        eval_loader = build_clanet_dataloader(
            spec,
            split=args.eval_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            crop_resolution=args.crop_resolution,
            label_offset=args.label_offset,
        )
        model.to(device)
        metrics = evaluate_clanet(model, eval_loader, device)
        print(
            "Quantized eval: acc1 %.6f acc2 %.6f acc3 %.6f"
            % (metrics["acc1"], metrics["acc2"], metrics["acc3"])
        )
        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            run_record = {
                "model": "clanet",
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "quantized_checkpoint": args.output,
                "method": args.method,
                "qep": args.qep,
                "wbits": args.wbits,
                "weight_only": True,
                "groupsize": args.groupsize,
                "nsamples": args.nsamples,
                "calib_split": args.calib_split,
                "eval_split": args.eval_split,
                "acc1": metrics["acc1"],
                "acc3": metrics["acc3"],
            }
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(run_record, handle, indent=2)
            print(f"Saved quantized metrics to {metrics_path}")


if __name__ == "__main__":
    main()
