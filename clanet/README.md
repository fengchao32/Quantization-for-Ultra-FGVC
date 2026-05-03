# CLA-NET Migration

This package ports the usable CLA-NET pieces from `/home/fengchao/CLA-NET-main` into
the QEP workspace.

## Original Conventions

- Model: `models/LoadModel.py::MainModel`.
- Forward output: `[class_logits, swap_logits, covariance_logits, feature]` when
  `use_cdrm=True`.
- Evaluation: original `test.py` uses `outputs[0]` for top-k accuracy.
- Per-image report: `--acc_report` saves a `result_gather_*.pt` dict containing
  `top1_cat`, `top2_cat`, `top3_cat`, score values, and `label`.
- Annotation format: `anno/{train,val,test}.txt`, each row as
  `relative/image/path label`. Labels are 1-based and are converted to zero-based
  by subtracting `1`.
- Test transform: resize to `384 x 384`, convert to tensor, normalize with
  ImageNet mean/std.

## Evaluate

Run from the repository root:

```bash
PYTHONPATH=src python -m clanet.eval \
  --data STCAR \
  --data-root /home/fengchao/CLA-NET-main/data \
  --checkpoint /path/to/clanet.pth \
  --acc-report
```

## Quantize

The quantization entry reuses this repository's `gptq.Helper`, which already
supports `nn.Conv2d` and `nn.Linear`.

```bash
PYTHONPATH=src python -m clanet.quantize \
  --data STCAR \
  --data-root /home/fengchao/CLA-NET-main/data \
  --checkpoint /path/to/clanet.pth \
  --output /path/to/clanet-qep-gptq-w4.pth \
  --method gptq \
  --qep \
  --wbits 4 \
  --nsamples 128 \
  --eval
```
