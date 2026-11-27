"""Export the released CoC Xception checkpoint to a pinned ONNX artifact.

Development/setup utility only; it is not used at analysis runtime. Run in a
Python 3.10 environment with torch 2.2.x plus the ScreenDeepfakeBench source
checkout. Example:

    python scripts/export_coc_xception_onnx.py \
      --repo-root path/to/ScreenDeepfakeBench \
      --checkpoint path/to/checkpoints/coc.pth \
      --output models/coc_xception_screen_deepfake.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = args.repo_root.resolve() / "training"
    if not (training / "networks" / "xception.py").is_file():
        raise SystemExit(f"not a ScreenDeepfakeBench checkout: {args.repo_root}")
    sys.path.insert(0, str(training))
    from networks.xception import Xception

    class LogitsOnly(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _features = self.model(x)
            return out

    model = Xception({"mode": "original", "num_classes": 2, "inc": 3, "dropout": False})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(
        {key.removeprefix("backbone."): value for key, value in checkpoint.items()},
        strict=True,
    )
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        LogitsOnly(model).eval(),
        torch.zeros(1, 3, 299, 299),
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image_bgr_normalized"],
        output_names=["logits"],
        dynamic_axes=None,
    )
    print(f"wrote {args.output}")
    print(f"sha256 {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
