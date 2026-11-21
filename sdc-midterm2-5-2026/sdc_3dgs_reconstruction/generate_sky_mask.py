import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class SkySegmenter:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        model_id = "nvidia/segformer-b5-finetuned-ade-640-640"
        print(f"[Init] Loading {model_id}...")
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def process_image(self, image_path, output_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        upscaled_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # ADE20K label index 2 is sky.
        prediction = torch.argmax(upscaled_logits, dim=1).squeeze(0).cpu().numpy()
        mask = (prediction == 2).astype(np.uint8) * 255
        cv2.imwrite(str(output_path), mask)


def default_paths(track, data_root):
    dataset_dir = data_root / track
    return {
        "input_dir": dataset_dir / "data" / "image",
        "output_dir": dataset_dir / "result" / "sky_masks",
    }


def parse_args():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate binary sky masks for 3DGS training.")
    parser.add_argument("--track", default="Track2", choices=["Track1", "Track2", "Track3"])
    parser.add_argument("--data-root", type=Path, default=repo_dir / "data")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = default_paths(args.track, args.data_root)
    input_dir = args.input_dir or paths["input_dir"]
    output_dir = args.output_dir or paths["output_dir"]

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"[Info] Found {len(image_files)} images in {input_dir}")
    print(f"[Info] Writing masks to {output_dir}")

    segmenter = SkySegmenter(device=args.device)
    for image_path in tqdm(image_files, desc="Processing sky masks"):
        output_path = output_dir / f"{image_path.stem}.png"
        if output_path.exists() and not args.overwrite:
            continue
        segmenter.process_image(image_path, output_path)

    print(f"[Done] Masks saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
