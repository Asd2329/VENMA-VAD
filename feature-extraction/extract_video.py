import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Represent the video content for anomaly detection."
FRAME_INTERVAL = 16
WINDOW_LENGTH = 7
WINDOW_STRIDE = 1


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    parser.add_argument("--model-dir", type=Path, default=os.getenv("MODEL_DIR"))
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu", default=os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("GPU", "0")))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error("Provide --model-dir or MODEL_DIR.")
    if args.video_root is None:
        parser.error("Provide --video-root.")
    if args.rank < 0 or args.world_size < 1 or args.rank >= args.world_size:
        parser.error("Require 0 <= rank < world-size.")
    args.model_dir = Path(args.model_dir).expanduser().resolve()
    args.video_root = args.video_root.expanduser().resolve()
    args.output_root = (args.output_root or ROOT / "video-features" / args.dataset).expanduser().resolve()
    if not args.model_dir.is_dir() or not args.video_root.is_dir():
        parser.error("The model directory and video root must exist.")
    return args


def video_files(root, dataset):
    if dataset == "ucf":
        return sorted(path for path in root.rglob("*.mp4") if path.is_file())
    return sorted(path for path in root.iterdir() if path.is_file() and path.name.lower().endswith(".mp4"))


def output_path(video, root, output_root, dataset):
    if dataset == "ucf":
        return (output_root / video.relative_to(root)).with_suffix(".npy")
    return output_root / f"{video.stem}.npy"


def windows(video_path, video_reader, pil_image, cpu):
    reader = video_reader(str(video_path), ctx=cpu(0))
    indices = list(range(0, len(reader), FRAME_INTERVAL))
    for start in range(0, len(indices) - WINDOW_LENGTH + 1, WINDOW_STRIDE):
        frames = reader.get_batch(indices[start:start + WINDOW_LENGTH]).asnumpy()
        yield [pil_image.fromarray(frame) for frame in frames]


def extract(video_path, target, model, batch_size, video_reader, pil_image, cpu, torch, np):
    batches = []
    pending = []
    for frames in windows(video_path, video_reader, pil_image, cpu):
        pending.append({"video": frames, "instruction": INSTRUCTION})
        if len(pending) == batch_size:
            with torch.no_grad():
                result = model.process(pending)
            batches.append(result.cpu())
            pending = []
            del result
            torch.cuda.empty_cache()
    if pending:
        with torch.no_grad():
            result = model.process(pending)
        batches.append(result.cpu())
        del result
        torch.cuda.empty_cache()
    if not batches:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    features = torch.cat(batches, dim=0).float().numpy()
    np.save(target, features)
    return True


def main():
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(args.model_dir))
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from PIL import Image
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

    model = Qwen3VLEmbedder(model_name_or_path=str(args.model_dir), max_pixels=600000)
    tasks = video_files(args.video_root, args.dataset)[args.rank::args.world_size]
    batch_size = 2 if args.dataset == "ucf" else 1
    completed = 0
    skipped = 0
    for video in tasks:
        target = output_path(video, args.video_root, args.output_root, args.dataset)
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            if extract(video, target, model, batch_size, VideoReader, Image, cpu, torch, np):
                completed += 1
            else:
                skipped += 1
                print(f"Skipped short video: {video}")
        except Exception as exc:
            print(f"Failed: {video}: {exc}", file=sys.stderr)
    print(f"Extracted {completed} videos; skipped {skipped}; assigned {len(tasks)}.")


if __name__ == "__main__":
    main()
