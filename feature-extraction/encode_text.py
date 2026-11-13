import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UCF_NORMAL_PREFIX = "A surveillance video of "
UCF_ANOMALY_PREFIX = "A surveillance video of anomalous event: "
XD_NORMAL_PREFIX = "Video footage of "
XD_ANOMALY_PREFIX = "Real-world video footage of anomalous event: "


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    parser.add_argument("--model-dir", type=Path, default=os.getenv("MODEL_DIR"))
    parser.add_argument("--descriptions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default=os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("GPU", "0")))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error("Provide --model-dir or MODEL_DIR.")
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    args.model_dir = Path(args.model_dir).expanduser().resolve()
    args.descriptions = (args.descriptions or ROOT / "data" / f"{args.dataset}_descriptions.json").expanduser().resolve()
    args.output = (args.output or ROOT / "data" / f"{args.dataset}_text_bank_cpu.npz").expanduser().resolve()
    if not args.model_dir.is_dir() or not args.descriptions.is_file():
        parser.error("The model directory and descriptions file must exist.")
    if args.output.exists() and not args.overwrite:
        parser.error("The output exists; pass --overwrite to replace it.")
    return args


def prompt_groups(data, dataset):
    if dataset == "ucf":
        effective = data["effective_prompts"]
        normal = effective["normal"]
        classes = effective["anomalies"]
        if (effective["normal_embedding_template"] != UCF_NORMAL_PREFIX or
                effective["anomaly_embedding_template"] != UCF_ANOMALY_PREFIX):
            raise ValueError("UCF embedding templates do not match the feature protocol.")
        if normal != data["prompt_config"]["normal_immunity_texts"]:
            raise ValueError("UCF normal prompts do not match the effective prompt list.")
        if [group["class"] for group in classes] != sorted(data["content"]["anomalies"]):
            raise ValueError("UCF anomaly class order does not match the effective prompt list.")
        groups = [group["texts"] for group in classes]
        return normal, groups, UCF_NORMAL_PREFIX, UCF_ANOMALY_PREFIX
    normal = [text for group in data["prompt_config"]["scenario_normals"].values() for text in group]
    classes = data["prompt_config"]["class_overrides"]
    groups = [data["content"]["anomalies"][classes[code]] for code in sorted(classes)]
    return normal, groups, XD_NORMAL_PREFIX, XD_ANOMALY_PREFIX


def encode(texts, prefix, model, batch_size, torch, functional):
    if not texts:
        raise ValueError("The prompt group is empty.")
    parts = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            requests = [{"text": prefix + text} for text in texts[start:start + batch_size]]
            parts.append(model.process(requests))
    return functional.normalize(torch.cat(parts, dim=0), p=2, dim=1)


def main():
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(args.model_dir))
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

    raw = args.descriptions.read_bytes()
    data = json.loads(raw)
    normal_texts, groups, normal_prefix, anomaly_prefix = prompt_groups(data, args.dataset)
    model = Qwen3VLEmbedder(model_name_or_path=str(args.model_dir))
    normal = encode(normal_texts, normal_prefix, model, args.batch_size, torch, functional)
    anomaly_groups = [encode(group, anomaly_prefix, model, args.batch_size, torch, functional)
                      for group in groups]
    normal_array = normal.float().cpu().numpy()
    anomaly_arrays = [group.float().cpu().numpy() for group in anomaly_groups]
    arrays = {
        "normal": normal_array,
        "anomaly": np.concatenate(anomaly_arrays),
        "anomaly_means": torch.stack([group.mean(dim=0) for group in anomaly_groups]).float().cpu().numpy(),
        "descriptions_sha256": np.asarray(hashlib.sha256(raw).hexdigest()),
    }
    if args.dataset == "ucf":
        arrays["sizes"] = np.asarray([len(group) for group in groups], dtype=np.int64)
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key != "descriptions_sha256"):
        raise ValueError("The text bank contains nonfinite values.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved {len(normal_texts)} normal prompts and {len(groups)} anomaly groups to {args.output}.")


if __name__ == "__main__":
    main()
