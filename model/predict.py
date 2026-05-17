"""
predict.py – Inference entry point.

Usage:
    python predict.py -i path/to/input_file -o path/to/output_file \
                      [--model lstm|transformer|gan|vae] \
                      [--weights_dir model/weights]

The input file contains lines of the form:
    [DAY] [MONTH] [LEAP] [DECADE]

The output file contains lines of the form:
    [DAY] [MONTH] [LEAP] [DECADE] dd-mm-yyyy
"""

from __future__ import annotations
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset  import ConditionsDataset
from utils.tokenizer import decode_conditions, format_output_line
from model.inference import decode_model_output


def resolve_project_path(path: str) -> str:
    """Resolve relative paths from cwd first, then from the repo root."""
    if os.path.isabs(path):
        return path

    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path

    return os.path.join(PROJECT_ROOT, path)


def load_model(model_name: str, weights_dir: str, device: torch.device):
    checkpoint_names = {
        "lstm": "lstm_best.pt",
        "transformer": "transformer_best.pt",
        "gan": "gan_best.pt",
        "vae": "vae_best.pt",
    }
    training_commands = {
        "lstm": "python model/train_seq.py --model lstm --epochs 30",
        "transformer": "python model/train_seq.py --model transformer --epochs 30",
        "gan": "python model/train_gan.py --epochs 50",
        "vae": "python model/train_vae.py --epochs 40",
    }
    if model_name not in checkpoint_names:
        raise ValueError(f"Unknown model: {model_name}")

    weights_dir = resolve_project_path(weights_dir)
    checkpoint_path = os.path.join(weights_dir, checkpoint_names[model_name])
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            f"Train the model first, for example:\n"
            f"  {training_commands[model_name]}\n"
            f"or pass the folder containing {checkpoint_names[model_name]} with "
            f"--weights_dir."
        )

    if model_name == "lstm":
        from model.model_lstm import ConditionalSeq2SeqLSTM
        model = ConditionalSeq2SeqLSTM()
        ckpt  = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model, "lstm"

    elif model_name == "transformer":
        from model.model_transformer import ConditionalTransformerDecoder
        model = ConditionalTransformerDecoder()
        ckpt  = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model, "transformer"

    elif model_name == "gan":
        from model.model_gan import ConditionEncoder, Generator
        cond_enc = ConditionEncoder()
        model    = Generator(noise_dim=64, cond_enc=cond_enc)
        ckpt     = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["G_state"])
        model.to(device).eval()
        return model, "gan"

    elif model_name == "vae":
        from model.model_vae import ConditionalVAE
        model = ConditionalVAE()
        ckpt  = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model, "vae"


@torch.no_grad()
def run_inference(model, model_type: str, dataset: ConditionsDataset,
                  device: torch.device, batch_size: int = 256) -> list[str]:
    """Run inference on all samples and return output date strings."""
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = []

    for day_b, mon_b, leap_b, dec_b in loader:
        day_b  = day_b.to(device)
        mon_b  = mon_b.to(device)
        leap_b = leap_b.to(device)
        dec_b  = dec_b.to(device)

        # ── generate tokens per model type ─────────────────────────────
        if model_type in ("lstm", "transformer"):
            seqs = model.generate(day_b, mon_b, leap_b, dec_b)
        elif model_type in ("gan", "vae"):
            seqs = model.generate_tokens(day_b, mon_b, leap_b, dec_b)
        else:
            raise ValueError(model_type)

        for b in range(day_b.size(0)):
            d, m, l, decade = decode_conditions(
                day_b[b].item(), mon_b[b].item(),
                leap_b[b].item(), dec_b[b].item()
            )
            toks     = seqs[b]
            date_str = decode_model_output(toks, d, m, l, decade,
                                           fallback_seed=b)
            results.append(format_output_line(d, m, l, decade, date_str))

    return results


def main():
    parser = argparse.ArgumentParser(description="Date Generator – Inference")
    parser.add_argument("-i", "--input",       required=True,
                        help="Path to input conditions file")
    parser.add_argument("-o", "--output",      required=True,
                        help="Path to output predictions file")
    parser.add_argument("--model",             default="lstm",
                        choices=["lstm", "transformer", "gan", "vae"])
    parser.add_argument("--weights_dir",       default="model/weights")
    parser.add_argument("--batch_size",        type=int, default=256)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] model={args.model}  device={device}")

    try:
        model, model_type = load_model(args.model, args.weights_dir, device)
    except FileNotFoundError as exc:
        print(f"[predict] {exc}", file=sys.stderr)
        sys.exit(1)

    dataset = ConditionsDataset(resolve_project_path(args.input))
    print(f"[predict] {len(dataset)} samples to predict")

    results = run_inference(model, model_type, dataset, device, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print(f"[predict] Written {len(results)} predictions → {args.output}")


if __name__ == "__main__":
    main()
