# DSAI 490 – Assignment 2 : Dates Generator

## Project Structure

```
dates_generator/
├── data/
│   ├── data.txt              # full dataset (given)
│   └── example_input.txt     # example input for inference (given)
├── model/
│   ├── model_lstm.py         # Model 1 – Conditional Seq2Seq LSTM (in-course)
│   ├── model_gan.py          # Model 2 – Conditional GAN (in-course, required)
│   ├── model_transformer.py  # Model 3 – Conditional Transformer Decoder (outside-course)
│   ├── model_vae.py          # Model 4 – Conditional VAE (outside-course)
│   ├── train_seq.py          # Training for LSTM & Transformer
│   ├── train_gan.py          # Training for GAN
│   ├── train_vae.py          # Training for VAE
│   ├── predict.py            # ← Required inference entry point
│   ├── evaluate.py           # Validity-rate evaluation
│   ├── inference.py          # Shared decode + fallback logic
│   └── weights/              # Saved checkpoints (after training)
├── utils/
│   ├── tokenizer.py          # Custom tokenizer, date encoder/decoder, validator
│   └── dataset.py            # PyTorch Dataset classes
├── environment.yml           # Conda environment spec
└── README.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate dates_generator
```

## Training

```bash
# From the repo root:

# Generate data files if data/data.txt is missing
python scripts/generate_data.py

# Model 1 – LSTM
python model/train_seq.py --model lstm --epochs 30

# Model 2 – GAN  (required)
python model/train_gan.py --epochs 50

# Model 3 – Transformer
python model/train_seq.py --model transformer --epochs 30

# Model 4 – VAE
python model/train_vae.py --epochs 40
```

## Inference

```bash
python model/predict.py \
  -i data/example_input.txt \
  -o predictions.txt \
  --model lstm          # or transformer | gan | vae
```

## Evaluation

```bash
python model/evaluate.py --predictions predictions.txt
```

## Models Summary

| # | Model | Type | Architecture |
|---|-------|------|-------------|
| 1 | Seq2Seq LSTM | In-course | Encoder (condition embeddings → h0,c0) + LSTM decoder |
| 2 | Conditional GAN | In-course (**required**) | MLP Generator (noise+cond) + MLP Discriminator |
| 3 | Transformer Decoder | Outside-course | Causal self-attention with condition prefix tokens |
| 4 | Conditional VAE | Outside-course | MLP Encoder + Decoder with KL divergence |

## Validation Metric

**Validity rate** = fraction of generated dates that satisfy all four conditions
(day-of-week, month, leap-year, decade). This is the primary training signal
because multiple correct outputs exist per input.
