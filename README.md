# GA-LoRA: Gradient-Aligned Low-Rank Adaptation for CLIP

**GA-LoRA** is a parameter-efficient few-shot fine-tuning framework for Vision-Language Models (CLIP). It adapts pre-trained CLIP models to downstream classification tasks using Low-Rank Adaptation (LoRA) while preserving zero-shot generalization capabilities through **Gradient Alignment** between the cross-entropy task loss ($\mathcal{L}_{\text{CE}}$) and a Kullback-Leibler consistency loss ($\mathcal{L}_{\text{KL}}$) with the frozen CLIP backbone.

---

## Method Overview

During few-shot fine-tuning, standard cross-entropy optimization can lead to overfitting and catastrophic forgetting of pre-trained vision-language representations. GA-LoRA mitigates these issues by computing two gradients with respect to the trainable LoRA parameters at each optimization step:

1. **Task gradient $\boldsymbol{G}_{ce}$**: The gradient of the cross-entropy loss on the few-shot training data.
2. **Consistency gradient $\boldsymbol{G}_{KL}$**: The gradient of the KL divergence between the fine-tuned model's output distribution and that of the frozen zero-shot CLIP model.

When the two gradients conflict, i.e., $\boldsymbol{G}_{ce}^{\top}\boldsymbol{G}_{KL}<0$, GA-LoRA removes a fraction of the projection of $\boldsymbol{G}_{ce}$ onto $\boldsymbol{G}_{KL}$:

$$
\boldsymbol{G}_{align}
=
\boldsymbol{G}_{ce}
-
\min\left(
0,
\lambda
\frac{
\boldsymbol{G}_{ce}^{\top}\boldsymbol{G}_{KL}
}{
\|\boldsymbol{G}_{KL}\|_2^2
}
\right)
\boldsymbol{G}_{KL},
$$

where $0\leq\lambda\leq1$ controls the alignment strength. A larger $\lambda$ removes more of the conflicting component, reducing the first-order increase in KL divergence caused by the task update.

---

## Project Structure

```text
GA-LoRA/
├── arg_parse.py             # CLI argument parser and hyperparameter validation
├── main.py                  # Main entry point for training and evaluation
├── datasets/                # Few-shot dataset wrappers
│   ├── __init__.py
│   ├── eurosat.py           # EuroSAT dataset
│   ├── flower102.py         # Oxford 102 Flowers dataset
│   ├── stanford_cars.py     # Stanford Cars dataset
│   ├── caltech101.py        # Caltech-101 dataset
│   ├── food101.py           # Food-101 dataset
│   ├── pet.py               # Oxford-IIIT Pet dataset
│   └── utils.py             # Few-shot subset sampling utility
├── model/                   # GA-LoRA and CLIP wrapper implementation
│   ├── __init__.py
│   ├── CLIP_processor.py    # Hugging Face CLIPProcessor wrapper
│   ├── GA_LoRA.py           # GA-LoRA model, training, and evaluation logic
│   └── utils.py             # Top-k accuracy metric utility
└── figure/                  # Architecture diagrams and experimental plots
```

---

## Requirements & Installation

1. **Python**: 3.10+
2. **CUDA-capable GPU**: Required (mixed-precision training uses `torch.amp.autocast` and `GradScaler` on CUDA).
3. **Dependencies**: Install the required packages via `pip`:

```bash
pip install torch torchvision transformers peft tqdm numpy
```

> **Note:** `torchvision >= 0.19.0` is recommended so that `torchvision.datasets.Flowers102` includes the `.classes` attribute used by `datasets/flower102.py`.

---

## Dataset Preparation

Datasets are stored under the `./data` directory by default:

- **EuroSAT (`--dataset eurosat`)** and **Oxford Flowers (`--dataset flower`)**: Automatically downloaded to `./data` on first run via `torchvision.datasets`.
- **Stanford Cars (`--dataset car`)**: Uses `download=False` in `datasets/stanford_cars.py` (since the original Stanford hosting URL was deprecated in `torchvision`). Before running with `--dataset car`, place the dataset inside `./data/stanford_cars/` following the standard `torchvision` structure:
  ```text
  ./data/stanford_cars/
  ├── cars_train/
  ├── cars_test/
  ├── cars_test_annos_withlabels.mat
  └── devkit/
      ├── cars_meta.mat
      └── cars_train_annos.mat
  ```

---

## Usage

### Basic Run

Run training and evaluation with the default configuration (ViT-B/16, EuroSAT, 16-shot, rank 1):

```bash
python main.py
```

### Custom Configuration Examples

**1. Train ViT-B/16 on EuroSAT (16-shot) targeting only Vision encoder Value/Output projections:**
```bash
python main.py \
    --model B16 \
    --dataset eurosat \
    --shot 16 \
    --target_encoder vision \
    --target_position all \
    --target_component vo \
    --rank 8 \
    --alpha 16 \
    --lambda_val 0.7 \
    --total_iters 500
```

**2. Train ViT-L/14 on Oxford Flowers (4-shot):**
```bash
python main.py \
    --model L14 \
    --dataset flower \
    --shot 4 \
    --target_encoder all \
    --target_component qv \
    --rank 4 \
    --lambda_val 1.0
```

**3. Train ViT-B/32 on Stanford Cars (8-shot) and save results to a custom file:**
```bash
python main.py \
    --model B32 \
    --dataset car \
    --shot 8 \
    --res_path ./results/cars_b32.json
```

---

## Command-Line Arguments

| Argument | Type | Default | Choices / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | `str` | `B16` | `B32`, `B16`, `L14` | Pre-trained CLIP backbone (`openai/clip-vit-base-patch32`, `base-patch16`, or `large-patch14`). |
| `--target_encoder` | `str` | `all` | `vision`, `text`, `all` | Which CLIP encoder(s) to apply LoRA adapters to. |
| `--target_position` | `str` | `all` | `top1`, `top2`, `top3`, `bottom`, `mid`, `up`, `half-up`, `half-bottom`, `all` | Transformer block layers to inject LoRA into. |
| `--target_component` | `str` | `all` | `qv`, `all`, `qvk`, `vo`, `v`, `o` | Attention projection matrices to adapt (`q_proj`, `k_proj`, `v_proj`, `out_proj`). |
| `--rank` | `int` | `1` | `> 0` | LoRA rank ($r$). |
| `--alpha` | `float` | `2` | `> 0` | LoRA scaling factor ($\alpha$). |
| `--lambda_val` | `float` | `1.0` | `[0.0, 1.0]` | Gradient alignment coefficient ($\lambda$). Set `0.0` for standard CE gradient only, `1.0` for full orthogonal projection. |
| `--dropout` | `float` | `0.25` | `[0.0, 1.0)` | Dropout probability for LoRA layers. |
| `--dataset` | `str` | `eurosat` | `eurosat`, `flower`, `car` | Target benchmark dataset. |
| `--shot` | `int` | `16` | `1`, `2`, `4`, `8`, `16` | Number of few-shot training examples per class. |
| `--total_iters` | `int` | `500` | `> 0` | Total number of training iterations. |
| `--cache_dir` | `str` | `./models` | Path | Directory to cache downloaded Hugging Face CLIP weights. |
| `--res_path` | `str` | `./results/results.json` | Path | Path to JSON file where evaluation results are appended. |
| `--seed` | `int` | `42` | Any integer | Random seed for reproducibility. |

---

## Output Format

After each run, evaluation results are appended as a JSON entry to `--res_path` (default `./results/results.json`):

```json
[
    {
        "count": 0,
        "model": "B16",
        "target_encoder": "all",
        "target_position": "all",
        "target_component": "all",
        "rank": 1,
        "alpha": 2,
        "lambda_val": 1.0,
        "dropout": 0.25,
        "dataset": "eurosat",
        "shot": 16,
        "total_iters": 500,
        "cache_dir": "./models",
        "res_path": "./results/results.json",
        "seed": 42,
        "result": 0.9315
    }
]
```
