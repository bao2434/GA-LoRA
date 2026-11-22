import argparse

CLIP_MODEL = {
    "B32": "openai/clip-vit-base-patch32",
    "B16": "openai/clip-vit-base-patch16",
    "L14": "openai/clip-vit-large-patch14",
}

TARGET_ENCODER = {
    "vision": ["vision_model"],
    "text": ["text_model"],
    "all": ["vision_model", "text_model"]
}

TARGET_COMPONENT = {
    'qv': ["q_proj", "v_proj"],
    'all': ["q_proj", "k_proj", "v_proj", "out_proj"],
    'qvk': ["q_proj", "v_proj", "k_proj"],
    'vo': ["v_proj", "out_proj"],
    'v': ["v_proj"],
    'o': ["out_proj"]
}

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

def get_args():
    parser = argparse.ArgumentParser(description="GA-LoRA configuration")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="B16",
        choices=CLIP_MODEL.keys()
    )

    parser.add_argument(
        "--target_encoder",
        type=str,
        default="all",
        choices=TARGET_ENCODER.keys()
    )

    parser.add_argument(
        "--target_position",
        type=str,
        default="all",
        choices=INDEX_POSITIONS_TEXT.keys()
    )

    parser.add_argument(
        "--target_component",
        type=str,
        default="all",
        choices=TARGET_COMPONENT.keys()
    )

    # LoRA hyperparameters
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--lambda_val", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.25)

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="eurosat",
        choices=["eurosat", "flower", "car"]
    )

    # Number of shots
    parser.add_argument(
        "--shot",
        type=int,
        default=16,
        choices=[1, 2, 4, 8, 16]
    )

    # Training configuration
    parser.add_argument(
        "--total_iters",
        type=int,
        default=500,
        help="Total number of training iterations"
    )

    # Directory configuration
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--res_path", type=str, default="./results/results.json")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate hyperparameters
    if args.rank <= 0:
        parser.error("--rank must be a positive integer")

    if args.alpha <= 0:
        parser.error("--alpha must be positive")

    if not 0 <= args.lambda_val <= 1:
        parser.error("--lambda_val must be between 0 and 1")

    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be between 0 and 1 (exclusive)")

    if args.total_iters <= 0:
        parser.error("--total_iters must be a positive integer")

    return args