import torch
from model import CLIP_Processor, GA_LoRA
from datasets import EuroSatDataset, OxfordFlowers, StanfordCars
import random
from torchvision import transforms
import numpy as np
import json
import os
from arg_parse import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():

    args = get_args()
    set_seed(args.seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
   
    processor = CLIP_Processor(model=args.model)

    if args.dataset == "eurosat":
        dataset = EuroSatDataset(
            transform_train=train_transform, 
            transform_test=processor.process_image, 
            shot_num=args.shot
        )
    elif args.dataset == "flower":
        dataset = OxfordFlowers(
            transform_train=train_transform, 
            transform_test=processor.process_image, 
            shot_num=args.shot
        )
    elif args.dataset == "car":
        dataset = StanfordCars(
            transform_train=train_transform, 
            transform_test=processor.process_image, 
            shot_num=args.shot
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.res_path)), exist_ok=True)

    try:
        with open(args.res_path, "r") as f:
            result_list = json.load(f)

        if not isinstance(result_list, list):
            raise ValueError("Result file must contain a JSON list")

    except (FileNotFoundError, json.JSONDecodeError):
        result_list = []

    count = len(result_list)
    class_description=[
        f"a photo of {text}" 
        for text in dataset.class_dict.values()
    ]

    model = GA_LoRA(
        processor=processor,
        model=args.model,
        target_encoder=args.target_encoder,
        target_position=args.target_position,
        target_component=args.target_component,
        rank=args.rank,
        alpha=args.alpha,
        lambda_val= args.lambda_val,
        dropout=args.dropout,
        cache_dir=args.cache_dir
    )


    print("start training")
    dataset.train_mode()
    model.train_model(
        dataset=dataset, 
        class_description=class_description,
        total_iters=args.total_iters
    )
    print("start evaluation")
    dataset.test_mode()
    result = model.test_model(
        dataset=dataset, 
        class_description=class_description
    )
    print("result: ", result)

    result_list.append({
        "count": count,
        **vars(args),
        "result": result
    })

    with open(args.res_path, "w") as f:
        json.dump(result_list, f, indent=4)

if __name__ == "__main__":
    main()