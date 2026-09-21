import torch
from model import CLIP_Processor, GA_LoRA
from datasets import OxfordFlowers
import random
from torchvision import transforms
import numpy as np
import gc
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prioritize speed  over exact reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def main():

    res_dir = "result/results.json"
    clip_model_name = "B32"
    print("clip model: ", clip_model_name)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    shot_list = [1, 2, 4, 8]
    target_encoder_list = ["all"]
    target_position_list = ['all']
    target_component_list = ['v', 'o']
    rank_list = [1, 8, 16]
    lambda_list = [0.2, 0.4, 0.7, 0.9, 1.0]
   
    processor = CLIP_Processor(model=clip_model_name)

    dataset_list = []
    for shot in shot_list:
        dataset = OxfordFlowers(transform_train=train_tranform, transform_test=processor.process_image, shot_num=shot)
        dataset_list.append(dataset)

    try:
        with open(res_dir, "r") as f:
            result_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        result_list = []

    breakpoint = len(result_list)

    count=-1
    
    for index, shot in enumerate(shot_list):
        for target_encoder in target_encoder_list:
            for target_position in target_position_list:
                for target_component in target_component_list:
                    for rank in rank_list:
                        for lambda_val in lambda_list:
                            
                            count+=1
                            if count < breakpoint:
                                continue

                            print(f"shot: {shot}, target encoder: {target_encoder}, target position: {target_position}, target component: {target_component}, rank: {rank}")
                            
                            set_seed()
                            dataset = dataset_list[index]
                            class_description=[f"a photo of {text}" for text in dataset.class_dict.values()]

                            model = GA_LoRA(
                                processor=processor,
                                model=clip_model_name,
                                target_encoder=target_encoder,
                                target_postion=target_position,
                                target_component=target_component,
                                rank=rank,
                                alpha=2*rank,
                                lambda_val= lambda_val
                            )


                            print("start training")
                            dataset.train_mode()
                            model.train_model(dataset=dataset, class_description=class_description,total_iters=500)
                            print("start evaluation")
                            dataset.test_mode()
                            result = model.test_model(dataset=dataset, class_description=class_description)
                            print("result: ", result)

                            result_list.append({
                                "count": count,
                                "shot": shot, 
                                "target encoder": target_encoder, 
                                "target position": target_position, 
                                "target component": target_component, 
                                "rank": rank,
                                "lambda": lambda_val,
                                "result": result
                            })

                            with open(res_dir, "w") as f:
                                json.dump(result_list, f, indent=4)

                            del model
                            gc.collect()
                            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()