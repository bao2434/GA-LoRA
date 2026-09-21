import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import torch.nn.functional as F

from .utils import *

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

INDEX_POSITIONS_VISION = {
    'B16': {
        'top1': [11],
        'top2': [10, 11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'B32': {
        'top1': [11],
        'top2': [10, 11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'L14': {
        'top1': [23],
        'top2': [22, 23],
        'top3': [21, 22, 23],
        'bottom': [0, 1, 2, 3, 4, 5, 6, 7],
        'mid': [8, 9, 10, 11, 12, 13, 14, 15],
        'up': [16, 17, 18, 19, 20, 21, 22, 23],
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


class GA_LoRA(nn.Module):
    def __init__(self, processor,
                model="B16", 
                target_encoder="all", 
                target_postion="all", 
                target_component = "all", 
                rank=1, 
                alpha=2, 
                lambda_val = 1.0,
                dropout=0.25,
                cache_dir="./models"):
        super().__init__()
        self.target_encoder = target_encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor
        self.backbone = CLIPModel.from_pretrained(CLIP_MODEL[model], cache_dir=cache_dir).to(self.device)
        for param in self.backbone.parameters():
            param.requires_grad = False
        backbone = CLIPModel.from_pretrained(CLIP_MODEL[model], cache_dir=cache_dir)
        target_modules = []
        for encoder in TARGET_ENCODER[target_encoder]:
            if (encoder == "vision_model"):
                for position in INDEX_POSITIONS_VISION[model][target_postion]:
                    for component in TARGET_COMPONENT[target_component]:
                        target_modules.append(f"{encoder}.encoder.layers.{position}.self_attn.{component}")
            elif (encoder == "text_model"):
                for position in INDEX_POSITIONS_TEXT[target_postion]:
                    for component in TARGET_COMPONENT[target_component]:
                        target_modules.append(f"{encoder}.encoder.layers.{position}.self_attn.{component}")
        config = LoraConfig(
            r = rank,
            lora_alpha = alpha,
            target_modules= target_modules,
            lora_dropout = dropout,
            bias = "none",
            task_type = TaskType.FEATURE_EXTRACTION
        )
        self.model = get_peft_model(backbone, config).to(self.device)
        self.model.print_trainable_parameters()

        self.lambda_val = lambda_val
    
    def forward(self, image, class_descriptions, encode_text = True, use_backbone= False):
        logit_scale = 100

        if encode_text:
            texts = {k: v.to(self.device) for k, v in class_descriptions.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  
                if use_backbone:
                    text_outputs = self.backbone.get_text_features(**texts)
                else:
                    text_outputs = self.model.get_text_features(**texts)
            class_embeddings = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs
            class_embeddings = class_embeddings.float()
            text_features = class_embeddings/(class_embeddings.norm(dim=-1, keepdim=True) + 1e-6)
        else:
            text_features = class_descriptions.to(self.device)            

        image = image.to(self.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  
            if use_backbone:
                image_outputs = self.backbone.get_image_features(pixel_values=image)
            else:
                image_outputs = self.model.get_image_features(pixel_values=image)
        image_embeddings = image_outputs.pooler_output if hasattr(image_outputs, "pooler_output") else image_outputs
        image_embeddings = image_embeddings.float()
        image_features = image_embeddings/(image_embeddings.norm(dim=-1, keepdim=True)  + 1e-6)

        logits = logit_scale*image_features @ text_features.t()
        return logits
    
    def train_model(self, dataset, class_description,total_iters, lr=5e-5, batch_size =32):

        class_description = self.processor(text=class_description, return_tensors = "pt", padding=True)

        train_loader =  torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=8,
                                                    pin_memory=True)

    
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], 
                                      weight_decay=1e-2, 
                                      betas=(0.9, 0.999), 
                                      lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
        
        scaler = torch.amp.GradScaler('cuda')
        count_iters = 0
        failed_iters = 0
        max_failed_iters = 100
        self.model.train()
        self.backbone.eval()
        trainable_params = [i for i in self.model.parameters() if i.requires_grad]

        while count_iters < total_iters:
            acc_train = 0
            tot_samples = 0
            loss_epoch = 0.

            for images, target in tqdm(train_loader):
                target = target.to(self.device)
                
                logits_lora = self(images, class_description)
                with torch.no_grad():
                    logits_base = self(images, class_description, use_backbone=True)

                loss_ce = F.cross_entropy(logits_lora, target)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_lora, dim=1),
                    F.log_softmax(logits_base, dim=1),
                    reduction="batchmean",
                    log_target=True,
                )
                
                g_ce = torch.autograd.grad(scaler.scale(loss_ce), trainable_params,
                                        retain_graph=True, allow_unused=True)
                g_kl = torch.autograd.grad(scaler.scale(loss_kl), trainable_params, allow_unused=True)

                g_ce = [torch.zeros_like(p) if g is None else g.float() for p, g in zip(trainable_params, g_ce)]
                g_kl = [torch.zeros_like(p) if g is None else g.float() for p, g in zip(trainable_params, g_kl)]

                for p, a, b in zip(trainable_params, g_ce, g_kl):

                    a_flat = a.view(-1).float()
                    b_flat = b.view(-1).float()

                    dot_prod = torch.dot(a_flat, b_flat)
                    g_kl_norm_sq = torch.clamp(
                        torch.dot(b_flat, b_flat),
                        min=1e-6
                    ) 
                    coeff = torch.minimum(
                        torch.zeros_like(dot_prod),
                        self.lambda_val * dot_prod / g_kl_norm_sq
                    )
                    p.grad = (a - coeff * b) 

                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                print("gradient norm:", total_norm.item())

                scale_before = scaler.get_scale()
                scaler.step(optimizer)      
                scaler.update()  

                if scaler.get_scale() < scale_before:  
                    failed_iters += 1
                    optimizer.zero_grad(set_to_none=True)
                    if failed_iters >= max_failed_iters:
                        raise RuntimeError("Too many skipped iterations due to non-finite gradients.")
                    continue

                scheduler.step()                 
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    acc_train += cls_acc(logits_lora, target) * target.shape[0]
                    loss_epoch += loss_ce.item() * target.shape[0]
                    tot_samples += target.shape[0]

                count_iters += 1
                if count_iters >= total_iters:
                    break
                    
            if tot_samples > 0:
                avg_acc = acc_train / tot_samples
                avg_loss = loss_epoch / tot_samples
                current_lr = scheduler.get_last_lr()[0]
                print(f'LR: {current_lr:.6f}, Acc: {avg_acc:.4f}, Loss: {avg_loss:.4f}')


    def test_model(self, dataset, class_description):

        class_description = self.processor(text=class_description, return_tensors = "pt", padding=True)

        test_loader =  torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=256, 
                                                    shuffle=False, 
                                                    num_workers=8,
                                                    pin_memory=True)
        
        self.model.eval()
        total_correct = 0.
        tot_samples = 0
        i=0
        with torch.no_grad():
            for images, target in test_loader:
                print("evaluate batch ",i) 
                i+=1
                target = target.to(self.device)
                cosine_similarity = self(images, class_description)
                total_correct += cls_acc(cosine_similarity, target) * images.size(0)
                tot_samples += images.size(0)

        return total_correct / tot_samples