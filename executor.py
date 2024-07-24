from typing import List, Dict, Union, Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import spacy
import hashlib
import os
import os.path as osp

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import tqdm

from diffusion.models import get_sd_model, get_scheduler_config
from interpreter import Box


class VGDiffZeroExecutor:
    def __init__(self, version: str = '2-1', n_trials: int = 1, n_samples: List[int] = [5, 10], device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "sum") -> None:
        IMPLEMENTED_METHODS = ["crop", "mask"]
        # Raise an error if an unsupported box representation method is used
        if any(m not in IMPLEMENTED_METHODS for m in box_representation_method.split(",")):
            raise NotImplementedError
        IMPLEMENTED_AGGREGATORS = ["max", "sum"]
        # Raise an error if an unsupported method aggregator is used
        if method_aggregator not in IMPLEMENTED_AGGREGATORS:
            raise NotImplementedError
        self.device = device
        self.box_representation_method = box_representation_method
        self.method_aggregator = method_aggregator

        self.version = version
        self.n_trials = n_trials
        self.n_samples = n_samples
        # Load the Stable Diffusion model and its components
        self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler = get_sd_model(version)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.unet = self.unet.to(device)
        self.scheduler_config = get_scheduler_config(version)

        # Define image transformation steps
        self.image_transform = transforms.Compose([
            transforms.Resize(512, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.preprocesses = [self.image_transform]

    def preprocess_text(self, text: str) -> torch.Tensor:
        # Preprocess the text input for the model
        text_input = self.tokenizer(["a photo of " + text.lower()], padding="max_length",
                           max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        return text_input
    
    def preprocess_image(self, image: Image) -> List[torch.Tensor]:
        # Preprocess the image input for the model
        return [preprocess(image) for preprocess in self.preprocesses]
    
    def tensorize_inputs(self, caption: str, image: Image, boxes: List[Box]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        images = []
        for preprocess in self.preprocesses:
            images.append([])
       
        if "crop" in self.box_representation_method:
            # Process each bounding box by cropping the image
            for i in range(len(boxes)):
                image_i = image.copy()
                box = [
                    max(boxes[i].left, 0),
                    max(boxes[i].top, 0),
                    min(boxes[i].right, image_i.width),
                    min(boxes[i].bottom, image_i.height)
                ]
                image_i = image_i.crop(box)
                preprocessed_images = self.preprocess_image(image_i)
                for j, img in enumerate(preprocessed_images):
                    images[j].append(img.to(self.device))
        if "mask" in self.box_representation_method:
            # Process each bounding box by masking the image
            for i in range(len(boxes)):
                image_i = image.copy()
                mask = Image.new('L', image_i.size, 0)
                draw = ImageDraw.Draw(mask)
                box = (
                    max(boxes[i].left, 0),
                    max(boxes[i].top, 0),
                    min(boxes[i].right, image_i.width),
                    min(boxes[i].bottom, image_i.height)
                )
                draw.rectangle([box[:2], box[2:]], fill=255)
                mask = mask.convert('RGB')   
                masked = ImageChops.multiply(image_i, mask) 
                preprocessed_images = self.preprocess_image(masked)
                for j, img in enumerate(preprocessed_images):
                    images[j].append(img.to(self.device))
        imgs = [torch.stack(image_list) for image_list in images]
        
        # Preprocess the text input
        text_tensor = self.preprocess_text(caption.lower()).to(self.device)
        return imgs, text_tensor

    def eval_error(self, latent, all_noise, ts, noise_idxs,
                   text_embed, batch_size=32, dtype='float32', loss='l2'):
        # Evaluate the error between the predicted noise and the actual noise
        assert len(ts) == len(noise_idxs)
        pred_errors = torch.zeros(len(ts), device='cpu')
        idx = 0
        with torch.inference_mode():
            for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
                batch_ts = torch.tensor(ts[idx: idx + batch_size])
                noise = all_noise[noise_idxs[idx: idx + batch_size]]
                noised_latent = latent * (self.scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(self.device) + \
                                noise * ((1 - self.scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(self.device)
                t_input = batch_ts.to(self.device).half() if dtype == 'float16' else batch_ts.to(self.device)
                text_input = torch.cat([text_embed]*noised_latent.shape[0], dim=0)
                noise_pred = self.unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
                if loss == 'l2':
                    error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                elif loss == 'l1':
                    error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                elif loss == 'huber':
                    error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                else:
                    raise NotImplementedError
                pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
                idx += len(batch_ts)
        return pred_errors

    def eval_single_prompt(self, latent, text_embed, latent_height, latent_width, all_noise=None):
        # Evaluate a single prompt and return the mean error
        scheduler_config = get_scheduler_config(self.version)
        T = scheduler_config['num_train_timesteps']
        max_n_samples = max(self.n_samples)

        if all_noise is None:
            all_noise = torch.randn((max_n_samples * self.n_trials, 4, latent_height, latent_width), device=self.device)

        t_evaluated = set()
        start = T // max_n_samples // 2
        t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]
        
        for n_samples in self.n_samples:
            ts = []
            noise_idxs = []
            curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
            curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t] * self.n_trials)
                noise_idxs.extend(list(range(self.n_trials * t_idx, self.n_trials * (t_idx + 1))))
            t_evaluated.update(curr_t_to_eval)
            pred_errors = self.eval_error(latent, all_noise, ts, noise_idxs,
                                        text_embed, batch_size=32, dtype='float32', loss='l2')
            
        error = pred_errors.mean()
        return error

    def __call__(self, caption: str, image: Image, boxes: List[Box]) -> torch.Tensor:
        # Main function to execute the VGDiffZero evaluation
        images, text_tensor = self.tensorize_inputs(caption, image, boxes)
        box_representation_methods = self.box_representation_method.split(',')
        embeddings = []
        with torch.inference_mode():
            for i in range(0, len(text_tensor.input_ids), 100):
                text_embeddings = self.text_encoder(
                    text_tensor.input_ids[i: i + 100].to(self.device),
                )[0]
                embeddings.append(text_embeddings)
        text_embeddings = torch.cat(embeddings, dim=0)
        
        all_logits_per_text = []
        with torch.inference_mode():
            for i in range(len(box_representation_methods) * len(boxes)): 
                image = images[0][i].unsqueeze(0)
                latent_height, latent_width = image.shape[2] // 8, image.shape[3] // 8
                x0 = self.vae.encode(image).latent_dist.mean
                x0 *= 0.18215
               

 logits_per_text = self.eval_single_prompt(x0, text_embeddings, latent_height, latent_width, all_noise = None)
                all_logits_per_text.append(logits_per_text)
        if len(box_representation_methods) > 1:
            all_logits_per_text1 = F.softmax(torch.stack(all_logits_per_text[:len(boxes)]), dim=0)
            all_logits_per_text2 = F.softmax(torch.stack(all_logits_per_text[len(boxes):]), dim=0)
            if self.method_aggregator == "max":
                max_logit1, max_logit2 = torch.max(all_logits_per_text1), torch.max(all_logits_per_text1)
                if max_logit1 > max_logit2:
                    all_logits_per_text = all_logits_per_text1
                else:
                    all_logits_per_text = all_logits_per_text2
                all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).max(dim=0, keepdim=True)[0]
            elif self.method_aggregator == "sum":
                all_logits_per_text = all_logits_per_text1 + all_logits_per_text2
                all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).sum(dim=0, keepdim=True)
        else:
            all_logits_per_text = torch.stack(all_logits_per_text)
            if self.method_aggregator == "max":
                all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).max(dim=0, keepdim=True)[0]
            elif self.method_aggregator == "sum":
                all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).sum(dim=0, keepdim=True)
        return all_logits_per_text.view(-1)
