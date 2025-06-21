import argparse
import os
from typing import List, Tuple
from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.utils.img_util import resize_image


def load_pipeline(model_path, accelerator, weight_dtype):
    pipeline = OmniGen2Pipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
  
    if args.scheduler == "dpmsolver":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(accelerator.device)
      
    return pipeline


def preprocess(input_image_path: List[str] = []) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the input images."""
  
    # Process input images
    input_images = None

    if input_image_path:
        input_images = []
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                          for f in os.listdir(input_image_path[0])]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images


def run(width, height, num_inference_step, text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end, 
        num_images_per_prompt, accelerator, pipeline, instruction, negative_prompt, input_images):
    """Run the image generation pipeline with the given parameters."""
          
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width,
        height=height,
        num_inference_steps=num_inference_step,
        max_sequence_length=1024,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        cfg_range=(cfg_range_start, cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
          
    return results


def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
  
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)


def main(model_path, dtype, input_images, root_dir, width, height, num_inference_step, text_guidance_scale, 
         image_guidance_scale, cfg_range_start, cfg_range_end, num_images_per_prompt, instruction, negative_prompt):
    """Main function to run the image generation process."""
  
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=dtype if dtype != 'fp32' else 'no')

    # Set weight dtype
    weight_dtype = torch.float32
    if dtype == 'fp16':
        weight_dtype = torch.float16
    elif dtype == 'bf16':
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(model_path, accelerator, weight_dtype)

    # Generate and save image
    results = run(width, height, num_inference_step, text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end, 
                  num_images_per_prompt, accelerator, pipeline, instruction, negative_prompt, input_images)


class LoadOmniGen2Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "assets/demo.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "input_image_path"
    CATEGORY = "OmniGen2"

    def input_image(self, image_path):
        input_images = preprocess(image_path)
        return (input_images,)


class LoadOmniGen2Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "OmniGen2/OmniGen2"}),
                "dtype": (["fp32", "fp16", "bf16"], {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "OmniGen2"

    def load_model(self, model_path, dtype):
        # Initialize accelerator
        accelerator = Accelerator(mixed_precision=dtype if dtype != 'fp32' else 'no')
            
        # Set weight dtype
        weight_dtype = torch.float32
        if dtype == 'fp16':
            weight_dtype = torch.float16
        elif dtype == 'bf16':
            weight_dtype = torch.bfloat16
    
        # Load pipeline and process inputs
        pipeline = load_pipeline(model_path, accelerator, weight_dtype)
        
        return (pipeline,)

