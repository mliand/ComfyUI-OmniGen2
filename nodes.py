import os
from typing import List, Tuple
from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator

from .omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from .omnigen2.utils.img_util import resize_image

#from diffusers.hooks import apply_group_offloading # only exists in very recent commits of diffusers


def load_pipeline(model_path, accelerator, weight_dtype, scheduler, offload_type):    
    pipeline = OmniGen2Pipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype,
        #trust_remote_code=True,
    )
  
    if scheduler == "dpmsolver":
        from .omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler
    else:
        from .omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    
    if offload_type == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload()
    elif offload_type == "cpu_offload":
        pipeline.enable_model_cpu_offload()
    #elif offload_type == "group_offload":
    #    apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    #    apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    #    apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(accelerator.device)
      
    return pipeline


def preprocess(image1_path: str, image2_path: str, image3_path: str) -> List[Image.Image]:
    """Preprocess the input images."""
    
    image_paths = [p for p in [image1_path, image2_path, image3_path] if p.strip() != ""]
    
    input_images = [Image.open(path).convert("RGB") for path in image_paths]
    input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images


def run(pipeline, input_images, seed, width, height, num_inference_step, text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end, 
        num_images_per_prompt, accelerator, instruction, negative_prompt):
    """Run the image generation pipeline with the given parameters."""
          
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

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


def create_collage(images: List[torch.Tensor]) -> torch.Tensor:
    """Create a horizontal collage from a list of images."""
  
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return canvas


class LoadOmniGen2Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1_path": ("STRING", {"default": "assets/demo.png", "tooltip": "Required"}),
            },
            "optional": {
                "image2_path": ("STRING", {"default": "", "tooltip": "Optional"}),
                "image3_path": ("STRING", {"default": "", "tooltip": "Optional"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("A list of PIL images. You should connect this directly to 'OmniGen2'.",)
    FUNCTION = "input_image"
    CATEGORY = "OmniGen2"
    DESCRIPTION = """
This node loads images from their paths and transposes them accordingly to their EXIF data.
ATTENTION: the output of this node is a list of PIL images. ComfyUI normally transfers images as torch.Tensor in between nodes so its highly recommended you connect the output of this node directly to 'OmniGen2'.
"""

    def input_image(self, image1_path, image2_path="", image3_path=""):
        for p in [image1_path, image2_path, image3_path]:
            if (p.strip() != "" or p == image1_path) and not os.path.isfile(p):
                raise RuntimeError(f"[ERROR] File not found: {p}")
        
        return (preprocess(image1_path, image2_path, image3_path),)


class LoadOmniGen2Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "OmniGen2/OmniGen2"}),
                "dtype": (["fp32", "fp16", "bf16"], {"default": "bf16"}),
                "scheduler": (["euler", "dpmsolver"], {"default": "euler"}),
                #"device": (["cuda", "cpu"], {"default": "cuda"}),
                "offload_type": (
                    #["none", "sequential_cpu_offload", "cpu_offload", "group_offload"], 
                    ["none", "sequential_cpu_offload", "cpu_offload"], 
                    {"default": "sequential_cpu_offload"}
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "OmniGen2"
    
    def load_model(self, model_path, dtype, scheduler, offload_type):
        # Initialize accelerator
        accelerator = Accelerator(mixed_precision=dtype if dtype != 'fp32' else 'no')
            
        # Set weight dtype
        weight_dtype = torch.float32
        if dtype == 'fp16':
            weight_dtype = torch.float16
        elif dtype == 'bf16':
            weight_dtype = torch.bfloat16
    
        # Load pipeline and process inputs
        pipeline = load_pipeline(model_path, accelerator, weight_dtype, scheduler, offload_type)
        
        return (pipeline,)


class OmniGen2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("MODEL",),
                "input_images": ("IMAGE", {"tooltip": "TIP: Connect this to 'Load OmniGen2 Image' node."}),
                "dtype": (["fp32", "fp16", "bf16"], {"default": "bf16"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "num_inference_step": ("INT", {"default": 50}),
                "text_guidance_scale": ("FLOAT", {"default": 5.0}),
                "image_guidance_scale": ("FLOAT", {"default": 2.0}),
                "cfg_range_start": ("FLOAT", {"default": 0.0}),
                "cfg_range_end": ("FLOAT", {"default": 1.0}),
                "num_images_per_prompt": ("INT", {"default": 1}),
                "instruction": ("STRING", {"default": "A dog running in the park"}),
                "negative_prompt": ("STRING", {"default": "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("images", "collage",)
    OUTPUT_IS_LIST = (True, False,)
    FUNCTION = "generate"
    CATEGORY = "OmniGen2"

    def generate(self, pipeline, input_images, dtype, seed, width, height, num_inference_step, text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end, 
                      num_images_per_prompt, instruction, negative_prompt):

        # Initialize accelerator
        accelerator = Accelerator(mixed_precision=dtype if dtype != 'fp32' else 'no')
                          
        results = run(pipeline, input_images, seed, width, height, num_inference_step, text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end, 
                      num_images_per_prompt, accelerator, instruction, negative_prompt)
                      
        images = [to_tensor(image) * 2 - 1 for image in results.images]
        collage = create_collage(images)
        
        return (images, collage,)



