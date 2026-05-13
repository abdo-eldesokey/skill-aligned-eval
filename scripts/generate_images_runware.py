# %%
import json
import os
import io
from pathlib import Path
import asyncio

import aiohttp
from dotenv import load_dotenv
import numpy as np
from PIL import Image

from runware import Runware, IImageInference

# Import centralized configuration
from config import GENERATION_PROMPTS_DIR, GENERATED_IMAGES_DIR


# -------------------------
# Configuration
# -------------------------
load_dotenv()
TEST_RUN = False
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")

MODEL_ID_TO_NAME = {
    "flux2-pro": "bfl:5@1",  # 0.03$ per image (1.5$/50 images)
    "flux2-dev": "runware:400@1",
    "nano-banana-1": "google:4@1",  # 0.04$ per image (2$/50 images)
    "z-image": "runware:z-image@turbo",  # 0.0032$ per image (0.16$/50 images)
    "wan-2.5-preview": "runware:201@10",  # 0.03$ per image (1.5$/50 images)
    "seedream-4": "bytedance:5@0",  # 0.03$ per image (1.5$/50 images)
    "qwen-image": "runware:108@1", # 0.005$
    "kontext-dev": "runware:106@1",
    "nano-banana-2": "google:4@2",  # 0.138$ per image (6.9$/50 images)
    "flux1-dev": "runware:101@1",  # 0.0038$  
}

# List of models to process (can specify multiple)
MODELS = [
    # "wan-2.5-preview",
    "z-image",
    # "flux2-pro",
    "flux2-dev",
    # "nano-banana-1",
    "flux1-dev",
    # "qwen-image"
]  # Add more models like: ["z-image", "flux2-pro", "seedream-4"]
DATASET_ID = "v8.1-gpt-5-mini"

PROMPTS_PATH = GENERATION_PROMPTS_DIR / DATASET_ID / "text_rendering_collection.json"
OUT_DIR = GENERATED_IMAGES_DIR / DATASET_ID


# %%
# -------------------------
# Load dataset
# -------------------------
def load_prompts():
    """Load all prompts from JSON files in the dataset directory (new ID-only format)."""
    from utils.prompt_utils import load_collection_file
    
    prompts = load_collection_file(PROMPTS_PATH, DATASET_ID)

    print(f"Loaded {len(prompts)} prompts from {PROMPTS_PATH}")
    return prompts


# %%
# -------------------------
# Download and save images
# -------------------------
async def download_image(session: aiohttp.ClientSession, image_url: str, output_path: Path):
    """Download an image from URL and save it to the specified path."""
    async with session.get(image_url) as response:
        response.raise_for_status()
        content = await response.read()

        if output_path.suffix.lower() == ".webp":
            image = Image.open(io.BytesIO(content))
            image.save(output_path, "WEBP")
        else:
            with open(output_path, "wb") as f:
                f.write(content)


# %%
# -------------------------
# Generate images
# -------------------------
async def generate_images_for_model(model_name: str, prompts_data: list, batch_size: int = 10):
    """Generate images for a specific model concurrently in batches."""
    model_id = MODEL_ID_TO_NAME[model_name]
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name} ({model_id})")
    print(f"{'='*80}")

    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    # Create aiohttp session for downloading images
    async with aiohttp.ClientSession() as session:
        total = len(prompts_data)
        print(f"Starting image generation for {total} prompts...")

        for i in range(0, total, batch_size):
            batch = prompts_data[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"\nProcessing batch {batch_num}/{(total + batch_size - 1) // batch_size} ({len(batch)} prompts)...")

            # Create concurrent tasks for the batch
            tasks = []
            prompt_data_to_generate = []
            for prompt_data in batch:
                prompt_id = prompt_data["prompt_id"]
                
                # Check for existing image in any supported format
                existing_file = None
                for ext in [".webp", ".png", ".jpg"]:
                    check_path = OUT_DIR / f"{prompt_id}_{model_name}{ext}"
                    if check_path.exists():
                        existing_file = check_path
                        break

                # Check if image already exists
                if existing_file:
                    print(f"  ⏭️  Skipping prompt {prompt_id}: Image already exists at {existing_file}")
                    continue

                request = IImageInference(positivePrompt=prompt_data["prompt"], model=model_id, width=1024, height=1024)
                tasks.append(runware.imageInference(requestImage=request))
                prompt_data_to_generate.append(prompt_data)

            # Execute batch concurrently (only for prompts that need generation)
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = []

            # Handle results and save images
            for j, result in enumerate(results):
                prompt_data = prompt_data_to_generate[j]
                prompt_id = prompt_data["prompt_id"]

                if isinstance(result, Exception):
                    print(f"  ❌ Failed to generate image for prompt {prompt_id}: {result}")
                else:
                    try:
                        image_url = result[0].imageURL
                        # Default to .webp
                        output_path = OUT_DIR / f"{prompt_id}_{model_name}.webp"

                        # Download and save the image
                        await download_image(session, image_url, output_path)
                        print(f"  ✓ Generated and saved image for prompt {prompt_id}: {output_path}")

                    except Exception as e:
                        print(f"  ❌ Error processing result for prompt {prompt_id}: {e}")


async def generate_images_batch(prompts_data: list, batch_size: int = 10):
    """Generate images for all configured models."""
    # Ensure output directory exists
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each model sequentially (to avoid overwhelming the API)
    for model_name in MODELS:
        await generate_images_for_model(model_name, prompts_data, batch_size)


async def main():
    """Main entry point for image generation."""
    # Load all prompts
    prompts_data = load_prompts()

    if TEST_RUN:
        print(f"\n⚠️  TEST_RUN mode: limiting to first 5 prompts")
        prompts_data = prompts_data[:5]

    # Generate images
    await generate_images_batch(prompts_data, batch_size=10)

    print(f"\n✅ Image generation complete!")


if __name__ == "__main__":
    asyncio.run(main())
