# %%
import json
import os
import sys

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import openai
import asyncio
from dotenv import load_dotenv
import instructor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.skills import TaggedPromptWithVQA
from config import TAGGING_PROMPTS_DIR, GENERATION_PROMPTS_DIR, GECKO_PROMPTS_FILE


# -------------------------
# Configuration
# -------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


load_dotenv()
TEST_RUN = True
NUM_SAMPLES = 1000
SEED = 42

SAVE_BATCH_SIZE = 10  # Save every N processed prompts (set to 1 for save after each prompt)
MODEL_NAME = "gpt-5-mini"
PROMPT_VERSION = "v8.1"
PROMPT_PATH = TAGGING_PROMPTS_DIR / f"prompt_{PROMPT_VERSION}.txt"

MODEL_ID = MODEL_NAME
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
    )
client = instructor.from_openai(openai.AsyncOpenAI(api_key=api_key))
logger.info(f"Using OpenAI API with model: {MODEL_ID}")


with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

OUT_DIR = GENERATION_PROMPTS_DIR / f"{PROMPT_VERSION}-{MODEL_NAME}"
PROCESSED_PROMPTS_FILE = OUT_DIR / "processed_prompts.json"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# Global dictionary to store all results (loaded and new)
all_results = {}

# Counter for batch saves and lock for thread safety
processed_count = 0
save_lock = asyncio.Lock()  # Lock for thread-safe batch saves


# %%
# -------------------------
# Helper functions
# -------------------------
def load_processed_prompts():
    """Load existing prompts from the JSON file if it exists.

    Returns:
        dict: Dictionary mapping prompt_id to result dict
    """
    if os.path.exists(PROCESSED_PROMPTS_FILE):
        try:
            with open(PROCESSED_PROMPTS_FILE, "r", encoding="utf-8") as f:
                results_list = json.load(f)
            # Convert list to dict keyed by prompt_id for easy lookup
            results_dict = {result["prompt_id"]: result for result in results_list}
            logger.info(f"Loaded {len(results_dict)} processed prompts from {PROCESSED_PROMPTS_FILE}")
            return results_dict
        except Exception as e:
            logger.warning(f"Failed to load processed prompts: {e}. Starting fresh.")
            return {}
    else:
        logger.info(f"No processed prompts file found at {PROCESSED_PROMPTS_FILE}. Starting fresh.")
        return {}


async def save_processed_prompts(force: bool = False):
    """Save all results to the JSON file.

    Args:
        force: If True, save regardless of batch size
    """
    global processed_count

    async with save_lock:
        # Only save if we've reached the batch size or force is True
        if not force and processed_count < SAVE_BATCH_SIZE:
            return

        try:
            # Convert dict back to list for saving
            results_list = list(all_results.values())
            with open(PROCESSED_PROMPTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results_list, f, ensure_ascii=False, indent=1)
            logger.info(
                f"Saved {len(results_list)} processed prompts to {PROCESSED_PROMPTS_FILE} (batch of {processed_count})"
            )
            processed_count = 0  # Reset counter after save
        except Exception as e:
            logger.error(f"Failed to save processed prompts: {e}")


# %%
# -------------------------
# Load dataset
# -------------------------
prompts_df = pd.read_csv(GECKO_PROMPTS_FILE)

if TEST_RUN:
    prompts_df = prompts_df.sample(n=NUM_SAMPLES, random_state=SEED, replace=False).reset_index(drop=True)


# %%
async def run_tagging_and_vqa_generation(row_dict: dict, skip_existing: bool = True) -> Optional[dict]:
    """Process a single prompt with error handling.

    Returns:
        dict: Result dictionary on success
        None: On failure (error is logged)
    """
    prompt_id = row_dict.get("prompt_id", "unknown")

    try:
        # Check if already processed in loaded results
        if skip_existing and prompt_id in all_results:
            logger.info(f"Prompt `{prompt_id}` is already processed. Skipping.")
            return all_results[prompt_id]

        prompt_text = row_dict.get("prompt", "")
        # logger.info(f"Processing prompt `{prompt_id}`: {prompt_text}")

        # Call LLM API
        tagging_result = await client.chat.completions.create(
            model=MODEL_ID,
            response_model=TaggedPromptWithVQA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
        )

        # Create output dict
        output_dict = {
            "prompt_id": prompt_id,
            "prompt": row_dict.get("prompt"),
            "prompt_type": row_dict.get("prompt_type"),
            "dataset_id": row_dict.get("dataset_id"),
            "prompt_version": PROMPT_VERSION,
            "llm_model": MODEL_NAME,
        }
        annotations = tagging_result.model_dump().get("annotations", [])

        output_dict.update({"annotations": annotations})

        # Add to global results
        all_results[prompt_id] = output_dict

        # Increment counter and save in batches
        global processed_count
        async with save_lock:
            processed_count += 1

        await save_processed_prompts(force=False)

        logger.info(f"Successfully processed prompt `{prompt_id}`")
        return output_dict

    except Exception as e:
        logger.error(f"Failed to process prompt `{prompt_id}`: {type(e).__name__}: {e}")
        return None


async def main():
    global all_results, processed_count

    # Reset counter
    processed_count = 0

    # Load existing results at startup
    all_results = load_processed_prompts()

    logger.info(f"Starting processing of {len(prompts_df)} prompts")
    logger.info(f"Batch save configured: saving every {SAVE_BATCH_SIZE} prompts")

    tasks = [run_tagging_and_vqa_generation(row.to_dict(), skip_existing=True) for (_, row) in prompts_df.iterrows()]

    # Use return_exceptions=True to continue processing even if some tasks fail
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful and failed tasks
    successful_count = 0
    failed_count = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {i} raised an exception: {type(result).__name__}: {result}")
            failed_count += 1
        elif result is None:
            failed_count += 1
        else:
            successful_count += 1

    logger.info(f"Processing complete: {successful_count} successful, {failed_count} failed")

    # Final save to ensure all results are persisted
    logger.info("Performing final save...")
    await save_processed_prompts(force=True)

    logger.info(f"Total results in file: {len(all_results)}")


if __name__ == "__main__":
    asyncio.run(main())
