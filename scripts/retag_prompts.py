import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional
import asyncio
from dotenv import load_dotenv
import instructor
import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.skills import TaggedPromptWithVQA
from config import TAGGING_PROMPTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# -------------------------
# Configuration - MODIFY THESE
# -------------------------
MODEL_NAME = "gpt-5-mini"  # Change to desired model: "gpt-5", "gpt-5-mini", "gpt-4.1"
PROMPT_VERSION = "v8.1"  # Change to desired prompt version
INPUT_JSON_PATH = r"assets\generation_prompts\v8.1-gpt-5-mini\sampled_prompts_50.json"
OUTPUT_JSON_PATH_OVERRIDE = None  # Set to a path string to override auto-generation

SAVE_BATCH_SIZE = 10
SKIP_EXISTING = True  # Set to False to re-process all prompts

# -------------------------
# Setup paths
# -------------------------
PROMPT_PATH = TAGGING_PROMPTS_DIR / f"prompt_{PROMPT_VERSION}.txt"

if OUTPUT_JSON_PATH_OVERRIDE is None:
    # Auto-generate output path: replace model name in input path
    input_path_obj = os.path.normpath(INPUT_JSON_PATH)
    input_dir = os.path.dirname(input_path_obj)
    input_filename = os.path.basename(input_path_obj)

    # Create new directory name based on new model
    parent_dir = os.path.dirname(input_dir)
    new_dir_name = f"{PROMPT_VERSION}-{MODEL_NAME}"
    output_dir = os.path.join(parent_dir, new_dir_name)

    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_JSON_PATH = os.path.join(output_dir, input_filename)
else:
    OUTPUT_JSON_PATH = OUTPUT_JSON_PATH_OVERRIDE

logger.info(f"Input JSON: {INPUT_JSON_PATH}")
logger.info(f"Output JSON: {OUTPUT_JSON_PATH}")

# -------------------------
# Initialize LLM client
# -------------------------
MODEL_ID = MODEL_NAME
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
    )
client = instructor.from_openai(openai.AsyncOpenAI(api_key=api_key))
logger.info(f"Using OpenAI API with model: {MODEL_ID}")

# Load system prompt
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Global dictionary to store all results
all_results = {}
processed_count = 0
save_lock = asyncio.Lock()


# -------------------------
# Helper functions
# -------------------------
def load_input_prompts():
    """Load prompts from input JSON file."""
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    logger.info(f"Loaded {len(prompts)} prompts from {INPUT_JSON_PATH}")
    return prompts


def load_processed_prompts():
    """Load existing processed prompts from output file if it exists."""
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
                results_list = json.load(f)
            results_dict = {result["prompt_id"]: result for result in results_list}
            logger.info(f"Loaded {len(results_dict)} already processed prompts from {OUTPUT_JSON_PATH}")
            return results_dict
        except Exception as e:
            logger.warning(f"Failed to load processed prompts: {e}. Starting fresh.")
            return {}
    else:
        logger.info(f"No existing output file found. Starting fresh.")
        return {}


async def save_processed_prompts(force: bool = False):
    """Save all results to the output JSON file."""
    global processed_count

    async with save_lock:
        if not force and processed_count < SAVE_BATCH_SIZE:
            return

        try:
            results_list = list(all_results.values())
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(results_list, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Saved {len(results_list)} processed prompts to {OUTPUT_JSON_PATH} (batch of {processed_count})"
            )
            processed_count = 0
        except Exception as e:
            logger.error(f"Failed to save processed prompts: {e}")


async def retag_prompt(prompt_dict: dict, skip_existing: bool = True) -> Optional[dict]:
    """Re-tag a single prompt with the new model.

    Args:
        prompt_dict: Dictionary containing prompt data
        skip_existing: If True, skip prompts already in output file

    Returns:
        dict: Result dictionary on success
        None: On failure
    """
    prompt_id = prompt_dict.get("prompt_id", "unknown")

    try:
        # Check if already processed
        if skip_existing and prompt_id in all_results:
            logger.info(f"Prompt `{prompt_id}` already processed. Skipping.")
            return all_results[prompt_id]

        prompt_text = prompt_dict.get("prompt", "")

        # Call LLM API
        tagging_result = await client.chat.completions.create(
            model=MODEL_ID,
            response_model=TaggedPromptWithVQA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
        )

        # Create output dict - preserve original metadata but update model info
        output_dict = {
            "prompt_id": prompt_id,
            "prompt": prompt_dict.get("prompt"),
            "prompt_type": prompt_dict.get("prompt_type"),
            "dataset_id": prompt_dict.get("dataset_id"),
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

        logger.info(f"Successfully re-tagged prompt `{prompt_id}`")
        return output_dict

    except Exception as e:
        logger.error(f"Failed to re-tag prompt `{prompt_id}`: {type(e).__name__}: {e}")
        return None


async def main():
    global all_results, processed_count

    processed_count = 0

    # Load input prompts
    input_prompts = load_input_prompts()

    # Load existing processed prompts from output file
    all_results = load_processed_prompts()

    logger.info(f"Starting re-tagging of {len(input_prompts)} prompts")
    logger.info(f"Using model: {MODEL_NAME}, prompt version: {PROMPT_VERSION}")
    logger.info(f"Batch save configured: saving every {SAVE_BATCH_SIZE} prompts")

    # Create tasks for all prompts
    tasks = [retag_prompt(prompt_dict, skip_existing=SKIP_EXISTING) for prompt_dict in input_prompts]

    # Process all tasks
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

    logger.info(f"Re-tagging complete: {successful_count} successful, {failed_count} failed")

    # Final save
    logger.info("Performing final save...")
    await save_processed_prompts(force=True)

    logger.info(f"Total results in output file: {len(all_results)}")


if __name__ == "__main__":
    asyncio.run(main())
