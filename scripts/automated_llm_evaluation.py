"""
Automated LLM-based Image Evaluation Script.

Evaluates generated images using an LLM, automating the human annotation process.
Supports three evaluation modes:
1. Binary QA (yes/no/unsure)
2. Anchor Likert (0-5 score with reference images)
3. Text Rendering Accuracy (word-level or Likert)

Usage:
    python llm_evaluate_images.py --task-id full_evaluation
    python llm_evaluate_images.py --task-id full_evaluation --model-filter flux2-dev --prompt-filter 92
    python llm_evaluate_images.py --task-id full_evaluation --dry-run
"""

import argparse
import base64
import json
import os
import re
import sys
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ASSETS_DIR,
    GENERATED_IMAGES_DIR,
    ANCHORS_DIR,
    ANNOTATIONS_DIR,
)
from utils.prompt_utils import load_collection_file

load_dotenv()

# ─── Debug configuration ──────────────────────────────────────────────────────

debug_config = {
    "enabled": False,
    "output_dir": None,  # Will be set to <output_base>/debug/
}


def debug_save_image(b64_data: str, filename: str):
    """Save a base64-encoded image to the debug directory."""
    if not debug_config["enabled"] or not debug_config["output_dir"]:
        return
    debug_dir: Path = debug_config["output_dir"]
    debug_dir.mkdir(parents=True, exist_ok=True)
    img_path = debug_dir / filename
    img_bytes = base64.b64decode(b64_data)
    with open(img_path, "wb") as f:
        f.write(img_bytes)
    print(f"    [DEBUG] Saved image: {img_path}")


def debug_log_prompt(prompt_text: str, label: str = ""):
    """Print the prompt text being sent to the LLM."""
    if not debug_config["enabled"]:
        return
    print(f"    [DEBUG] ── Prompt ({label}) ──")
    for line in prompt_text.strip().splitlines():
        print(f"    [DEBUG]   {line}")
    print(f"    [DEBUG] ── End Prompt ──")


def debug_log_response(response, label: str = ""):
    """Print the raw pydantic response from the LLM."""
    if not debug_config["enabled"]:
        return
    print(f"    [DEBUG] ── Response ({label}) ──")
    print(f"    [DEBUG]   {response.model_dump_json(indent=2)}")
    print(f"    [DEBUG] ── End Response ──")


# ─── Pydantic response models ─────────────────────────────────────────────────

class BinaryAnswer(BaseModel):
    """Single binary QA answer."""
    uid: str
    answer: str  # 'yes', 'no', or 'unsure'


class BinaryResponse(BaseModel):
    """Batch binary QA response."""
    answers: List[BinaryAnswer]


class AnchorScore(BaseModel):
    """Single anchor Likert score."""
    uid: str
    score: str  # '0'-'5' or 'unsure'


class AnchorResponse(BaseModel):
    """Batch anchor Likert response."""
    scores: List[AnchorScore]


class WordAccuracy(BaseModel):
    """Single word rendering accuracy."""
    key: str       # e.g. "0_hello"
    correct: str   # 'yes' or 'no'


class TextArtifact(BaseModel):
    """Single text artifact (gap/spacing between words)."""
    index: str     # e.g. "0", "1" ...
    has_artifact: str  # 'yes' or 'no'


class TextRenderingResponse(BaseModel):
    """Response for word-level text rendering evaluation."""
    words: List[WordAccuracy]
    text_artifacts: List[TextArtifact]


class TextLikertResponse(BaseModel):
    """Response for Likert-based text rendering evaluation."""
    score: int  # 0-5


class AestheticsResponse(BaseModel):
    """Overall image aesthetics Likert score."""
    score: int  # 1-5


# ─── OpenAI client setup ──────────────────────────────────────────────────────

from utils.llm_client import create_client


# ─── Image utilities ─────────────────────────────────────────────────────────

def image_to_base64(image_path: Path) -> str:
    """Read an image file and return base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_best_image_path(dataset_version: str, prompt_id: str, model_name: str) -> Optional[Path]:
    """Find the image path for a given prompt and model, prioritizing .webp > .png > .jpg"""
    version_dir = GENERATED_IMAGES_DIR / dataset_version
    base_name = f"{prompt_id}_{model_name}"
    for ext in [".webp", ".png", ".jpg", ".jpeg"]:
        path = version_dir / f"{base_name}{ext}"
        if path.exists():
            return path
    return None


def create_anchor_grid(
    generated_image_path: Path,
    anchor_image_paths: List[Path],
    target_height: int = 512,
) -> str:
    """Create a side-by-side grid image with generated + anchor images.
    
    Returns the base64-encoded PNG of the grid.
    """
    images = []
    labels = ["Generated Image"]

    # Load generated image
    gen_img = Image.open(generated_image_path).convert("RGB")
    images.append(gen_img)

    # Load anchor images (up to 2)
    for i, anchor_path in enumerate(anchor_image_paths[:2]):
        anchor_img = Image.open(anchor_path).convert("RGB")
        images.append(anchor_img)
        labels.append(f"Reference {i + 1}")

    # Resize all images to the same height
    resized = []
    for img in images:
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        resized.append(img.resize((new_width, target_height), Image.LANCZOS))

    # Calculate grid dimensions
    label_height = 30
    padding = 10
    total_width = sum(img.width for img in resized) + padding * (len(resized) - 1)
    total_height = target_height + label_height

    # Create the grid
    grid = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to use a readable font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    x_offset = 0
    for img, label in zip(resized, labels):
        # Paste image
        grid.paste(img, (x_offset, label_height))
        # Draw label
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x_offset + (img.width - text_width) // 2
        draw.text((text_x, 5), label, fill=(0, 0, 0), font=font)
        x_offset += img.width + padding

    # Encode to base64
    buffer = io.BytesIO()
    grid.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def get_anchor_images(dataset_version: str, prompt_id: str, uid: str) -> List[Path]:
    """Get anchor reference images for a specific question.
    
    Anchor images are stored at: anchors/<version>/<prompt_id>/<uid>_1.png, <uid>_2.png, etc.
    """
    anchor_dir = ANCHORS_DIR / dataset_version / str(prompt_id)
    if not anchor_dir.exists():
        return []

    # Gather all anchor images matching the uid pattern
    images = []
    for img_file in sorted(anchor_dir.glob(f"{uid}_*")):
        if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            images.append(img_file)

    return images[:2]  # Limit to 2 anchors


# ─── Annotation classification ────────────────────────────────────────────────

def classify_annotations(
    annotations: List[Dict[str, Any]],
    anchor_config: Dict[str, Any],
    prompt_id: str,
    use_anchor: bool,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Classify annotations into binary QA, anchor questions, and text rendering.
    
    Returns:
        (binary_questions, anchor_questions, text_rendering_questions)
    """
    binary_questions = []
    anchor_questions = []
    text_rendering_questions = []

    anchor_annotations = anchor_config.get("annotations", [])

    for annot in annotations:
        skill = annot.get("skill", "")
        subskill = annot.get("subskill", "")
        uid = annot.get("uid", "")

        # Text rendering questions
        if skill == "text_rendering" and subskill in ("rendering_accuracy", "numerical"):
            text_rendering_questions.append(annot)
            continue

        # Check if this question has an anchor
        has_anchor = False
        if use_anchor:
            anchor_entry = next(
                (a for a in anchor_annotations
                 if str(a.get("prompt_id")) == str(prompt_id) and a.get("uid") == uid),
                None
            )
            if anchor_entry and anchor_entry.get("needs_anchor") is True:
                has_anchor = True

        if has_anchor:
            anchor_questions.append(annot)
        else:
            binary_questions.append(annot)

    return binary_questions, anchor_questions, text_rendering_questions


# ─── Evaluation functions ─────────────────────────────────────────────────────

def evaluate_aesthetics(
    client, model_id: str, image_b64: str,
    prompt_id: str = "", model_name: str = "",
) -> int:
    """Evaluate overall image aesthetics as a single Likert score (1-5)."""
    prompt_text = (
        "Rate the overall aesthetics, visual quality, and attractiveness of this image on a scale from 1 to 5:\n"
        "  1 = Very poor quality, unattractive, or heavily distorted\n"
        "  2 = Poor quality, noticeable flaws or unattractive\n"
        "  3 = Average quality, acceptable but not particularly impressive\n"
        "  4 = Good quality, visually pleasing with minor to no flaws\n"
        "  5 = Excellent quality, highly attractive and visually striking\n"
    )

    debug_log_prompt(prompt_text, f"aesthetics p{prompt_id}_{model_name}")
    debug_save_image(image_b64, f"p{prompt_id}_{model_name}_aesthetics.png")

    response = client.chat.completions.create(
        model=model_id,
        response_model=AestheticsResponse,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
    )

    debug_log_response(response, f"aesthetics p{prompt_id}_{model_name}")

    return max(1, min(5, response.score))


def evaluate_binary_qa(
    client, model_id: str, image_b64: str, questions: List[Dict[str, Any]],
    prompt_id: str = "", model_name: str = "",
) -> Dict[str, str]:
    """Evaluate binary QA questions (yes/no/unsure) in one batch."""
    if not questions:
        return {}

    questions_text = (
        "Answer the following questions about the image with only 'yes', 'no', or 'unsure'.\n"
        "For questions about presence of objects, if the prompt says a/an <object>, "
        "but the image has more, this should be flagged as wrong:\n\n"
    )
    for annot in questions:
        questions_text += f"- UID: {annot['uid']}\n  Question: {annot['question']}\n\n"

    debug_log_prompt(questions_text, f"binary_qa p{prompt_id}_{model_name}")
    debug_save_image(image_b64, f"p{prompt_id}_{model_name}_binary_qa.png")

    response = client.chat.completions.create(
        model=model_id,
        response_model=BinaryResponse,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": questions_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
    )

    debug_log_response(response, f"binary_qa p{prompt_id}_{model_name}")

    results = {}
    for answer in response.answers:
        ans = answer.answer.lower().strip()
        if "yes" in ans:
            results[answer.uid] = "yes"
        elif "no" in ans:
            results[answer.uid] = "no"
        else:
            results[answer.uid] = "unsure"

    # Fill missing
    for annot in questions:
        if annot["uid"] not in results:
            results[annot["uid"]] = "unsure"

    return results


def evaluate_anchor_likert(
    client, model_id: str,
    generated_image_path: Path,
    questions: List[Dict[str, Any]],
    dataset_version: str,
    prompt_id: str,
    use_anchor: bool,
) -> Dict[str, Any]:
    """Evaluate anchor Likert questions (0-5 score) one by one."""
    if not questions:
        return {}

    results = {}
    for annot in questions:
        uid = annot["uid"]
        question = annot["question"]
        phrase = annot.get("phrase", "")

        if use_anchor:
            # Get anchor images
            anchor_images = get_anchor_images(dataset_version, prompt_id, uid)
            if anchor_images:
                # Create grid image
                grid_b64 = create_anchor_grid(generated_image_path, anchor_images)
                prompt_text = (
                    f"You are evaluating a generated image against reference images.\n"
                    f"The leftmost image is the generated image. The other image(s) are reference examples.\n"
                    f"Question UID: {uid}\n"
                    f"Question: {question}\n"
                    f"Phrase being evaluated: \"{phrase}\"\n\n"
                    f"Rate how well the generated image matches the reference for this aspect "
                    f"on a scale from 0 to 5:\n"
                    f"  0 = completely wrong / not present\n"
                    f"  1 = barely recognizable\n"
                    f"  2 = somewhat resembles\n"
                    f"  3 = moderately accurate\n"
                    f"  4 = mostly accurate\n"
                    f"  5 = perfectly matches\n"
                    f"Or answer 'unsure' if you cannot determine.\n"
                )
                image_b64 = grid_b64
            else:
                # No anchor images found, fall back to no-anchor mode
                image_b64 = image_to_base64(generated_image_path)
                prompt_text = (
                    f"Evaluate the following aspect of this generated image.\n"
                    f"Question UID: {uid}\n"
                    f"Question: {question}\n"
                    f"Phrase being evaluated: \"{phrase}\"\n\n"
                    f"Rate on a scale from 0 to 5:\n"
                    f"  0 = completely wrong / not present\n"
                    f"  1 = barely recognizable\n"
                    f"  2 = somewhat resembles\n"
                    f"  3 = moderately accurate\n"
                    f"  4 = mostly accurate\n"
                    f"  5 = perfectly matches\n"
                    f"Or answer 'unsure' if you cannot determine.\n"
                )
        else:
            # No anchor mode
            image_b64 = image_to_base64(generated_image_path)
            prompt_text = (
                f"Evaluate the following aspect of this generated image.\n"
                f"Question UID: {uid}\n"
                f"Question: {question}\n"
                f"Phrase being evaluated: \"{phrase}\"\n\n"
                f"Rate on a scale from 0 to 5:\n"
                f"  0 = completely wrong / not present\n"
                f"  1 = barely recognizable\n"
                f"  2 = somewhat resembles\n"
                f"  3 = moderately accurate\n"
                f"  4 = mostly accurate\n"
                f"  5 = perfectly matches\n"
                f"Or answer 'unsure' if you cannot determine.\n"
            )

        debug_log_prompt(prompt_text, f"anchor_likert p{prompt_id}_{uid}")
        debug_save_image(image_b64, f"p{prompt_id}_anchor_uid{uid}.png")

        response = client.chat.completions.create(
            model=model_id,
            response_model=AnchorResponse,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
        )

        debug_log_response(response, f"anchor_likert p{prompt_id}_{uid}")

        # We evaluate these one by one, so we can just grab the first score.
        # This prevents issues where the LLM hallucinates the UID (e.g., uses the phrase instead).
        found_score = False
        for score_item in response.scores:
            if str(score_item.uid) == str(uid):
                score_val = str(score_item.score).strip().lower()
                found_score = True
                break

        # Fallback to the first item if the UID didn't match
        if not found_score and response.scores:
            score_val = str(response.scores[0].score).strip().lower()
            found_score = True

        if found_score:
            if score_val == "unsure":
                results[uid] = "unsure"
            else:
                try:
                    results[uid] = int(score_val)
                except ValueError:
                    results[uid] = "unsure"
        else:
            results[uid] = "unsure"

    return results


def evaluate_text_rendering_per_word(
    client, model_id: str, image_b64: str, annot: Dict[str, Any],
    prompt_id: str = "", model_name: str = "",
) -> Dict[str, Any]:
    """Evaluate text rendering at word level.
    
    Returns a nested dict matching the format:
    {
        "0_hello": "yes",
        "1_world": "no",
        "_text_artifacts_": { "0": "no", "1": "no", ... }
    }
    """
    phrase = annot.get("phrase", "")
    # Strip surrounding quotes
    phrase = phrase.strip('"').strip("'")
    
    # Split into words
    words = phrase.split()
    if not words:
        return {}

    # Build word list with indices
    word_entries = []
    for i, word in enumerate(words):
        key = f"{i}_{word}"
        word_entries.append({"index": i, "word": word, "key": key})

    # Number of gaps = len(words) + 1 (before first word, between each pair, after last word)
    num_gaps = len(words) + 1

    prompt_text = (
        f"You are evaluating text rendering accuracy in this image.\n"
        f"The expected text is: \"{phrase}\"\n\n"
        f"For each word below, determine if it is correctly rendered in the image "
        f"(answer 'yes' if correctly rendered, 'no' if missing, misspelled, or incorrect):\n\n"
    )
    for entry in word_entries:
        prompt_text += f"  - Key: \"{entry['key']}\" → Word: \"{entry['word']}\"\n"

    prompt_text += (
        f"\nAlso evaluate text artifacts (unwanted marks/characters in gaps between and around words).\n"
        f"There are {num_gaps} gaps (before the first word, between each pair of consecutive words, "
        f"and after the last word), indexed 0 to {num_gaps - 1}.\n"
        f"For each gap, answer 'yes' if there are unwanted text artifacts, 'no' if the gap is clean.\n"
    )

    uid = annot.get("uid", "")
    debug_log_prompt(prompt_text, f"text_per_word p{prompt_id}_{model_name}_uid{uid}")
    debug_save_image(image_b64, f"p{prompt_id}_{model_name}_text_uid{uid}.png")

    response = client.chat.completions.create(
        model=model_id,
        response_model=TextRenderingResponse,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
    )

    debug_log_response(response, f"text_per_word p{prompt_id}_{model_name}_uid{uid}")

    # Build result dict
    result = {}
    for word_acc in response.words:
        ans = word_acc.correct.lower().strip()
        result[word_acc.key] = "yes" if "yes" in ans else "no"

    # Fill any missing words
    for entry in word_entries:
        if entry["key"] not in result:
            result[entry["key"]] = "no"

    # Build text artifacts
    artifacts = {}
    for artifact in response.text_artifacts:
        ans = artifact.has_artifact.lower().strip()
        artifacts[artifact.index] = "yes" if "yes" in ans else "no"

    # Fill any missing gaps
    for i in range(num_gaps):
        if str(i) not in artifacts:
            artifacts[str(i)] = "no"

    result["_text_artifacts_"] = artifacts
    return result


def evaluate_text_rendering_likert(
    client, model_id: str, image_b64: str, annot: Dict[str, Any],
    prompt_id: str = "", model_name: str = "",
) -> int:
    """Evaluate text rendering as a single Likert score (0-5)."""
    phrase = annot.get("phrase", "")
    phrase = phrase.strip('"').strip("'")

    prompt_text = (
        f"You are evaluating text rendering accuracy in this image.\n"
        f"The expected text is: \"{phrase}\"\n\n"
        f"Rate the overall text rendering accuracy on a scale from 0 to 5:\n"
        f"  0 = No text visible at all\n"
        f"  1 = Text barely recognizable\n"
        f"  2 = Some characters correct but many errors\n"
        f"  3 = Most characters correct but noticeable errors\n"
        f"  4 = Nearly perfect with minor issues\n"
        f"  5 = Perfect rendering\n"
    )

    uid = annot.get("uid", "")
    debug_log_prompt(prompt_text, f"text_likert p{prompt_id}_{model_name}_uid{uid}")
    debug_save_image(image_b64, f"p{prompt_id}_{model_name}_text_likert_uid{uid}.png")

    response = client.chat.completions.create(
        model=model_id,
        response_model=TextLikertResponse,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
    )

    debug_log_response(response, f"text_likert p{prompt_id}_{model_name}_uid{uid}")

    return max(0, min(5, response.score))


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_annotation_tasks() -> List[Dict[str, Any]]:
    """Load annotation tasks from config file."""
    tasks_file = ASSETS_DIR / "annotation_tasks.json"
    if not tasks_file.exists():
        return []
    with open(tasks_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_anchor_config(dataset_version: str) -> Dict[str, Any]:
    """Load anchor configuration for a dataset version."""
    config_path = ANCHORS_DIR / dataset_version / "needs_anchor.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_annotations(output_path: Path) -> Dict[str, Any]:
    """Load existing annotations from file."""
    if not output_path.exists():
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(output_path: Path, annotator_key: str, answers: Dict[str, Any]):
    """Save annotations to file, merging with existing data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_annotations(output_path)
    existing[annotator_key] = answers

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ─── Main processing ─────────────────────────────────────────────────────────

def process_image(
    client,
    model_id: str,
    generated_image_path: Path,
    prompt_data: Dict[str, Any],
    anchor_config: Dict[str, Any],
    dataset_version: str,
    prompt_id: str,
    use_anchor: bool,
    text_eval_mode: str,
    model_name: str = "",
    existing_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Process a single image through all evaluation modes.
    
    Returns a merged dict of all answers keyed by uid.
    """
    if existing_results is None:
        existing_results = {}

    annotations = prompt_data.get("annotations", [])
    
    # Classify annotations
    binary_qs, anchor_qs, text_qs = classify_annotations(
        annotations, anchor_config, prompt_id, use_anchor
    )

    # Filter out already answered questions
    binary_qs = [q for q in binary_qs if q["uid"] not in existing_results]
    anchor_qs = [q for q in anchor_qs if q["uid"] not in existing_results]
    text_qs = [q for q in text_qs if q["uid"] not in existing_results]
    needs_aesthetics = "_aesthetics_rating" not in existing_results

    # Load image once if needed
    needs_image_b64 = bool(binary_qs) or bool(text_qs) or needs_aesthetics
    image_b64 = image_to_base64(generated_image_path) if needs_image_b64 else ""

    all_results = existing_results.copy()

    # 1. Binary QA
    if binary_qs:
        print(f"    → Binary QA: {len(binary_qs)} questions")
        binary_results = evaluate_binary_qa(
            client, model_id, image_b64, binary_qs,
            prompt_id=prompt_id, model_name=model_name,
        )
        all_results.update(binary_results)

    # 2. Anchor Likert (sent one by one with grid images)
    if anchor_qs:
        print(f"    → Anchor Likert: {len(anchor_qs)} questions")
        anchor_results = evaluate_anchor_likert(
            client, model_id, generated_image_path, anchor_qs,
            dataset_version, prompt_id, use_anchor,
        )
        all_results.update(anchor_results)

    # 3. Text Rendering
    if text_qs:
        for tq in text_qs:
            uid = tq["uid"]
            if text_eval_mode == "likert":
                print(f"    → Text Rendering Likert: uid={uid}")
                score = evaluate_text_rendering_likert(
                    client, model_id, image_b64, tq,
                    prompt_id=prompt_id, model_name=model_name,
                )
                all_results[uid] = score
            else:
                print(f"    → Text Rendering Per-Word: uid={uid}")
                word_results = evaluate_text_rendering_per_word(
                    client, model_id, image_b64, tq,
                    prompt_id=prompt_id, model_name=model_name,
                )
                all_results[uid] = word_results

    # 4. Aesthetics Evaluation
    if needs_aesthetics:
        print(f"    → Aesthetics Evaluation")
        score = evaluate_aesthetics(
            client, model_id, image_b64,
            prompt_id=prompt_id, model_name=model_name,
        )
        all_results["_aesthetics_rating"] = score

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Automated LLM-based Image Evaluation")
    parser.add_argument("--task-id", required=True, help="Task ID from annotation_tasks.json")
    parser.add_argument("--model-filter", nargs="*", help="Filter to specific image generation models")
    parser.add_argument("--prompt-filter", nargs="*", type=int, help="Filter to specific prompt IDs")
    parser.add_argument("--annotator-name", help="Custom annotator key (default: llm_judge:<model_id>)")
    parser.add_argument("--use-anchor", action="store_true", default=True,
                        help="Use anchor images for anchor-eligible questions (default)")
    parser.add_argument("--no-use-anchor", action="store_true",
                        help="Disable anchor images (evaluate without reference)")
    parser.add_argument("--text-eval-mode", choices=["per_word", "likert"], default="per_word",
                        help="Text rendering evaluation mode (default: per_word)")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate even if annotation exists")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated, no API calls")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: save images sent to LLM and print prompts/responses")
    parser.add_argument("--llm-model", default="gpt-5-mini", help="LLM model name (default: gpt-5-mini)")
    args = parser.parse_args()

    use_anchor = not args.no_use_anchor

    # Load task config
    tasks = load_annotation_tasks()
    task = next((t for t in tasks if t["id"] == args.task_id), None)
    if not task:
        print(f"Error: Task '{args.task_id}' not found in annotation_tasks.json")
        sys.exit(1)

    dataset_version = task["dataset_version"]
    prompts_file = task["prompts_file"]
    task_models = task.get("models", [])

    # Apply model filter
    if args.model_filter:
        models_to_process = [m for m in task_models if m in args.model_filter]
    else:
        models_to_process = task_models

    if not models_to_process:
        print("Error: No models to process")
        sys.exit(1)

    # Load prompts
    prompts_path = ASSETS_DIR / "generation_prompts" / dataset_version / prompts_file
    prompts = load_collection_file(prompts_path, dataset_version)

    # Apply prompt filter
    if args.prompt_filter:
        prompts = [p for p in prompts if p["prompt_id"] in args.prompt_filter]

    if not prompts:
        print("Error: No prompts to process")
        sys.exit(1)

    # Load anchor config
    anchor_config = load_anchor_config(dataset_version)

    # Output directory: annotations/<task_id>_llm/vqa/<model>/<prompt_id>.json
    output_base = ANNOTATIONS_DIR / f"{args.task_id}_llm" / "vqa"

    # Setup debug mode
    if args.debug:
        debug_config["enabled"] = True
        debug_config["output_dir"] = ANNOTATIONS_DIR / f"{args.task_id}_llm" / "debug"
        debug_config["output_dir"].mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Debug mode enabled. Saving debug output to: {debug_config['output_dir']}")

    # Determine annotator key and model ID
    if not args.dry_run:
        client, model_id = create_client(args.llm_model)
    else:
        model_id = MODEL_NAME_TO_AZURE.get(args.llm_model, args.llm_model)
        client = None

    annotator_key = args.annotator_name or f"llm_judge:{model_id}"

    print(f"\n{'='*60}")
    print(f"Task: {args.task_id}")
    print(f"Dataset: {dataset_version}")
    print(f"Models: {models_to_process}")
    print(f"Prompts: {len(prompts)}")
    print(f"Annotator key: {annotator_key}")
    print(f"Use anchor: {use_anchor}")
    print(f"Text eval mode: {args.text_eval_mode}")
    print(f"Debug mode: {args.debug}")
    print(f"Output dir: {output_base}")
    print(f"{'='*60}\n")

    # Process each prompt × model combination
    total = len(prompts) * len(models_to_process)
    processed = 0
    skipped = 0
    errors = 0

    for prompt_data in prompts:
        prompt_id = str(prompt_data["prompt_id"])
        prompt_text = prompt_data.get("prompt", "")[:80]

        for model in models_to_process:
            processed += 1
            label = f"[{processed}/{total}] prompt={prompt_id} model={model}"

            # Check if image exists
            image_path = get_best_image_path(dataset_version, prompt_id, model)
            if not image_path:
                print(f"{label} — SKIP (no image)")
                skipped += 1
                continue

            # Check if already annotated
            output_path = output_base / model / f"{prompt_id}.json"
            existing_annotator_data = {}
            if not args.overwrite:
                existing = load_existing_annotations(output_path)
                if annotator_key in existing:
                    existing_annotator_data = existing[annotator_key]
                    annotations = prompt_data.get("annotations", [])
                    all_uids_present = all(q["uid"] in existing_annotator_data for q in annotations)
                    if all_uids_present and "_aesthetics_rating" in existing_annotator_data:
                        print(f"{label} — SKIP (already annotated)")
                        skipped += 1
                        continue

            # Classify annotations for dry-run info
            annotations = prompt_data.get("annotations", [])
            binary_qs, anchor_qs, text_qs = classify_annotations(
                annotations, anchor_config, prompt_id, use_anchor
            )

            # Filter for logging
            binary_qs_todo = [q for q in binary_qs if q["uid"] not in existing_annotator_data]
            anchor_qs_todo = [q for q in anchor_qs if q["uid"] not in existing_annotator_data]
            text_qs_todo = [q for q in text_qs if q["uid"] not in existing_annotator_data]
            needs_aesthetics = "_aesthetics_rating" not in existing_annotator_data

            if args.dry_run:
                print(f"{label} — WOULD EVALUATE")
                print(f"    Prompt: {prompt_text}...")
                print(f"    Binary QA: {len(binary_qs_todo)}, Anchor: {len(anchor_qs_todo)}, Text: {len(text_qs_todo)}, Aesthetics: {needs_aesthetics}")
                for aq in anchor_qs_todo:
                    anchors = get_anchor_images(dataset_version, prompt_id, aq["uid"])
                    print(f"    Anchor uid={aq['uid']}: {len(anchors)} reference images")
                continue

            # Actually evaluate
            print(f"{label} — Evaluating...")
            try:
                results = process_image(
                    client=client,
                    model_id=model_id,
                    generated_image_path=image_path,
                    prompt_data=prompt_data,
                    anchor_config=anchor_config,
                    dataset_version=dataset_version,
                    prompt_id=prompt_id,
                    use_anchor=use_anchor,
                    text_eval_mode=args.text_eval_mode,
                    model_name=model,
                    existing_results=existing_annotator_data,
                )
                save_annotations(output_path, annotator_key, results)
                print(f"    ✓ Saved {len(results)} answers")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                errors += 1

    print(f"\n{'='*60}")
    print(f"Done! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
