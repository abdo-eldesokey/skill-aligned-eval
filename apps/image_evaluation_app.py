"""
Flask backend for Image Generation Evaluation Tool
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import threading
from dotenv import load_dotenv
from pydantic import BaseModel


# Import centralized configuration
from config import (
    BASE_DIR,
    ASSETS_DIR,
    GENERATED_IMAGES_DIR,
    GENERATION_PROMPTS_DIR,
    ANNOTATORS_FILE,
    ANCHORS_DIR,
    IMAGE_EVALUATION_STATIC,
    ANNOTATIONS_DIR,
    AI_ANSWERS_DIR,
)

load_dotenv()

app = Flask(__name__, static_folder=str(IMAGE_EVALUATION_STATIC), static_url_path="")
CORS(app)

# Global state for batch VQA processing
batch_vqa_state = {
    "is_running": False,
    "cancel_requested": False,
    "total": 0,
    "processed": 0,
    "current_image": "",
    "lock": threading.Lock(),
}

# Ensure directories exist
ANCHORS_DIR.mkdir(exist_ok=True, parents=True)

from utils.llm_client import get_client

MODEL_NAME = "gpt-5-mini"
MODEL_ID = MODEL_NAME


class VQAAnswer(BaseModel):
    """Single VQA answer."""

    uid: str
    answer: str  # 'yes' or 'no'


class VQAResponse(BaseModel):
    """Batch VQA response."""

    answers: List[VQAAnswer]


def load_annotators() -> List[str]:
    """Load list of annotators from config file (excluding llm_judge)."""
    if not ANNOTATORS_FILE.exists():
        return ["annotator_01", "annotator_02"]

    with open(ANNOTATORS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        annotators = data.get("annotators", [])
        # Exclude llm_judge from the list
        return [a for a in annotators if a != "llm_judge"]


# Annotation tasks config file
ANNOTATION_TASKS_FILE = ASSETS_DIR / "annotation_tasks.json"


def load_annotation_tasks() -> List[Dict[str, Any]]:
    """Load annotation tasks from config file."""
    if not ANNOTATION_TASKS_FILE.exists():
        return []

    try:
        with open(ANNOTATION_TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def get_dataset_versions() -> List[str]:
    """Get list of available dataset version directories."""
    versions = []
    if not GENERATION_PROMPTS_DIR.exists():
        return versions

    for version_dir in GENERATION_PROMPTS_DIR.iterdir():
        if version_dir.is_dir():
            versions.append(version_dir.name)

    return sorted(versions)


def get_prompt_files(dataset_version: str) -> List[str]:
    """Get list of JSON files in a dataset version directory."""
    version_dir = GENERATION_PROMPTS_DIR / dataset_version
    if not version_dir.exists():
        return []

    json_files = []
    for f in version_dir.glob("*.json"):
        json_files.append(f.name)

    return sorted(json_files)


def load_prompts_from_file(dataset_version: str, filename: str) -> List[Dict[str, Any]]:
    """Load prompts from a collection file (new ID-only format).
    
    Collection files contain prompt IDs that reference the source file (processed_prompts.json).
    """
    from utils.prompt_utils import load_collection_file
    
    file_path = GENERATION_PROMPTS_DIR / dataset_version / filename
    if not file_path.exists():
        return []

    try:
        return load_collection_file(file_path, dataset_version)
    except (json.JSONDecodeError, IOError, FileNotFoundError) as e:
        print(f"Error loading prompts from {filename}: {e}")
        return []


def get_available_models() -> List[str]:
    """Get list of available model directories."""
    models = []
    if not GENERATED_IMAGES_DIR.exists():
        return models

    for model_dir in GENERATED_IMAGES_DIR.iterdir():
        if model_dir.is_dir():
            models.append(model_dir.name)

    return sorted(models)


def get_best_image_path(dataset_version: str, prompt_id: str, model_name: str) -> Optional[Path]:
    """Find the image path for a given prompt and model, prioritizing .webp > .png > .jpg"""
    version_dir = GENERATED_IMAGES_DIR / dataset_version
    base_name = f"{prompt_id}_{model_name}"
    
    # Check extensions in order of preference
    for ext in [".webp", ".png", ".jpg", ".jpeg"]:
        path = version_dir / f"{base_name}{ext}"
        if path.exists():
            return path
            
    return None


def get_images_for_prompt(
    prompt_id: str, dataset_version: str, models: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Get all images for a specific prompt_id across specified models.

    Directory structure: <dataset_version>/<prompt_id>_<model_name>.png

    Args:
        prompt_id: The prompt ID to find images for
        dataset_version: The dataset version directory
        models: Optional list of models to filter by. If None, check all models.

    Returns:
        List of image metadata dictionaries with model and path info
    """
    images = []

    version_dir = GENERATED_IMAGES_DIR / dataset_version
    if not version_dir.exists():
        return images

    # Look for files matching pattern: <prompt_id>_<model_name>.*
    # We want to support .webp, .png, .jpg
    # Prioritize .webp > .png > .jpg
    
    model_images = {}
    
    for img_file in version_dir.glob(f"{prompt_id}_*"):
        if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.webp']:
            continue
            
        filename = img_file.stem  # Remove extension
        parts = filename.split("_", 1)  # Split on first underscore
        if len(parts) != 2:
            continue
            
        model_name = parts[1]
        
        # Determine if this file should be the chosen one for this model
        if model_name not in model_images:
            model_images[model_name] = img_file
        else:
            current = model_images[model_name]
            # If current is webp, keep it
            if current.suffix.lower() == '.webp':
                continue
            # If new file is webp, replace
            if img_file.suffix.lower() == '.webp':
                model_images[model_name] = img_file
            # If current is png and new is not webp (e.g. jpg), keep png (arbitrary preference)
            # If current is jpg and new is png, swap? Let's just prefer webp > png > jpg
            elif img_file.suffix.lower() == '.png' and current.suffix.lower() != '.png':
                model_images[model_name] = img_file

    for model_name, img_file in model_images.items():
        # Filter by models if specified
        if models and model_name not in models:
            continue

        images.append(
            {
                "model": model_name,
                "prompt_id": str(prompt_id),
                "image_path": f"{dataset_version}/{img_file.name}",
            }
        )

    # Sort by model name
    images.sort(key=lambda x: x["model"])
    return images


# Annotations base directory (convert string to Path and make absolute)
ANNOTATIONS_BASE_DIR = BASE_DIR / ANNOTATIONS_DIR


def get_annotation_status(task_id: str, model: str, prompt_id: str, annotator: str) -> bool:
    """Check if an image has been annotated by a specific annotator."""
    annotation_file = ANNOTATIONS_BASE_DIR / task_id / "vqa" / model / f"{prompt_id}.json"

    if not annotation_file.exists():
        return False

    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
        return annotator in annotations


def load_annotations(task_id: str, model: str, prompt_id: str, annotator: str) -> Optional[Dict[str, Any]]:
    """Load annotations for a specific image and annotator.
    
    For LLM judge / AI annotators (keys starting with 'llm_judge:'), this function first 
    checks the centralized AI answers directory (assets/ai_answers/<model>/<prompt_id>.json)
    before falling back to the VQA annotations directory.
    
    AI answers are shared across all tasks (task-independent).
    """
    # For ChatGPT annotators, check centralized AI answers first
    if annotator.startswith("llm_judge:"):
        ai_answer_file = AI_ANSWERS_DIR / model / f"{prompt_id}.json"
        if ai_answer_file.exists():
            try:
                with open(ai_answer_file, "r", encoding="utf-8") as f:
                    ai_annotations = json.load(f)
                    if annotator in ai_annotations:
                        return ai_annotations.get(annotator)
            except (json.JSONDecodeError, IOError):
                pass  # Fall through to check VQA annotations
    
    # Check VQA annotations directory (original location)
    annotation_file = ANNOTATIONS_BASE_DIR / task_id / "vqa" / model / f"{prompt_id}.json"

    if not annotation_file.exists():
        return None

    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
        return annotations.get(annotator)


def save_annotations(task_id: str, model: str, prompt_id: str, annotator: str, answers: Dict[str, Any]):
    """Save annotations for a specific image and annotator.
    
    For LLM judge / AI annotators (keys starting with 'llm_judge:'), saves to the centralized
    AI answers directory (assets/ai_answers/<model>/<prompt_id>.json).
    
    For human annotators, saves to the VQA annotations directory.
    """
    # For ChatGPT annotators, save to centralized AI answers directory
    if annotator.startswith("llm_judge:"):
        ai_dir = AI_ANSWERS_DIR / model
        ai_dir.mkdir(exist_ok=True, parents=True)
        
        ai_answer_file = ai_dir / f"{prompt_id}.json"
        
        # Load existing AI annotations
        if ai_answer_file.exists():
            with open(ai_answer_file, "r", encoding="utf-8") as f:
                ai_annotations = json.load(f)
        else:
            ai_annotations = {}
        
        # Update annotations for this AI annotator
        ai_annotations[annotator] = answers
        
        # Save back
        with open(ai_answer_file, "w", encoding="utf-8") as f:
            json.dump(ai_annotations, f, indent=2, ensure_ascii=False)
        return
    
    # For human annotators, save to VQA directory
    vqa_dir = ANNOTATIONS_BASE_DIR / task_id / "vqa" / model
    vqa_dir.mkdir(exist_ok=True, parents=True)

    annotation_file = vqa_dir / f"{prompt_id}.json"

    # Load existing annotations
    if annotation_file.exists():
        with open(annotation_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    # Update annotations for this annotator
    annotations[annotator] = answers

    # Save back
    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def get_chatgpt_annotations(
    image_path: str, prompt_data: Dict[str, Any], preprocess_with_al: bool = True
) -> Dict[str, str]:
    """Get VQA annotations from ChatGPT for an image using batch processing.

    Args:
        image_path: Path to the image file
        prompt_data: Prompt data containing annotations
        preprocess_with_al: If True, preprocess VQA questions with Active Learning logic
    """
    # Read image and encode to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    annotations_list = prompt_data.get("annotations", [])

    # Filter out text_rendering questions with rendering_accuracy or numerical subskills
    # These require per-word human evaluation and cannot be answered by VLMs
    filtered_annotations = []
    skipped_annotations = []
    
    for annotation in annotations_list:
        skill = annotation.get("skill", "")
        subskill = annotation.get("subskill", "")
        
        # Skip text_rendering questions with rendering_accuracy or numerical subskills
        if skill == "text_rendering" and (subskill == "rendering_accuracy" or subskill == "numerical"):
            skipped_annotations.append(annotation)
        else:
            filtered_annotations.append(annotation)

    # Build questions list from filtered annotations
    if preprocess_with_al:
        questions_text = """Answer the following questions about the image with only 'yes' or 'no'. 
        Use Active Learning preprocessing: Only answer questions whose dependencies are satisfied.
        For questions about presence of objects, if the prompts says a/an <object>, but the image has more, 
        this should be flagged as wrong:\n\n
        """
    else:
        questions_text = """Answer the following questions about the image with only 'yes' or 'no'. 
        For questions about presence of objects, if the prompts says a/an <object>, but the image has more, 
        this should be flagged as wrong:\n\n
        """

    for annotation in filtered_annotations:
        questions_text += f"- UID: {annotation['uid']}\n  Question: {annotation['question']}\n\n"

    # If all questions were filtered out, return skipped markers for all
    if not filtered_annotations:
        return {annotation["uid"]: "skipped" for annotation in annotations_list}

    try:
        response = get_client(MODEL_NAME).chat.completions.create(
            model=MODEL_ID,
            response_model=VQAResponse,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": questions_text,
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                    ],
                }
            ],
        )

        # Convert to dict
        answers = {}
        for vqa_answer in response.answers:
            # Normalize answer
            answer_lower = vqa_answer.answer.lower()
            if "yes" in answer_lower:
                answers[vqa_answer.uid] = "yes"
            elif "no" in answer_lower:
                answers[vqa_answer.uid] = "no"
            else:
                answers[vqa_answer.uid] = "unknown"

        # Fill in any missing answers from filtered annotations
        for annotation in filtered_annotations:
            if annotation["uid"] not in answers:
                answers[annotation["uid"]] = "unknown"
        
        # Mark skipped annotations
        for annotation in skipped_annotations:
            answers[annotation["uid"]] = "skipped"

        return answers

    except Exception as e:
        print(f"Error getting ChatGPT annotations: {e}")
        # Return error for filtered questions, skipped for text_rendering
        answers = {}
        for annotation in filtered_annotations:
            answers[annotation["uid"]] = "error"
        for annotation in skipped_annotations:
            answers[annotation["uid"]] = "skipped"
        return answers



@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(IMAGE_EVALUATION_STATIC, "index.html")


@app.route("/api/annotators")
def get_annotators():
    """Get list of annotators."""
    annotators = load_annotators()
    return jsonify({"annotators": annotators})


@app.route("/api/annotation_tasks")
def api_get_annotation_tasks():
    """Get list of annotation tasks."""
    tasks = load_annotation_tasks()
    return jsonify({"tasks": tasks})


@app.route("/api/task/<task_id>/prompts")
def api_get_task_prompts(task_id: str):
    """Load prompts for a specific annotation task."""
    tasks = load_annotation_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    prompts = load_prompts_from_file(task["dataset_version"], task["prompts_file"])
    return jsonify({"prompts": prompts, "task": task})


@app.route("/api/task/<task_id>/images_for_prompt/<prompt_id>")
def api_get_task_images_for_prompt(task_id: str, prompt_id: str):
    """Get all images for a specific prompt based on task configuration."""
    annotator = request.args.get("annotator", "").strip()
    if not annotator:
        return jsonify({"error": "annotator query param is required"}), 400

    tasks = load_annotation_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    dataset_version = task.get("dataset_version")
    if not dataset_version:
        return jsonify({"error": "Task missing dataset_version"}), 400

    models = task.get("models", None)

    images = get_images_for_prompt(prompt_id, dataset_version, models)

    # Add annotation status for each image
    for img in images:
        img["annotated"] = get_annotation_status(task_id, img["model"], img["prompt_id"], annotator)

    return jsonify({"images": images})


@app.route("/api/image/<task_id>/<model>/<prompt_id>")
def get_image_data(task_id: str, model: str, prompt_id: str):
    """Get annotation data for a specific image (user and chatgpt annotations)."""
    annotator = request.args.get("annotator", "").strip()
    if not annotator:
        return jsonify({"error": "annotator query param is required"}), 400

    # Load user annotations
    user_annotations = load_annotations(task_id, model, prompt_id, annotator)

    # Load chatgpt annotations (with model name in key)
    chatgpt_annotations = load_annotations(task_id, model, prompt_id, f"llm_judge:{MODEL_ID}")

    # Check if chatgpt annotations exist
    has_chatgpt = chatgpt_annotations is not None

    # Check if artifact mask exists
    artifact_mask_path = ANNOTATIONS_BASE_DIR / task_id / "artifact_mask" / model / f"{prompt_id}_{annotator}.png"
    has_artifact_mask = artifact_mask_path.exists()

    # Check if artifact points exist
    artifact_points_path = ANNOTATIONS_BASE_DIR / task_id / "artifact_points" / model / f"{prompt_id}.json"
    has_artifact_points = False
    artifact_points = []
    if artifact_points_path.exists():
        try:
            with open(artifact_points_path, "r", encoding="utf-8") as f:
                all_points = json.load(f)
            if annotator in all_points:
                has_artifact_points = True
                artifact_points = all_points[annotator]
        except (json.JSONDecodeError, IOError):
            pass

    return jsonify(
        {
            "user_annotations": user_annotations or {},
            "chatgpt_annotations": chatgpt_annotations or {},
            "has_chatgpt": has_chatgpt,
            "has_artifact_mask": has_artifact_mask,
            "has_artifact_points": has_artifact_points,
            "artifact_points": artifact_points,
        }
    )


@app.route("/api/generate_chatgpt_annotations/<task_id>/<model>/<prompt_id>", methods=["POST"])
def generate_chatgpt_annotations_endpoint(task_id: str, model: str, prompt_id: str):
    """Generate ChatGPT annotations for an image."""
    # Get prompt_data, dataset_version, and preprocess_with_al from request body
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    prompt_data = data.get("prompt_data")
    dataset_version = data.get("dataset_version")
    preprocess_with_al = data.get("preprocess_with_al", True)

    if not prompt_data or not dataset_version:
        return jsonify({"error": "Prompt data or dataset_version not provided"}), 400

    # Generate annotations
    image_path = get_best_image_path(dataset_version, prompt_id, model)
    
    if not image_path or not image_path.exists():
        return jsonify({"error": "Image not found"}), 404

    try:
        annotations = get_chatgpt_annotations(str(image_path), prompt_data, preprocess_with_al)
        save_annotations(task_id, model, prompt_id, f"llm_judge:{MODEL_ID}", annotations)
        return jsonify({"success": True, "annotations": annotations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/save_annotations", methods=["POST"])
def save_annotations_endpoint():
    """Save annotations for an image."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    task_id = data.get("task_id")
    model = data.get("model")
    prompt_id = data.get("prompt_id")
    annotator = data.get("annotator")
    answers = data.get("answers")

    if not all([task_id, model, prompt_id, annotator, answers]):
        return jsonify({"error": "Missing required fields"}), 400

    save_annotations(task_id, model, prompt_id, annotator, answers)

    return jsonify({"success": True})


@app.route("/api/save_artifact_mask", methods=["POST"])
def save_artifact_mask():
    """Save artifact mask for an image."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    task_id = data.get("task_id")
    model = data.get("model")
    prompt_id = data.get("prompt_id")
    annotator = data.get("annotator")
    mask_data = data.get("mask_data")  # base64 encoded PNG

    if not all([task_id, model, prompt_id, annotator, mask_data]):
        return jsonify({"error": "Missing required fields"}), 400

    # Create artifact_mask directory
    artifact_mask_dir = ANNOTATIONS_BASE_DIR / task_id / "artifact_mask" / model
    artifact_mask_dir.mkdir(exist_ok=True, parents=True)

    # Save mask with annotator in filename
    mask_file = artifact_mask_dir / f"{prompt_id}_{annotator}.png"

    # Decode base64 and save
    mask_bytes = base64.b64decode(mask_data.split(",")[1])
    with open(mask_file, "wb") as f:
        f.write(mask_bytes)

    return jsonify({"success": True})


@app.route("/api/delete_artifact_mask", methods=["POST"])
def delete_artifact_mask():
    """Delete artifact mask for an image."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    task_id = data.get("task_id")
    model = data.get("model")
    prompt_id = data.get("prompt_id")
    annotator = data.get("annotator")

    if not all([task_id, model, prompt_id, annotator]):
        return jsonify({"error": "Missing required fields"}), 400

    # Get mask file path
    artifact_mask_dir = ANNOTATIONS_BASE_DIR / task_id / "artifact_mask" / model
    mask_file = artifact_mask_dir / f"{prompt_id}_{annotator}.png"

    # Delete if exists
    if mask_file.exists():
        mask_file.unlink()
        return jsonify({"success": True, "deleted": True})
    else:
        return jsonify({"success": True, "deleted": False})


@app.route("/api/save_artifact_points", methods=["POST"])
def save_artifact_points():
    """Save artifact point annotations for an image."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    task_id = data.get("task_id")
    model = data.get("model")
    prompt_id = data.get("prompt_id")
    annotator = data.get("annotator")
    points = data.get("points")  # list of [x, y] tuples

    if not all([task_id, model, prompt_id, annotator]) or points is None:
        return jsonify({"error": "Missing required fields"}), 400

    # Create artifact_points directory
    artifact_points_dir = ANNOTATIONS_BASE_DIR / task_id / "artifact_points" / model
    artifact_points_dir.mkdir(exist_ok=True, parents=True)

    # Load existing file (single JSON per image, keyed by annotator)
    points_file = artifact_points_dir / f"{prompt_id}.json"
    all_points = {}
    if points_file.exists():
        try:
            with open(points_file, "r", encoding="utf-8") as f:
                all_points = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_points = {}

    # Store as list of [x, y] tuples under annotator key
    all_points[annotator] = [[p[0], p[1]] if isinstance(p, list) else [p.get("x", 0), p.get("y", 0)] for p in points]

    with open(points_file, "w", encoding="utf-8") as f:
        json.dump(all_points, f, indent=2)

    return jsonify({"success": True})


@app.route("/api/delete_artifact_points", methods=["POST"])
def delete_artifact_points():
    """Delete artifact point annotations for an image."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    task_id = data.get("task_id")
    model = data.get("model")
    prompt_id = data.get("prompt_id")
    annotator = data.get("annotator")

    if not all([task_id, model, prompt_id, annotator]):
        return jsonify({"error": "Missing required fields"}), 400

    # Get points file path
    artifact_points_dir = ANNOTATIONS_BASE_DIR / task_id / "artifact_points" / model
    points_file = artifact_points_dir / f"{prompt_id}.json"

    # Remove annotator entry from shared file
    if points_file.exists():
        try:
            with open(points_file, "r", encoding="utf-8") as f:
                all_points = json.load(f)
            if annotator in all_points:
                del all_points[annotator]
                with open(points_file, "w", encoding="utf-8") as f:
                    json.dump(all_points, f, indent=2)
                return jsonify({"success": True, "deleted": True})
        except (json.JSONDecodeError, IOError):
            pass
    return jsonify({"success": True, "deleted": False})


@app.route("/api/batch_vqa/start", methods=["POST"])
def start_batch_vqa():
    """Start batch VQA processing for all images in a task."""
    global batch_vqa_state

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    task_id = data.get("task_id")
    prompts = data.get("prompts", [])

    if not task_id or not prompts:
        return jsonify({"error": "Missing task_id or prompts"}), 400

    with batch_vqa_state["lock"]:
        if batch_vqa_state["is_running"]:
            return jsonify({"error": "Batch processing already running"}), 400
        batch_vqa_state["is_running"] = True
        batch_vqa_state["cancel_requested"] = False
        batch_vqa_state["processed"] = 0
        batch_vqa_state["current_image"] = ""

    # Get task config
    tasks = load_annotation_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        with batch_vqa_state["lock"]:
            batch_vqa_state["is_running"] = False
        return jsonify({"error": "Task not found"}), 404

    dataset_version = task.get("dataset_version")
    if not dataset_version:
        with batch_vqa_state["lock"]:
            batch_vqa_state["is_running"] = False
        return jsonify({"error": "Task missing dataset_version"}), 400

    models = task.get("models", None)

    # Build list of all images that need processing
    images_to_process = []
    for prompt_data in prompts:
        prompt_id = prompt_data.get("prompt_id")
        if not prompt_id:
            continue

        images = get_images_for_prompt(prompt_id, dataset_version, models)
        for img in images:
            # Check if chatgpt annotations already exist
            chatgpt_annotations = load_annotations(task_id, img["model"], prompt_id, f"llm_judge:{MODEL_ID}")
            if chatgpt_annotations is None:
                images_to_process.append({"prompt_data": prompt_data, "image": img})

    with batch_vqa_state["lock"]:
        batch_vqa_state["total"] = len(images_to_process)

    return jsonify(
        {
            "success": True,
            "total": len(images_to_process),
            "message": f"Found {len(images_to_process)} images to process",
        }
    )


@app.route("/api/batch_vqa/process_next", methods=["POST"])
def process_next_batch_vqa():
    """Process the next image in the batch queue."""
    global batch_vqa_state

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    task_id = data.get("task_id")
    prompt_data = data.get("prompt_data")
    image_info = data.get("image")

    if not all([task_id, prompt_data, image_info]):
        return jsonify({"error": "Missing required fields"}), 400

    dataset_version = data.get("dataset_version")
    preprocess_with_al = data.get("preprocess_with_al", True)

    if not dataset_version:
        return jsonify({"error": "Missing dataset_version"}), 400

    with batch_vqa_state["lock"]:
        if batch_vqa_state["cancel_requested"]:
            batch_vqa_state["is_running"] = False
            return jsonify({"cancelled": True})
        batch_vqa_state["current_image"] = f"{image_info['model']}/{image_info['prompt_id']}"

    # Generate annotations
    # Use the image path from the image info if available (it was resolved nicely in get_images_for_prompt)
    if "image_path" in image_info:
        image_path = GENERATED_IMAGES_DIR / image_info["image_path"]
    else:
        # Fallback to search
        image_path = get_best_image_path(dataset_version, image_info['prompt_id'], image_info['model'])

    if not image_path or not image_path.exists():
        with batch_vqa_state["lock"]:
            batch_vqa_state["processed"] += 1
        return jsonify({"error": "Image not found", "skipped": True})

    try:
        annotations = get_chatgpt_annotations(str(image_path), prompt_data, preprocess_with_al)
        save_annotations(
            task_id,
            image_info["model"],
            image_info["prompt_id"],
            f"llm_judge:{MODEL_ID}",
            annotations,
        )

        with batch_vqa_state["lock"]:
            batch_vqa_state["processed"] += 1

        return jsonify({"success": True, "annotations": annotations})
    except Exception as e:
        with batch_vqa_state["lock"]:
            batch_vqa_state["processed"] += 1
        return jsonify({"error": str(e)})


@app.route("/api/batch_vqa/status")
def get_batch_vqa_status():
    """Get current batch VQA processing status."""
    global batch_vqa_state

    with batch_vqa_state["lock"]:
        return jsonify(
            {
                "is_running": batch_vqa_state["is_running"],
                "total": batch_vqa_state["total"],
                "processed": batch_vqa_state["processed"],
                "current_image": batch_vqa_state["current_image"],
                "cancel_requested": batch_vqa_state["cancel_requested"],
            }
        )


@app.route("/api/batch_vqa/cancel", methods=["POST"])
def cancel_batch_vqa():
    """Cancel batch VQA processing."""
    global batch_vqa_state

    with batch_vqa_state["lock"]:
        batch_vqa_state["cancel_requested"] = True

    return jsonify({"success": True, "message": "Cancel requested"})


@app.route("/api/batch_vqa/reset", methods=["POST"])
def reset_batch_vqa():
    """Reset batch VQA state."""
    global batch_vqa_state

    with batch_vqa_state["lock"]:
        batch_vqa_state["is_running"] = False
        batch_vqa_state["cancel_requested"] = False
        batch_vqa_state["total"] = 0
        batch_vqa_state["processed"] = 0
        batch_vqa_state["current_image"] = ""

    return jsonify({"success": True})


@app.route("/assets/<path:filepath>")
def serve_assets(filepath):
    """Serve files from assets directory."""
    return send_from_directory(ASSETS_DIR, filepath)


@app.route("/generated_images/<path:filepath>")
def serve_generated_images(filepath):
    """Serve files from generated_images directory."""
    return send_from_directory(GENERATED_IMAGES_DIR, filepath)


@app.route("/annotations/<path:filepath>")
def serve_annotations(filepath):
    """Serve files from annotations directory (artifact masks, etc.)."""
    return send_from_directory(ANNOTATIONS_BASE_DIR, filepath)


@app.route("/anchors/<path:filepath>")
def serve_anchors(filepath):
    """Serve files from anchors directory (reference images for questions)."""
    return send_from_directory(ANCHORS_DIR, filepath)




def check_image_completion(task_config, prompt_data, annots, has_mask=False, has_points=False, anchor_config=None, prompt_id=None):
    """Check if an image is fully annotated based on task configuration."""
    annotations_config = task_config.get("annotations", []) # List of strings
    
    require_artifacts = "artifacts_brush" in annotations_config
    require_artifact_points = "artifacts_point" in annotations_config
    require_likert_global = "artifacts_likert" in annotations_config or any(a.startswith("likert_question@") for a in annotations_config)

    # specialized modes
    mode_text_likert_only = "text_likert" in annotations_config
    mode_text_per_word = "text_per_word" in annotations_config
    mode_text_bqa = "text_bqa" in annotations_config
    
    # helper to know if any text rendering checks are needed
    mode_any_text = mode_text_likert_only or mode_text_per_word or mode_text_bqa
    
    # Anchor modes
    mode_bqa_anchor = "bqa_anchor" in annotations_config
    mode_bqa_no_anchor = "bqa_no_anchor" in annotations_config
    mode_likert_anchor = "likert_anchor" in annotations_config or not annotations_config
    mode_any_anchor = mode_bqa_anchor or mode_bqa_no_anchor or mode_likert_anchor
    
    # Standard VQA check
    mode_vqa = "all_bqa" in annotations_config or mode_any_anchor
    if not annotations_config:
        mode_vqa = True 

    # Check Artifacts Brush
    if require_artifacts:
        if not has_mask: 
            return False

    # Check Artifacts Points
    if require_artifact_points:
        if not has_points: 
            return False

    # Check Global Likert
    if require_likert_global:
        if not annots or annots.get("_aesthetics_rating", 0) <= 0:
            return False

    # Check individual questions (Text Likert / Per Word / VQA)
    if mode_any_text or mode_vqa:
        if not annots: # Missing annotations file
            return False
            
        prompt_annotations = prompt_data.get("annotations", [])
        
        # Helper to check if annotation has anchor
        def check_has_anchor_internal(p_id, u_id):
            if not anchor_config: return False
            annotations = anchor_config.get("annotations", [])
            entry = next((item for item in annotations if item.get("prompt_id") == p_id and item.get("uid") == u_id), None)
            return entry and entry.get("needs_anchor") is True

        for annot_item in prompt_annotations:
            uid = annot_item.get("uid")
            if not uid: continue
            
            skill = annot_item.get("skill")
            subskill = annot_item.get("subskill")
            is_text_rendering = skill == "text_rendering" and (subskill == "rendering_accuracy" or subskill == "numerical")
            
            # Determine if this specific question is required
            required = False
            
            if is_text_rendering:
                if mode_text_likert_only or mode_text_per_word or mode_text_bqa or mode_likert_anchor:
                    required = True
            elif mode_vqa:
                # 1. If strict "all_bqa" is active, it's required
                if "all_bqa" in annotations_config:
                    required = True
                # 2. If legacy default, it's required (empty config)
                elif not annotations_config:
                    required = True
                # 3. If anchor modes, check anchor status
                elif mode_any_anchor:
                    has_anchor = check_has_anchor_internal(prompt_id, uid)
                    # Both bqa_anchor and bqa_no_anchor only show questions that HAVE an anchor
                    if has_anchor:
                        required = True
                        
            if required:
                # Check if answered
                
                # Case 1: Text Likert Mode (Validation: _text_likert exists and is valid integer >= 0)
                if is_text_rendering and mode_text_likert_only:
                    if "_text_likert" in annots:
                        val = annots["_text_likert"]
                        try:
                            # Allow 0 (No Text) as valid answer
                            if int(val) < 0:
                                return False
                        except (ValueError, TypeError):
                            return False
                    else:
                        # Missing _text_likert key
                        return False
                
                # Case 2: Text Per Word or Text BQA or Standard VQA (Validation: uid exists in annots)
                else:
                    if uid not in annots:
                        return False

    return True


@app.route("/api/task/<task_id>/completion_status")
def get_task_completion_status(task_id: str):
    """Get completion status for a task (how many annotations are completed)."""
    annotator = request.args.get("annotator")
    if not annotator:
        return jsonify({"error": "Missing annotator parameter"}), 400

    # Find the task
    tasks = load_annotation_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    dataset_version = task.get("dataset_version")
    prompts_file = task.get("prompts_file")
    models = task.get("models", [])
    annotations_config = task.get("annotations", [])

    # Determine requirements
    require_artifacts = "artifacts_brush" in annotations_config
    require_artifact_points = "artifacts_point" in annotations_config
    require_likert_global = "artifacts_likert" in annotations_config or any(a.startswith("likert_question@") for a in annotations_config)
    
    # specialized modes
    mode_text_likert_only = "text_likert" in annotations_config
    mode_text_per_word = "text_per_word" in annotations_config
    mode_text_bqa = "text_bqa" in annotations_config
    
    # helper to know if any text rendering checks are needed
    mode_any_text = mode_text_likert_only or mode_text_per_word or mode_text_bqa
    
    # Anchor modes
    mode_bqa_anchor = "bqa_anchor" in annotations_config
    mode_bqa_no_anchor = "bqa_no_anchor" in annotations_config
    mode_likert_anchor = "likert_anchor" in annotations_config or not annotations_config
    mode_any_anchor = mode_bqa_anchor or mode_bqa_no_anchor or mode_likert_anchor
    
    # Standard VQA check
    # If explicit "all_bqa" is present, OR anchor-specific modes (bqa_anchor, bqa_no_anchor, likert_anchor)
    mode_vqa = "all_bqa" in annotations_config or mode_any_anchor
    
    # detailed check for standard vqa if no other modes defined (backward compatibility)
    if not annotations_config:
        mode_vqa = True

    if not dataset_version or not prompts_file:
        return jsonify({"error": "Task missing dataset_version or prompts_file"}), 400
        
    # Load anchor config if needed
    anchor_config = {}
    if mode_any_anchor:
        try:
            config_path = ANCHORS_DIR / dataset_version / "needs_anchor.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    anchor_config = json.load(f)
        except Exception:
            pass # Ignore errors

    # Helper to check if annotation has anchor
    def check_has_anchor(prompt_id, uid):
        if not anchor_config: return False
        annotations = anchor_config.get("annotations", [])
        entry = next((item for item in annotations if item.get("prompt_id") == prompt_id and item.get("uid") == uid), None)
        return entry and entry.get("needs_anchor") is True

    # Load prompts for the task
    prompts = load_prompts_from_file(dataset_version, prompts_file)
    
    total_images = 0
    annotated_images = 0

    # Check each prompt and model combination
    for prompt in prompts:
        prompt_id = prompt.get("prompt_id")
        if not prompt_id:
            continue

        for model in models:
            # Check if image file exists
            image_path = get_best_image_path(dataset_version, prompt_id, model)
            if not image_path or not image_path.exists():
                continue
            
            total_images += 1
            
            # Load annotations once
            annots = load_annotations(task_id, model, prompt_id, annotator)

            # Check Artifacts Brush if required
            if require_artifacts:
                mask_path = ANNOTATIONS_BASE_DIR / task_id / "artifact_mask" / model / f"{prompt_id}_{annotator}.png"
                if not mask_path.exists():
                    continue

            # Check Artifacts Points if required
            if require_artifact_points:
                points_path = ANNOTATIONS_BASE_DIR / task_id / "artifact_points" / model / f"{prompt_id}.json"
                if not points_path.exists():
                    continue
                try:
                    with open(points_path, "r", encoding="utf-8") as f:
                        all_pts = json.load(f)
                    if annotator not in all_pts:
                        continue
                except (json.JSONDecodeError, IOError):
                    continue

            # Check Global Likert if required
            if require_likert_global:
                if not annots or annots.get("_aesthetics_rating", 0) <= 0:
                    continue

            # Check individual questions (Text Likert / Per Word / VQA)
            if mode_any_text or mode_vqa:
                if not annots: # Missing annotations file
                    continue
                
                prompt_annotations = prompt.get("annotations", [])
                all_questions_answered = True
                
                for annot_item in prompt_annotations:
                    uid = annot_item.get("uid")
                    if not uid: continue
                    
                    skill = annot_item.get("skill")
                    subskill = annot_item.get("subskill")
                    is_text_rendering = skill == "text_rendering" and (subskill == "rendering_accuracy" or subskill == "numerical")
                    
                    # Determine if this specific question is required
                    required = False
                    
                    if is_text_rendering:
                        if mode_text_likert_only or mode_text_per_word or mode_text_bqa:
                            required = True
                    elif mode_vqa:
                        # 1. If strict "all_bqa" is active, it's required
                        if "all_bqa" in annotations_config:
                            required = True
                        # 2. If legacy default, it's required
                        elif not annotations_config:
                            required = True
                        # 3. If anchor modes, check anchor status
                        elif mode_any_anchor:
                            has_anchor = check_has_anchor(prompt_id, uid)
                            # Both bqa_anchor and bqa_no_anchor only show questions that HAVE an anchor
                            if has_anchor:
                                required = True
                        
                    if required:
                        # Check if answered
                        
                        # Case 1: Text Likert Mode (Validation: _text_likert exists and is valid integer >= 0)
                        if is_text_rendering and mode_text_likert_only:
                            if "_text_likert" in annots:
                                val = annots["_text_likert"]
                                try:
                                    # Allow 0 (No Text) as valid answer
                                    if int(val) < 0:
                                        all_questions_answered = False
                                        break
                                except (ValueError, TypeError):
                                    all_questions_answered = False
                                    break
                            else:
                                # Missing _text_likert key
                                all_questions_answered = False
                                break
                        
                        # Case 2: Text Per Word or Text BQA or Standard VQA (Validation: uid exists in annots)
                        else:
                            # Standard VQA check (and Per Word check since per-word saves to uid key)
                            if uid not in annots:
                                all_questions_answered = False
                                break
                
                if not all_questions_answered:
                    continue
            
            annotated_images += 1

    return jsonify({
        "task_id": task_id,
        "annotator": annotator,
        "total_images": total_images,
        "annotated_images": annotated_images,
        "is_complete": total_images > 0 and annotated_images == total_images
    })


@app.route("/api/task/<task_id>/anchor_config")
def get_anchor_config(task_id):
    """Get the anchor configuration (needs_anchor.json) for a task."""
    # Find the task
    tasks = load_annotation_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    # Build path to needs_anchor.json
    dataset_version = task["dataset_version"]
    config_path = ANCHORS_DIR / dataset_version / "needs_anchor.json"

    # Check if config exists
    if not config_path.exists():
        return jsonify({"error": "Anchor configuration not found"}), 404

    # Load and return the config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": f"Error loading anchor config: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5002"))
    app.run(debug=True, host="0.0.0.0", port=port)
