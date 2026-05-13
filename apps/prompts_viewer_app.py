"""
Flask backend for Text-to-Image Prompts Skill Tagging Viewer
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

# Import the skill taxonomy
from utils.skills import SKILL_TAXONOMY

# Import centralized configuration
from config import GENERATION_PROMPTS_DIR, PROMPT_ANALYSIS_STATIC

app = Flask(__name__, static_folder=str(PROMPT_ANALYSIS_STATIC), static_url_path="")
CORS(app)

# Configuration
BASE_DATA_DIR = GENERATION_PROMPTS_DIR
DEFAULT_DIR = "v8.1-gpt-5-mini"


def get_available_directories() -> List[str]:
    """Get list of available prompt directories."""
    if not BASE_DATA_DIR.exists():
        return []

    directories = []
    for item in BASE_DATA_DIR.iterdir():
        if item.is_dir():
            directories.append(item.name)

    return sorted(directories)


def get_json_files_in_directory(directory: str) -> List[str]:
    """Get list of JSON files in a directory."""
    data_dir = BASE_DATA_DIR / directory
    if not data_dir.exists():
        return []

    json_files = []
    for item in data_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".json":
            json_files.append(item.name)

    return sorted(json_files)


def load_prompt_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Load prompt data - expects new annotations format only."""
    if "annotations" not in data:
        print(f"Warning: Prompt {data.get('prompt_id', 'unknown')} missing 'annotations' field")
    return data


def load_all_prompts(directory: str = DEFAULT_DIR, json_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all prompts from a JSON file in the directory.
    
    For collection files (new ID-only format), loads prompt details from source.
    For the source file (processed_prompts.json), loads directly.
    """
    from utils.prompt_utils import load_collection_file, load_source_prompts
    
    data_dir = BASE_DATA_DIR / directory

    if not data_dir.exists():
        return []

    # Use specified JSON file or default to processed_prompts.json
    if json_file:
        prompts_file = data_dir / json_file
    else:
        prompts_file = data_dir / "processed_prompts.json"

    if not prompts_file.exists():
        print(f"Warning: {prompts_file} not found")
        return []

    try:
        # If it's the source file, load directly
        if prompts_file.name == "processed_prompts.json":
            prompts = load_source_prompts(directory, prompts_file.name)
        else:
            # It's a collection file - use the new format loader
            prompts = load_collection_file(prompts_file, directory)

        # Process each prompt
        prompts = [load_prompt_data(p) for p in prompts]
        
        # Sort by prompt_id
        prompts.sort(key=lambda x: x.get("prompt_id", 0))
        return prompts

    except (json.JSONDecodeError, IOError, FileNotFoundError) as e:
        print(f"Error loading {prompts_file}: {e}")
        return []


def get_skill_statistics(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for skills and subskills from annotations."""
    skill_counts = Counter()
    subskill_counts = Counter()
    skill_subskill_counts = Counter()

    for prompt in prompts:
        annotations = prompt.get("annotations", [])
        for ann in annotations:
            skill = ann.get("skill", "")
            subskill = ann.get("subskill", "")

            skill_counts[skill] += 1

            if subskill:
                skill_subskill_str = f"{skill}:{subskill}"
                subskill_counts[skill_subskill_str] += 1
                skill_subskill_counts[skill_subskill_str] += 1
            else:
                skill_subskill_counts[skill] += 1

    return {
        "skill_counts": dict(skill_counts),
        "subskill_counts": dict(subskill_counts),
        "skill_subskill_counts": dict(skill_subskill_counts),
    }


@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(PROMPT_ANALYSIS_STATIC, "index.html")


@app.route("/api/directories")
def get_directories():
    """Get list of available prompt directories."""
    directories = get_available_directories()
    return jsonify({"directories": directories, "default": DEFAULT_DIR})


@app.route("/api/json_files")
def get_json_files():
    """Get list of JSON files in a directory."""
    directory = request.args.get("directory", DEFAULT_DIR)
    json_files = get_json_files_in_directory(directory)
    # Set default to processed_prompts.json if it exists, otherwise first file
    default_file = (
        "processed_prompts.json" if "processed_prompts.json" in json_files else (json_files[0] if json_files else "")
    )
    return jsonify({"json_files": json_files, "default": default_file})


@app.route("/api/skills")
def get_skills():
    """Get the skill taxonomy."""
    return jsonify(SKILL_TAXONOMY)


@app.route("/api/prompts")
def get_prompts():
    """Get all prompts with optional filtering."""
    # Get directory and json_file parameters
    directory = request.args.get("directory", DEFAULT_DIR)
    json_file = request.args.get("json_file")
    prompts = load_all_prompts(directory, json_file)

    # Get filter parameters
    skill_filter = request.args.get("skill")
    subskill_filter = request.args.get("subskill")

    # Apply filters
    if skill_filter or subskill_filter:
        filtered_prompts = []
        for prompt in prompts:
            annotations = prompt.get("annotations", [])

            match_found = False
            for ann in annotations:
                skill = ann.get("skill", "")
                subskill = ann.get("subskill", "")

                if skill_filter and subskill_filter:
                    # Filter by specific skill:subskill combination
                    if skill == skill_filter and subskill == subskill_filter:
                        match_found = True
                        break
                elif skill_filter:
                    # Filter by skill (with or without subskill)
                    if skill == skill_filter:
                        match_found = True
                        break

            if match_found:
                filtered_prompts.append(prompt)

        prompts = filtered_prompts

    return jsonify(prompts)


@app.route("/api/prompts/<int:prompt_id>")
def get_prompt(prompt_id: int):
    """Get a specific prompt by ID."""
    directory = request.args.get("directory", DEFAULT_DIR)
    json_file = request.args.get("json_file")
    prompts = load_all_prompts(directory, json_file)

    for prompt in prompts:
        if prompt.get("prompt_id") == prompt_id:
            return jsonify(prompt)

    return jsonify({"error": "Prompt not found"}), 404


@app.route("/api/statistics")
def get_statistics():
    """Get statistics about skills and subskills."""
    directory = request.args.get("directory", DEFAULT_DIR)
    json_file = request.args.get("json_file")
    prompts = load_all_prompts(directory, json_file)
    stats = get_skill_statistics(prompts)

    return jsonify({"total_prompts": len(prompts), "statistics": stats})


@app.route("/api/histogram")
def get_histogram():
    """Get histogram data for skills and subskills."""
    directory = request.args.get("directory", DEFAULT_DIR)
    json_file = request.args.get("json_file")
    prompts = load_all_prompts(directory, json_file)
    stats = get_skill_statistics(prompts)

    # Get the type of histogram requested
    hist_type = request.args.get("type", "skills")  # 'skills' or 'subskills'

    if hist_type == "subskills":
        data = stats["skill_subskill_counts"]
    else:
        data = stats["skill_counts"]

    # Sort by count descending
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

    return jsonify({"labels": [item[0] for item in sorted_data], "counts": [item[1] for item in sorted_data]})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
