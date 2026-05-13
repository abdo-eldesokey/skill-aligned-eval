"""
Centralized configuration for all paths and directories used across the project.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
APPS_DIR = BASE_DIR / "apps"

# Static directories for Apps
STATIC_DIR = APPS_DIR / "static"
PROMPT_ANALYSIS_STATIC = STATIC_DIR / "prompt_analysis_app"
IMAGE_EVALUATION_STATIC = STATIC_DIR / "image_evaluation_app"

# Assets subdirectories
GENERATED_IMAGES_DIR = ASSETS_DIR / "images"

ANNOTATIONS_DIR = ASSETS_DIR / "annotations"
AI_ANSWERS_DIR = ASSETS_DIR / "ai_answers"

# Prompts directory
GENERATION_PROMPTS_DIR = ASSETS_DIR / "generation_prompts"
TAGGING_PROMPTS_DIR = ASSETS_DIR / "tagging_prompts"
ANCHORS_DIR = ASSETS_DIR / "anchors"

# Configuration files
ANNOTATORS_FILE = ASSETS_DIR / "annotators.json"
GECKO_PROMPTS_FILE = ASSETS_DIR / "gecko_prompts.csv"
