import json
from pathlib import Path


def load_annotation_tasks(annotation_tasks_file: Path):
    """Load annotation tasks from the assets directory."""
    if not annotation_tasks_file.exists():
        print(f"Error: {annotation_tasks_file} not found.")
        return []
    
    with open(annotation_tasks_file, "r", encoding="utf-8") as f:
        return json.load(f)

