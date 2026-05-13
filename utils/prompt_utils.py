"""
Shared utilities for loading prompts by IDs from the source file.
All collection files store only prompt IDs, and this module provides
functions to load the full prompt details from the original source.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import GENERATION_PROMPTS_DIR


def load_source_prompts(dataset_version: str, source_file: str = "processed_prompts.json") -> List[Dict[str, Any]]:
    """Load all prompts from the source file (e.g., processed_prompts.json).
    
    Args:
        dataset_version: Dataset version directory name (e.g., "v8.1-gpt-5-mini")
        source_file: Name of the source prompts file
        
    Returns:
        List of all prompt dictionaries from the source file
    """
    source_path = GENERATION_PROMPTS_DIR / dataset_version / source_file
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source prompts file not found: {source_path}")
    
    with open(source_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_lookup(prompts: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Create a prompt_id -> prompt mapping for fast lookup.
    
    Args:
        prompts: List of prompt dictionaries
        
    Returns:
        Dictionary mapping prompt_id to full prompt data
    """
    return {p["prompt_id"]: p for p in prompts}


def load_prompts_by_ids(
    dataset_version: str,
    prompt_ids: List[int],
    source_file: str = "processed_prompts.json"
) -> List[Dict[str, Any]]:
    """Load prompt details for a list of IDs from the source file.
    
    Args:
        dataset_version: Dataset version directory name
        prompt_ids: List of prompt IDs to load
        source_file: Name of the source prompts file
        
    Returns:
        List of prompt dictionaries for the specified IDs (order preserved)
    """
    all_prompts = load_source_prompts(dataset_version, source_file)
    lookup = build_prompt_lookup(all_prompts)
    
    result = []
    for pid in prompt_ids:
        if pid in lookup:
            result.append(lookup[pid])
        else:
            print(f"Warning: prompt_id {pid} not found in source file")
    
    return result


def load_collection_file(filepath: Path, dataset_version: str) -> List[Dict[str, Any]]:
    """Load prompts from a collection file (new ID-only format).
    
    Collection files have the format:
    {
        "source_file": "processed_prompts.json",
        "dataset_version": "v8.1-gpt-5-mini",
        "prompt_ids": [123, 456, ...]
    }
    
    Args:
        filepath: Path to the collection JSON file
        dataset_version: Dataset version to use for loading source prompts
        
    Returns:
        List of full prompt dictionaries
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    source_file = data.get("source_file", "processed_prompts.json")
    prompt_ids = data.get("prompt_ids", [])
    
    return load_prompts_by_ids(dataset_version, prompt_ids, source_file)


def save_collection_file(
    filepath: Path,
    prompt_ids: List[int],
    dataset_version: str,
    source_file: str = "processed_prompts.json"
) -> None:
    """Save a collection file in the new ID-only format.
    
    Args:
        filepath: Path to save the collection JSON file
        prompt_ids: List of prompt IDs to save
        dataset_version: Dataset version string
        source_file: Name of the source prompts file
    """
    data = {
        "source_file": source_file,
        "dataset_version": dataset_version,
        "prompt_ids": prompt_ids
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
