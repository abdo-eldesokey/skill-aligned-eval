"""
Script to generate anchor configuration and download anchor images.
Combines generate_anchor_config.py and download_anchor_images.py into one workflow.

If needs_anchor.json exists, skips LLM analysis and goes straight to downloading.
Otherwise, generates the config first, then downloads images.
"""

import json
import os
import sys
import requests
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GENERATION_PROMPTS_DIR, ANCHORS_DIR
from utils.llm_client import get_client

load_dotenv()

MODEL_NAME = "gpt-5-mini"
MODEL_ID = MODEL_NAME

# Image search API configuration
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
BING_SEARCH_KEY = os.getenv("BING_SEARCH_KEY", "")


class AnchorDecision(BaseModel):
    """Decision about whether a specific annotation needs an anchor image."""

    needs_anchor: bool
    reasoning: str
    search_term: str


# Skills that need anchor image evaluation
SKILLS_NEEDING_ANCHORS = {
    ("style", "artistic_style"),
    ("named_entities", ""),
    ("environment_scene", "landmark"),
}


# ============================================================================
# PART 1: GENERATE ANCHOR CONFIG
# ============================================================================


def should_check_annotation(annotation: Dict[str, Any]) -> bool:
    """Check if this annotation should be evaluated for anchor images."""
    skill = annotation.get("skill", "")
    subskill = annotation.get("subskill", "")

    if (skill, subskill) in SKILLS_NEEDING_ANCHORS:
        return True

    if skill == "named_entities":
        return True

    return False


def analyze_annotation_with_chatgpt(prompt_text: str, phrase: str, question: str) -> AnchorDecision:
    """Use ChatGPT to analyze if a specific annotation needs an anchor image."""
    analysis_prompt = f"""You are analyzing a VQA annotation for image generation evaluation targeting a GLOBAL audience.

Given:
- Image generation prompt (for context only): "{prompt_text}"
- **PHRASE being evaluated**: "{phrase}"
- **Question about this phrase**: "{question}"

TASK: Decide if a reference anchor image for "{phrase}" would help annotators answer the question.

⚠️ IMPORTANT: Base your decision ONLY on the PHRASE "{phrase}" and the QUESTION about it.
The image generation prompt is provided ONLY for context - do NOT let other content in the prompt influence your decision about "{phrase}".

## REQUIRES ANCHOR (set needs_anchor=true) - only if "{phrase}" itself is:
1. **A Named Artist or Art Style**: e.g., "El Greco", "Monet", "Baroque", "Art Nouveau"
2. **A Landmark or Specific Place**: e.g., "Niagara Falls", "Eiffel Tower"
3. **A Named Person/Character**: e.g., "Albert Einstein", "The Simpsons"
4. **A Specific Product/Vehicle**: e.g., "1966 Ford Mustang", "iPhone 13 Pro"
5. **A Named Artwork**: e.g., "Mona Lisa", "Starry Night"

## NO ANCHOR NEEDED (set needs_anchor=false) - if "{phrase}" is:
1. Generic/universal: "a red car", "a person", "a building"
2. Technical attributes: "3D", "4K", "ultra detailed", "cinematic", "hyperrealistic"
3. Broad subjective terms: "beautiful", "high fashion", "cartoon" (without specific character)

## CRITICAL RULES:
- Evaluate ONLY "{phrase}" - ignore other named entities in the prompt that are NOT part of this phrase
- "hyperrealistic", "cinematic", "ultra detailed" → ALWAYS needs_anchor=false (these are generic style terms)
- Named artists (El Greco, Monet, etc.) in the phrase → ALWAYS needs_anchor=true
- ALWAYS provide a search_term if needs_anchor=true

## SEARCH TERM GUIDELINES (only when needs_anchor=true):
- For artists: "[artist name] painting" (e.g., "el greco painting")
- For art styles: "[style name] art" (e.g., "baroque art")
- For landmarks: Use the prompt context to determine if the scene is OUTDOOR or INDOOR:
  * If the prompt suggests an outdoor scene (street, nature, aerial view, etc.) → "[landmark name] exterior" (e.g., "royal albert hall exterior")
  * If the prompt suggests an indoor scene (concert, interior, inside, etc.) → "[landmark name] interior" (e.g., "royal albert hall interior")
  * If unclear → just use the landmark name (e.g., "niagara falls")
- For people: direct name (e.g., "albert einstein")
- For vehicles: "[make model year]" (e.g., "1966 ford mustang")

Analyze the phrase "{phrase}" and decide."""

    try:
        response = get_client(MODEL_NAME).chat.completions.create(
            model=MODEL_ID,
            response_model=AnchorDecision,
            messages=[
                {"role": "system", "content": "You are an expert in visual annotation. Be decisive: if in doubt about a named entity or specific style, set needs_anchor=true. Your reasoning must be consistent with your needs_anchor value."},
                {"role": "user", "content": analysis_prompt},
            ],
        )
        return response
    except Exception as e:
        print(f"    Error analyzing: {e}")
        return AnchorDecision(needs_anchor=False, reasoning=f"Error during analysis: {e}", search_term="")


def save_anchor_config(dataset_version: str, annotations_config: Dict[str, Any], show_summary: bool = True):
    """Save the anchor configuration to needs_anchor.json."""
    output_dir = ANCHORS_DIR / dataset_version
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "needs_anchor.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotations_config, f, indent=2, ensure_ascii=False)

    if show_summary:
        print("\n" + "=" * 80)
        print(f"Configuration saved to: {output_file}")
        print("=" * 80)

        needs_anchor = sum(1 for ann in annotations_config["annotations"] if ann["needs_anchor"])
        total = len(annotations_config["annotations"])
        print(f"\nSummary:")
        print(f"  Total annotations analyzed: {total}")
        print(f"  Need anchor images: {needs_anchor}")
        print(f"  Don't need anchor images: {total - needs_anchor}")


def generate_anchor_config(dataset_version: str, prompts_file: str) -> bool:
    """Generate anchor configuration using ChatGPT. 
    
    If needs_anchor.json exists, only processes NEW annotations not already in the config
    and appends them to the existing file.
    
    Returns True if successful.
    """
    from utils.prompt_utils import load_collection_file
    
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING ANCHOR CONFIGURATION")
    print("=" * 80)
    print(f"Using model: {MODEL_ID}")

    prompts_path = GENERATION_PROMPTS_DIR / dataset_version / prompts_file

    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}")
        return False

    # Load prompts using new ID-only format
    prompts = load_collection_file(prompts_path, dataset_version)

    # Check if config already exists and load it
    config_path = ANCHORS_DIR / dataset_version / "needs_anchor.json"
    existing_annotations = {}
    
    if config_path.exists():
        print(f"\n✓ Found existing configuration: {config_path}")
        print("  Will only process NEW annotations not already in config...")
        with open(config_path, "r", encoding="utf-8") as f:
            existing_config = json.load(f)
            # Build lookup of existing annotations by (prompt_id, uid)
            for ann in existing_config.get("annotations", []):
                key = (ann["prompt_id"], ann["uid"])
                existing_annotations[key] = ann
        
        # Start with existing config structure
        annotations_config = existing_config
    else:
        # Create new config structure
        annotations_config = {
            "dataset_version": dataset_version,
            "generated_by": "generate_and_download_anchors.py",
            "model": MODEL_ID,
            "annotations": [],
        }

    total_checked = 0
    new_annotations = 0

    for prompt_data in prompts:
        prompt_id = prompt_data.get("prompt_id")
        prompt_text = prompt_data.get("prompt", "")
        annotations = prompt_data.get("annotations", [])

        for annotation in annotations:
            if not should_check_annotation(annotation):
                continue

            uid = annotation.get("uid")
            
            # Skip if already processed
            if (prompt_id, uid) in existing_annotations:
                continue

            total_checked += 1
            new_annotations += 1

            phrase = annotation.get("phrase", "")
            question = annotation.get("question", "")
            skill = annotation.get("skill", "")
            subskill = annotation.get("subskill", "")

            print(f"\n[{prompt_id}/{uid}] Analyzing: '{phrase}'")
            print(f"  Skill: {skill}:{subskill if subskill else 'N/A'}")
            print(f"  Question: {question}")

            decision = analyze_annotation_with_chatgpt(prompt_text, phrase, question)

            status = "✓ NEEDS ANCHOR" if decision.needs_anchor else "✗ No anchor needed"
            print(f"  {status}")
            print(f"  Reasoning: {decision.reasoning}")
            if decision.needs_anchor:
                print(f"  Search term: '{decision.search_term}'")

            annotations_config["annotations"].append(
                {
                    "prompt_id": prompt_id,
                    "uid": uid,
                    "skill": skill,
                    "subskill": subskill,
                    "phrase": phrase,
                    "needs_anchor": decision.needs_anchor,
                    "reasoning": decision.reasoning,
                    "search_term": decision.search_term,
                }
            )

            # Save after each annotation (incremental save)
            save_anchor_config(dataset_version, annotations_config, show_summary=False)

    if new_annotations == 0:
        if existing_annotations:
            print(f"\n✓ No NEW annotations to process. All {len(existing_annotations)} annotations already in config.")
        else:
            print("\nNo annotations found matching SKILLS_NEEDING_ANCHORS.")
        return True  # Return True since config exists or nothing to process

    save_anchor_config(dataset_version, annotations_config)
    print(f"\n  New annotations processed: {new_annotations}")
    print(f"  Total annotations in config: {len(annotations_config['annotations'])}")
    return True


# ============================================================================
# PART 2: DOWNLOAD ANCHOR IMAGES
# ============================================================================


def search_serper_images(query: str, num_results: int = 1) -> List[str]:
    """Search for images using Serper API (free tier: 2500 searches/month)."""
    if not SERPER_API_KEY:
        return []

    url = "https://google.serper.dev/images"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results * 3}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        image_urls = []
        for item in data.get("images", [])[: num_results * 3]:
            image_urls.append(item["imageUrl"])

        return image_urls
    except Exception as e:
        print(f"Error searching Serper for '{query}': {e}")
        return []


def search_bing_images(query: str, num_results: int = 1) -> List[str]:
    """Search for images using Bing Image Search API (Azure)."""
    if not BING_SEARCH_KEY:
        return []

    url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY}
    params = {"q": query, "count": num_results * 3, "imageType": "Photo", "safeSearch": "Moderate"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        image_urls = []
        for item in data.get("value", [])[: num_results * 3]:
            image_urls.append(item["contentUrl"])

        return image_urls
    except Exception as e:
        print(f"Error searching Bing for '{query}': {e}")
        return []


def search_google_images(query: str, num_results: int = 1) -> List[str]:
    """Search for images using Google Custom Search API."""
    if not GOOGLE_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env file")
        return []

    if not GOOGLE_CSE_ID:
        return search_images_fallback(query, num_results)

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "searchType": "image",
        "num": num_results,
        "safe": "active",
        "imgSize": "medium",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        image_urls = []
        for item in data.get("items", []):
            image_urls.append(item["link"])

        return image_urls
    except requests.exceptions.RequestException as e:
        print(f"Error searching for '{query}': {e}")
        return []


def search_images_fallback(query: str, num_results: int = 1) -> List[str]:
    """Fallback image search - tries multiple sources in order."""
    if SERPER_API_KEY:
        print("  Using Serper API...")
        results = search_serper_images(query, num_results)
        if results:
            return results

    if BING_SEARCH_KEY:
        print("  Using Bing Image Search...")
        results = search_bing_images(query, num_results)
        if results:
            return results

    print("  No API keys configured for image search")
    return []


def download_image(url: str, save_path: Path) -> bool:
    """Download an image from URL and save to path. Returns True if successful."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(
            url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"  ✓ Downloaded: {save_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download from {url}: {e}")
        return False


def download_anchor_images(dataset_version: str, prompts_file: str, dry_run: bool = False):
    """Download anchor images based on the configuration."""
    from utils.prompt_utils import load_collection_file
    
    print("\n" + "=" * 80)
    print("STEP 2: DOWNLOADING ANCHOR IMAGES")
    print("=" * 80)

    config_path = ANCHORS_DIR / dataset_version / "needs_anchor.json"

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        anchor_config = json.load(f)

    prompts_path = GENERATION_PROMPTS_DIR / dataset_version / prompts_file

    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}")
        return

    print(f"\nProcessing: {prompts_path}")
    print(f"Output directory: {ANCHORS_DIR / dataset_version}")
    print(f"Using config: {config_path}")
    print("=" * 80)

    # Load prompts using new ID-only format
    prompts = load_collection_file(prompts_path, dataset_version)

    config_lookup = {(item["prompt_id"], item["uid"]): item for item in anchor_config.get("annotations", [])}

    total_annotations = 0
    downloaded = 0
    skipped = 0
    failed = 0

    for prompt_data in prompts:
        prompt_id = prompt_data.get("prompt_id")
        annotations = prompt_data.get("annotations", [])

        for annotation in annotations:
            uid = annotation.get("uid")
            config_item = config_lookup.get((prompt_id, uid))

            if not config_item or not config_item.get("needs_anchor"):
                continue

            total_annotations += 1
            phrase = annotation.get("phrase", "")
            skill = annotation.get("skill", "")
            subskill = annotation.get("subskill", "")

            # Check if all 3 images already exist
            base_dir = ANCHORS_DIR / dataset_version / str(prompt_id)
            all_exist = all((base_dir / f"{uid}_{i}.png").exists() for i in range(1, 4))

            if all_exist:
                print(f"[{prompt_id}/{uid}] Skipping (all 3 images already exist): {phrase}")
                skipped += 1
                continue

            search_query = config_item.get("search_term", "") or phrase

            print(f"\n[{prompt_id}/{uid}] Searching for: '{search_query}'")
            print(f"  Skill: {skill}:{subskill if subskill else 'N/A'}")

            if dry_run:
                print(f"  [DRY RUN] Would download 3 images to: {base_dir / f'{uid}_*.png'}")
                continue

            image_urls = search_google_images(search_query, num_results=3)

            if not image_urls:
                print(f"  ✗ No images found for '{search_query}'")
                failed += 1
                continue

            # Download up to 3 images
            images_downloaded = 0
            for i, url in enumerate(image_urls[:3]):
                save_path = base_dir / f"{uid}_{i+1}.png"
                
                # Skip if this specific image already exists
                if save_path.exists():
                    print(f"  ⊙ Image {i+1} already exists, skipping...")
                    images_downloaded += 1
                    continue
                
                print(f"  Downloading image {i+1}/3...")
                if download_image(url, save_path):
                    images_downloaded += 1
                time.sleep(0.5)

            if images_downloaded > 0:
                downloaded += images_downloaded
                print(f"  ✓ Successfully downloaded {images_downloaded}/3 images")
            else:
                failed += 1
                print(f"  ✗ Failed to download any images")

            time.sleep(1)

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Total annotations needing anchors: {total_annotations}")
    print(f"Already existed (skipped):         {skipped}")
    print(f"Successfully downloaded:           {downloaded}")
    print(f"Failed to download:                {failed}")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate anchor configuration and download anchor images (combined workflow)"
    )
    parser.add_argument("-v", "--dataset-version", default="v8.1-gpt-5-mini", help="Dataset version (e.g., v8.1-gpt-5-mini)")
    parser.add_argument("-p", "--prompts-file", default="sampled_prompts_50.json", help="Prompts JSON file name (e.g., anchor_based_collection.json)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate ALL annotations in config (ignores existing)")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("ANCHOR IMAGE GENERATOR & DOWNLOADER")
    print("=" * 80)
    print(f"\nDataset: {args.dataset_version}")
    print(f"Prompts file: {args.prompts_file}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No images will be downloaded\n")

    config_path = ANCHORS_DIR / args.dataset_version / "needs_anchor.json"

    if args.force_regenerate and config_path.exists():
        print(f"\n⚠️  Force regenerating ALL annotations (--force-regenerate flag set)")
        print(f"  Deleting existing config: {config_path}")
        config_path.unlink()

    # Always run generate_anchor_config - it handles incremental updates
    # (only processes new annotations not already in config)
    if not generate_anchor_config(args.dataset_version, args.prompts_file):
        print("\n✗ Failed to generate configuration. Exiting.")
        return

    api_configured = GOOGLE_CSE_ID or SERPER_API_KEY or BING_SEARCH_KEY

    if not api_configured:
        print("\n⚠️  ERROR: No image search API configured!")
        print("\nPlease configure at least one of the following in your .env file:")
        print("\n1. Serper API (Recommended - Free tier: 2500 searches/month):")
        print("   - Sign up at: https://serper.dev/")
        print("   - Add to .env: SERPER_API_KEY=your_key_here")
        print("\n2. Google Custom Search API:")
        print("   - Create engine at: https://programmablesearchengine.google.com/")
        print("   - Add to .env: GOOGLE_CSE_ID=your_cse_id_here")
        print("\n3. Bing Image Search (Azure):")
        print("   - Create resource at: https://portal.azure.com/")
        print("   - Add to .env: BING_SEARCH_KEY=your_key_here\n")
        return

    if SERPER_API_KEY:
        print("\n✓ Using Serper API for image search")
    elif BING_SEARCH_KEY:
        print("\n✓ Using Bing Image Search API")
    elif GOOGLE_CSE_ID:
        print("\n✓ Using Google Custom Search API")

    download_anchor_images(args.dataset_version, args.prompts_file, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("✓ ALL DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
