"""
Generate comprehensive statistics for the prompts dataset.

Statistics include:
1. Number of prompts per skill
2. Number of prompts per subskill
3. Distribution of prompt length (short, medium, long)
4. Prompt difficulty (easy, medium, hard) based on number of skills
5. Additional useful statistics
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

from utils.skills import SKILL_TAXONOMY


def load_prompts(input_path: Path) -> List[dict]:
    """Load prompts from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_prompt_skills(prompt: dict) -> Set[str]:
    """Get unique skills in a prompt."""
    skills = set()
    for ann in prompt.get("annotations", []):
        skills.add(ann.get("skill", ""))
    return skills


def get_prompt_subskills(prompt: dict) -> Set[Tuple[str, str]]:
    """Get unique (skill, subskill) pairs in a prompt."""
    pairs = set()
    for ann in prompt.get("annotations", []):
        pairs.add((ann.get("skill", ""), ann.get("subskill", "")))
    return pairs


def classify_prompt_length(prompt_text: str) -> str:
    """Classify prompt length as short, medium, or long."""
    word_count = len(prompt_text.split())
    if word_count <= 10:
        return "short"
    elif word_count <= 30:
        return "medium"
    else:
        return "long"


def classify_prompt_difficulty(num_skills: int) -> str:
    """Classify prompt difficulty based on number of unique skills."""
    if num_skills <= 2:
        return "easy"
    elif num_skills <= 4:
        return "medium"
    else:
        return "hard"


def compute_statistics(prompts: List[dict]) -> Dict[str, Any]:
    """Compute comprehensive statistics for the prompts dataset."""
    stats = {}

    # Basic counts
    stats["total_prompts"] = len(prompts)

    # Skill and subskill counters
    skill_counts = Counter()
    subskill_counts = Counter()
    skill_prompt_counts = Counter()  # Prompts containing each skill
    subskill_prompt_counts = Counter()  # Prompts containing each subskill

    # Length and difficulty
    length_distribution = Counter()
    difficulty_distribution = Counter()

    # Annotation statistics
    annotations_per_prompt = []
    skills_per_prompt = []

    # Node type distribution
    node_type_counts = Counter()

    # Dependency statistics
    prompts_with_dependencies = 0
    total_dependencies = 0

    # Prompt type distribution
    prompt_type_counts = Counter()

    # Dataset distribution
    dataset_counts = Counter()

    # Word count statistics
    word_counts = []
    char_counts = []

    for prompt in prompts:
        prompt_text = prompt.get("prompt", "")
        annotations = prompt.get("annotations", [])

        # Word and character counts
        word_count = len(prompt_text.split())
        char_count = len(prompt_text)
        word_counts.append(word_count)
        char_counts.append(char_count)

        # Length classification
        length_class = classify_prompt_length(prompt_text)
        length_distribution[length_class] += 1

        # Skills in this prompt
        prompt_skills = get_prompt_skills(prompt)
        prompt_subskills = get_prompt_subskills(prompt)

        skills_per_prompt.append(len(prompt_skills))
        annotations_per_prompt.append(len(annotations))

        # Difficulty classification
        difficulty = classify_prompt_difficulty(len(prompt_skills))
        difficulty_distribution[difficulty] += 1

        # Count skills and subskills
        for skill in prompt_skills:
            skill_prompt_counts[skill] += 1

        for skill, subskill in prompt_subskills:
            label = f"{skill}/{subskill}" if subskill else skill
            subskill_prompt_counts[label] += 1

        # Count annotations by skill/subskill
        for ann in annotations:
            skill = ann.get("skill", "")
            subskill = ann.get("subskill", "")
            node_type = ann.get("node_type", "")
            depends_on = ann.get("depends_on", [])

            skill_counts[skill] += 1
            if subskill:
                subskill_counts[f"{skill}/{subskill}"] += 1
            else:
                subskill_counts[skill] += 1

            node_type_counts[node_type] += 1

            if depends_on:
                total_dependencies += len(depends_on)

        # Check if prompt has dependencies
        has_deps = any(ann.get("depends_on", []) for ann in annotations)
        if has_deps:
            prompts_with_dependencies += 1

        # Prompt type
        prompt_type = prompt.get("prompt_type", "unknown")
        prompt_type_counts[prompt_type] += 1

        # Dataset
        dataset_id = prompt.get("dataset_id", "unknown")
        # Extract main dataset category (before colon or underscore)
        dataset_category = dataset_id.split(":")[0].split("_")[0] if dataset_id else "unknown"
        dataset_counts[dataset_category] += 1

    # Store computed statistics
    stats["skill_annotation_counts"] = dict(skill_counts.most_common())
    stats["subskill_annotation_counts"] = dict(subskill_counts.most_common())
    stats["skill_prompt_counts"] = dict(skill_prompt_counts.most_common())
    stats["subskill_prompt_counts"] = dict(subskill_prompt_counts.most_common())

    stats["length_distribution"] = dict(length_distribution)
    stats["difficulty_distribution"] = dict(difficulty_distribution)

    stats["node_type_distribution"] = dict(node_type_counts)
    stats["prompt_type_distribution"] = dict(prompt_type_counts)
    stats["dataset_distribution"] = dataset_counts.most_common(20)

    # Annotation statistics
    stats["total_annotations"] = sum(annotations_per_prompt)
    stats["avg_annotations_per_prompt"] = sum(annotations_per_prompt) / len(prompts) if prompts else 0
    stats["min_annotations"] = min(annotations_per_prompt) if annotations_per_prompt else 0
    stats["max_annotations"] = max(annotations_per_prompt) if annotations_per_prompt else 0

    # Skills per prompt
    stats["avg_skills_per_prompt"] = sum(skills_per_prompt) / len(prompts) if prompts else 0
    stats["min_skills_per_prompt"] = min(skills_per_prompt) if skills_per_prompt else 0
    stats["max_skills_per_prompt"] = max(skills_per_prompt) if skills_per_prompt else 0

    # Word count statistics
    stats["avg_word_count"] = sum(word_counts) / len(prompts) if prompts else 0
    stats["min_word_count"] = min(word_counts) if word_counts else 0
    stats["max_word_count"] = max(word_counts) if word_counts else 0

    # Character count statistics
    stats["avg_char_count"] = sum(char_counts) / len(prompts) if prompts else 0
    stats["min_char_count"] = min(char_counts) if char_counts else 0
    stats["max_char_count"] = max(char_counts) if char_counts else 0

    # Dependency statistics
    stats["prompts_with_dependencies"] = prompts_with_dependencies
    stats["prompts_with_dependencies_pct"] = (prompts_with_dependencies / len(prompts) * 100) if prompts else 0
    stats["total_dependencies"] = total_dependencies
    stats["avg_dependencies_per_prompt"] = total_dependencies / len(prompts) if prompts else 0

    # Coverage statistics
    all_skills = list(SKILL_TAXONOMY.keys())
    all_subskills = []
    for skill, subskills in SKILL_TAXONOMY.items():
        if subskills is None:
            all_subskills.append(skill)
        else:
            for subskill in subskills:
                all_subskills.append(f"{skill}/{subskill}")

    covered_skills = set(skill_prompt_counts.keys())
    covered_subskills = set(subskill_prompt_counts.keys())

    stats["taxonomy_skills_total"] = len(all_skills)
    stats["taxonomy_skills_covered"] = len(covered_skills & set(all_skills))
    stats["taxonomy_subskills_total"] = len(all_subskills)
    stats["taxonomy_subskills_covered"] = len(covered_subskills & set(all_subskills))

    # Uncovered skills/subskills
    stats["uncovered_skills"] = list(set(all_skills) - covered_skills)
    stats["uncovered_subskills"] = list(set(all_subskills) - covered_subskills)

    return stats


def print_statistics(stats: Dict[str, Any], verbose: bool = True):
    """Print statistics in a formatted way."""

    print("\n" + "=" * 70)
    print("                    PROMPT DATASET STATISTICS")
    print("=" * 70)

    # Basic info
    print(f"\n{'─' * 70}")
    print("📊 BASIC STATISTICS")
    print(f"{'─' * 70}")
    print(f"  Total prompts:              {stats['total_prompts']:,}")
    print(f"  Total annotations:          {stats['total_annotations']:,}")
    print(f"  Avg annotations/prompt:     {stats['avg_annotations_per_prompt']:.2f}")
    print(f"  Min/Max annotations:        {stats['min_annotations']} / {stats['max_annotations']}")

    # Skills per prompt
    print(f"\n{'─' * 70}")
    print("🎯 SKILLS PER PROMPT")
    print(f"{'─' * 70}")
    print(f"  Average:                    {stats['avg_skills_per_prompt']:.2f}")
    print(f"  Min/Max:                    {stats['min_skills_per_prompt']} / {stats['max_skills_per_prompt']}")

    # Prompt length
    print(f"\n{'─' * 70}")
    print("📏 PROMPT LENGTH (WORDS)")
    print(f"{'─' * 70}")
    print(f"  Average:                    {stats['avg_word_count']:.1f} words")
    print(f"  Min/Max:                    {stats['min_word_count']} / {stats['max_word_count']} words")
    print(f"  Average characters:         {stats['avg_char_count']:.1f}")

    # Length distribution
    print(f"\n{'─' * 70}")
    print("📐 LENGTH DISTRIBUTION")
    print(f"{'─' * 70}")
    total = stats["total_prompts"]
    for length, count in sorted(
        stats["length_distribution"].items(), key=lambda x: ["short", "medium", "long"].index(x[0])
    ):
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {length.capitalize():10} {count:5,} ({pct:5.1f}%) {bar}")

    # Difficulty distribution
    print(f"\n{'─' * 70}")
    print("🎮 DIFFICULTY DISTRIBUTION (based on # of skills)")
    print(f"{'─' * 70}")
    for diff, count in sorted(
        stats["difficulty_distribution"].items(), key=lambda x: ["easy", "medium", "hard"].index(x[0])
    ):
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {diff.capitalize():10} {count:5,} ({pct:5.1f}%) {bar}")

    # Node type distribution
    print(f"\n{'─' * 70}")
    print("🔗 NODE TYPE DISTRIBUTION")
    print(f"{'─' * 70}")
    total_nodes = sum(stats["node_type_distribution"].values())
    for node_type, count in sorted(stats["node_type_distribution"].items(), key=lambda x: -x[1]):
        pct = count / total_nodes * 100 if total_nodes else 0
        bar = "█" * int(pct / 2)
        print(f"  {node_type:12} {count:5,} ({pct:5.1f}%) {bar}")

    # Dependency statistics
    print(f"\n{'─' * 70}")
    print("🔀 DEPENDENCY STATISTICS")
    print(f"{'─' * 70}")
    print(
        f"  Prompts with dependencies:  {stats['prompts_with_dependencies']:,} ({stats['prompts_with_dependencies_pct']:.1f}%)"
    )
    print(f"  Total dependencies:         {stats['total_dependencies']:,}")
    print(f"  Avg dependencies/prompt:    {stats['avg_dependencies_per_prompt']:.2f}")

    # Prompt type distribution
    print(f"\n{'─' * 70}")
    print("📝 PROMPT TYPE DISTRIBUTION")
    print(f"{'─' * 70}")
    for ptype, count in sorted(stats["prompt_type_distribution"].items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {ptype:15} {count:5,} ({pct:5.1f}%) {bar}")

    # Skill coverage
    print(f"\n{'─' * 70}")
    print("✅ TAXONOMY COVERAGE")
    print(f"{'─' * 70}")
    print(f"  Skills covered:             {stats['taxonomy_skills_covered']}/{stats['taxonomy_skills_total']}")
    print(f"  Subskills covered:          {stats['taxonomy_subskills_covered']}/{stats['taxonomy_subskills_total']}")

    if stats["uncovered_skills"]:
        print(f"\n  ⚠️  Uncovered skills: {', '.join(stats['uncovered_skills'])}")
    if stats["uncovered_subskills"] and verbose:
        print(f"  ⚠️  Uncovered subskills: {', '.join(stats['uncovered_subskills'][:10])}")
        if len(stats["uncovered_subskills"]) > 10:
            print(f"      ... and {len(stats['uncovered_subskills']) - 10} more")

    # Prompts per skill
    print(f"\n{'─' * 70}")
    print("📈 PROMPTS PER SKILL (prompts containing each skill)")
    print(f"{'─' * 70}")
    max_count = max(stats["skill_prompt_counts"].values()) if stats["skill_prompt_counts"] else 1
    for skill, count in sorted(stats["skill_prompt_counts"].items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        bar_len = int(count / max_count * 30)
        bar = "█" * bar_len
        print(f"  {skill:25} {count:5,} ({pct:5.1f}%) {bar}")

    # Prompts per subskill (top 20)
    if verbose:
        print(f"\n{'─' * 70}")
        print("📊 PROMPTS PER SUBSKILL (top 25)")
        print(f"{'─' * 70}")
        items = list(stats["subskill_prompt_counts"].items())[:25]
        max_count = max(c for _, c in items) if items else 1
        for label, count in items:
            pct = count / total * 100 if total else 0
            bar_len = int(count / max_count * 25)
            bar = "█" * bar_len
            print(f"  {label:35} {count:5,} ({pct:5.1f}%) {bar}")

    # Dataset distribution (top 15)
    print(f"\n{'─' * 70}")
    print("📂 DATASET DISTRIBUTION (top 15)")
    print(f"{'─' * 70}")
    items = stats["dataset_distribution"][:15]
    max_count = max(c for _, c in items) if items else 1
    for dataset, count in items:
        pct = count / total * 100 if total else 0
        bar_len = int(count / max_count * 25)
        bar = "█" * bar_len
        print(f"  {dataset:20} {count:5,} ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate statistics for prompts dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="assets/generation_prompts/v8.1-gpt-5-mini/sampled_prompts_50.json",
        help="Path to input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save statistics as JSON (optional)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary statistics",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    print(f"Loading prompts from {input_path}...")
    prompts = load_prompts(input_path)
    print(f"Loaded {len(prompts)} prompts")

    print("Computing statistics...")
    stats = compute_statistics(prompts)

    print_statistics(stats, verbose=not args.quiet)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nStatistics saved to {output_path}")


if __name__ == "__main__":
    main()
