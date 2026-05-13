"""
Sample prompts to cover all skills and subskills.

Strategy: Set Cover with Balanced Repetition
1. Phase 1: Ensure every skill/subskill pair is covered at least once (prioritize rare pairs)
2. Phase 2: Balance coverage - iteratively add prompts that help the least-covered pairs
3. Target: Each skill (not subskill) should have at least 10% of total samples (if possible)
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from utils.skills import SKILL_TAXONOMY


def get_all_skill_subskill_pairs() -> List[Tuple[str, str]]:
    """Get all valid (skill, subskill) pairs from taxonomy."""
    pairs = []
    for skill, subskills in SKILL_TAXONOMY.items():
        if subskills is None:
            pairs.append((skill, ""))
        else:
            for subskill in subskills:
                pairs.append((skill, subskill))
    return pairs


def load_prompts(input_path: Path) -> List[dict]:
    """Load prompts from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_skill_to_prompts_map(prompts: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    """Build mapping from (skill, subskill) to list of prompts containing that skill."""
    skill_to_prompts = defaultdict(list)

    for prompt in prompts:
        # Track which skill/subskill pairs this prompt covers
        covered_pairs = set()
        for ann in prompt.get("annotations", []):
            skill = ann.get("skill", "")
            subskill = ann.get("subskill", "")
            pair = (skill, subskill)
            covered_pairs.add(pair)

        # Add prompt to each skill/subskill it covers
        for pair in covered_pairs:
            skill_to_prompts[pair].append(prompt)

    return skill_to_prompts


def get_prompt_pairs(prompt: dict) -> Set[Tuple[str, str]]:
    """Get all (skill, subskill) pairs covered by a prompt."""
    pairs = set()
    for ann in prompt.get("annotations", []):
        pairs.add((ann.get("skill", ""), ann.get("subskill", "")))
    return pairs


def get_prompt_skills(prompt: dict) -> Set[str]:
    """Get all skills (without subskill) covered by a prompt."""
    skills = set()
    for ann in prompt.get("annotations", []):
        skills.add(ann.get("skill", ""))
    return skills


def get_prompt_difficulty(prompt: dict) -> str:
    """Classify prompt difficulty based on number of unique skills."""
    num_skills = len(get_prompt_skills(prompt))
    if num_skills <= 2:
        return "easy"
    elif num_skills <= 4:
        return "medium"
    else:
        return "hard"


def sample_prompts(
    prompts: List[dict],
    target_count: int = 50,
    min_skill_ratio: float = 0.10,
    difficulty_targets: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Sample prompts using Set Cover with Balanced Repetition and Difficulty Balancing.

    Strategy:
    1. Phase 1: Cover all skill/subskill pairs at least once (prioritize rare pairs)
    2. Phase 2: Balance by skill - ensure each skill has at least min_skill_ratio of samples
    3. Phase 3: Balance by difficulty - target specified difficulty distribution
    4. Phase 4: Fill remaining slots helping the least-covered pairs catch up

    Args:
        prompts: List of tagged prompts
        target_count: Target number of prompts to sample
        min_skill_ratio: Minimum ratio of samples per skill (e.g., 0.10 = 10%)
        difficulty_targets: Target ratios for each difficulty level (default: 40% easy, 40% medium, 20% hard)
        verbose: Print progress information
        seed: Random seed for deterministic sampling (default: None = deterministic by prompt_id)
    """
    # Initialize random seed if provided
    if seed is not None:
        random.seed(seed)

    # Default difficulty targets
    if difficulty_targets is None:
        difficulty_targets = {"easy": 0.40, "medium": 0.40, "hard": 0.20}
    all_pairs = get_all_skill_subskill_pairs()
    all_skills = sorted(SKILL_TAXONOMY.keys())
    skill_to_prompts = build_skill_to_prompts_map(prompts)

    # Build skill (not subskill) to prompts map
    skill_only_to_prompts: Dict[str, List[dict]] = defaultdict(list)
    for prompt in prompts:
        for skill in get_prompt_skills(prompt):
            skill_only_to_prompts[skill].append(prompt)

    # Build difficulty to prompts map
    difficulty_to_prompts: Dict[str, List[dict]] = defaultdict(list)
    for prompt in prompts:
        diff = get_prompt_difficulty(prompt)
        difficulty_to_prompts[diff].append(prompt)

    # Track state
    covered_pairs: Set[Tuple[str, str]] = set()
    selected_prompts: List[dict] = []
    selected_prompt_ids: Set[int] = set()

    # Coverage counters
    pair_coverage_count: Dict[Tuple[str, str], int] = defaultdict(int)
    skill_coverage_count: Dict[str, int] = defaultdict(int)
    difficulty_count: Dict[str, int] = defaultdict(int)

    # Sort pairs by count (ascending) - prioritize rare pairs
    pair_counts = [(pair, len(skill_to_prompts.get(pair, []))) for pair in all_pairs]
    pair_counts.sort(key=lambda x: x[1])

    if verbose:
        print("\n=== Skill/Subskill Coverage in Dataset ===")
        for pair, count in pair_counts:
            skill, subskill = pair
            label = f"{skill}/{subskill}" if subskill else skill
            print(f"  {label}: {count} prompts")
        print()

    def add_prompt(prompt: dict, reason: str = ""):
        """Add a prompt to selection and update counters."""
        selected_prompts.append(prompt)
        selected_prompt_ids.add(prompt["prompt_id"])

        for pair in get_prompt_pairs(prompt):
            covered_pairs.add(pair)
            pair_coverage_count[pair] += 1

        for skill in get_prompt_skills(prompt):
            skill_coverage_count[skill] += 1

        difficulty_count[get_prompt_difficulty(prompt)] += 1

        if verbose and reason:
            print(f"Selected prompt {prompt['prompt_id']} - {reason}")

    # =========================================================================
    # Phase 1: Ensure every skill/subskill pair is covered at least once
    # =========================================================================
    if verbose:
        print("=== Phase 1: Initial Coverage ===")

    for pair, count in pair_counts:
        if len(selected_prompts) >= target_count:
            break

        if pair in covered_pairs:
            continue

        if count == 0:
            skill, subskill = pair
            label = f"{skill}/{subskill}" if subskill else skill
            if verbose:
                print(f"WARNING: No prompts found for {label}")
            continue

        # Find prompt that covers the most uncovered pairs
        candidates = []
        best_new_coverage = -1

        for prompt in skill_to_prompts[pair]:
            if prompt["prompt_id"] in selected_prompt_ids:
                continue

            new_coverage = len(get_prompt_pairs(prompt) - covered_pairs)

            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                candidates = [prompt]
            elif new_coverage == best_new_coverage:
                candidates.append(prompt)

        # Break ties: use seed if provided, otherwise use lowest prompt_id
        best_prompt = None
        if candidates:
            if seed is not None:
                best_prompt = random.choice(candidates)
            else:
                best_prompt = min(candidates, key=lambda p: p["prompt_id"])

        if best_prompt:
            skill, subskill = pair
            label = f"{skill}/{subskill}" if subskill else skill
            add_prompt(best_prompt, f"covers {label} (+{best_new_coverage} pairs)")

    # =========================================================================
    # Phase 2: Ensure minimum coverage per skill (10% target)
    # =========================================================================
    min_per_skill = max(1, int(target_count * min_skill_ratio))

    if verbose:
        print(f"\n=== Phase 2: Skill Balancing (target: {min_per_skill} per skill) ===")

    # Sort skills by current coverage (ascending) - help least-covered first
    while len(selected_prompts) < target_count:
        # Find skill with lowest coverage that's below minimum
        skills_below_min = [
            (skill, skill_coverage_count[skill])
            for skill in all_skills
            if skill_coverage_count[skill] < min_per_skill and len(skill_only_to_prompts.get(skill, [])) > 0
        ]

        if not skills_below_min:
            break  # All skills at or above minimum (or no prompts available)

        # Sort by coverage count, then by available prompts, then by skill name for determinism
        skills_below_min.sort(key=lambda x: (x[1], len(skill_only_to_prompts.get(x[0], [])), x[0]))
        target_skill = skills_below_min[0][0]

        # Find best prompt for this skill (one that helps other low-coverage skills too)
        candidates = []
        best_score = -1

        for prompt in skill_only_to_prompts[target_skill]:
            if prompt["prompt_id"] in selected_prompt_ids:
                continue

            # Score: prioritize prompts that help multiple underrepresented skills
            score = 0
            for skill in get_prompt_skills(prompt):
                if skill_coverage_count[skill] < min_per_skill:
                    # Higher score for skills further below minimum
                    score += min_per_skill - skill_coverage_count[skill]

            if score > best_score:
                best_score = score
                candidates = [prompt]
            elif score == best_score:
                candidates.append(prompt)

        # Break ties: use seed if provided, otherwise use lowest prompt_id
        best_prompt = None
        if candidates:
            if seed is not None:
                best_prompt = random.choice(candidates)
            else:
                best_prompt = min(candidates, key=lambda p: p["prompt_id"])

        if best_prompt:
            add_prompt(best_prompt, f"balancing {target_skill} (now {skill_coverage_count[target_skill]+1})")
        else:
            # No more prompts available for this skill, mark as exhausted
            if verbose:
                print(f"  No more prompts available for {target_skill}")
            # Remove from consideration by setting coverage to minimum
            skill_coverage_count[target_skill] = min_per_skill

    # =========================================================================
    # Phase 3: Balance by difficulty
    # =========================================================================
    difficulty_min = {diff: max(1, int(target_count * ratio)) for diff, ratio in difficulty_targets.items()}

    if verbose:
        print(f"\n=== Phase 3: Difficulty Balancing ===")
        for diff, target in difficulty_min.items():
            print(f"  Target {diff}: {target} ({difficulty_targets[diff]*100:.0f}%)")

    while len(selected_prompts) < target_count:
        # Find difficulty levels below target
        diffs_below_target = [
            (diff, difficulty_count[diff], difficulty_min[diff])
            for diff in ["easy", "medium", "hard"]
            if difficulty_count[diff] < difficulty_min[diff]
        ]

        if not diffs_below_target:
            break  # All difficulty levels at target

        # Sort by how far below target (ascending), then by difficulty order for determinism
        diffs_below_target.sort(key=lambda x: (x[1] - x[2], ["easy", "medium", "hard"].index(x[0])))
        target_diff = diffs_below_target[0][0]

        # Find best prompt of this difficulty that still helps skill coverage
        candidates = []
        best_score = -1

        for prompt in difficulty_to_prompts[target_diff]:
            if prompt["prompt_id"] in selected_prompt_ids:
                continue

            # Score: prefer prompts that help underrepresented skills
            score = 0
            for skill in get_prompt_skills(prompt):
                if skill_coverage_count[skill] < min_per_skill:
                    score += min_per_skill - skill_coverage_count[skill]
                else:
                    score += 0.1  # Small bonus for any skill coverage

            # Also consider pair coverage
            for pair in get_prompt_pairs(prompt):
                current = pair_coverage_count.get(pair, 0)
                score += 1.0 / (current + 1)

            if score > best_score:
                best_score = score
                candidates = [prompt]
            elif score == best_score:
                candidates.append(prompt)

        # Break ties: use seed if provided, otherwise use lowest prompt_id
        best_prompt = None
        if candidates:
            if seed is not None:
                best_prompt = random.choice(candidates)
            else:
                best_prompt = min(candidates, key=lambda p: p["prompt_id"])

        if best_prompt:
            add_prompt(best_prompt, f"difficulty {target_diff} (now {difficulty_count[target_diff]+1})")
        else:
            # No more prompts of this difficulty, mark as exhausted
            if verbose:
                print(f"  No more {target_diff} prompts available")
            difficulty_count[target_diff] = difficulty_min[target_diff]

    # =========================================================================
    # Phase 4: Fill remaining slots - help least-covered pairs catch up
    # =========================================================================
    if len(selected_prompts) < target_count:
        if verbose:
            print(f"\n=== Phase 4: Balanced Fill (to {target_count}) ===")

        remaining_prompts = [p for p in prompts if p["prompt_id"] not in selected_prompt_ids]

        while len(selected_prompts) < target_count and remaining_prompts:
            # Find the pair with minimum coverage
            min_coverage = min(pair_coverage_count.get(p, 0) for p in all_pairs)

            # Find prompts that help pairs at minimum coverage
            candidates = []
            best_score = -1

            for prompt in remaining_prompts:
                prompt_pairs = get_prompt_pairs(prompt)
                prompt_diff = get_prompt_difficulty(prompt)

                # Score: count how many min-coverage pairs this prompt helps
                # Plus small bonus for helping other low-coverage pairs
                score = 0
                for pair in prompt_pairs:
                    current = pair_coverage_count.get(pair, 0)
                    if current == min_coverage:
                        score += 10  # High priority for minimum coverage pairs
                    else:
                        score += 1.0 / (current + 1)  # Small bonus for other low pairs

                # Penalize if difficulty is already over target
                diff_target = difficulty_min.get(prompt_diff, 0)
                if difficulty_count[prompt_diff] >= diff_target:
                    score *= 0.5  # Reduce score for over-represented difficulties

                if score > best_score:
                    best_score = score
                    candidates = [prompt]
                elif score == best_score:
                    candidates.append(prompt)

            # Break ties: use seed if provided, otherwise use lowest prompt_id
            best_prompt = None
            if candidates:
                if seed is not None:
                    best_prompt = random.choice(candidates)
                else:
                    best_prompt = min(candidates, key=lambda p: p["prompt_id"])

            if best_prompt:
                add_prompt(best_prompt, f"fill (score: {best_score:.1f})")
                remaining_prompts.remove(best_prompt)
            else:
                break

    return selected_prompts


def print_coverage_summary(
    selected_prompts: List[dict],
    min_skill_ratio: float = 0.10,
    difficulty_targets: Optional[Dict[str, float]] = None,
):
    """Print summary of skill/subskill coverage and difficulty distribution in selected prompts."""
    if difficulty_targets is None:
        difficulty_targets = {"easy": 0.40, "medium": 0.40, "hard": 0.20}

    all_pairs = get_all_skill_subskill_pairs()
    all_skills = list(SKILL_TAXONOMY.keys())
    pair_counts = defaultdict(int)
    skill_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)

    for prompt in selected_prompts:
        prompt_pairs = set()
        prompt_skills = set()
        for ann in prompt.get("annotations", []):
            pair = (ann.get("skill", ""), ann.get("subskill", ""))
            prompt_pairs.add(pair)
            prompt_skills.add(ann.get("skill", ""))
        for pair in prompt_pairs:
            pair_counts[pair] += 1
        for skill in prompt_skills:
            skill_counts[skill] += 1
        difficulty_counts[get_prompt_difficulty(prompt)] += 1

    min_per_skill = max(1, int(len(selected_prompts) * min_skill_ratio))
    total = len(selected_prompts)

    # Difficulty distribution summary
    print("\n=== Difficulty Distribution ===")
    print(f"Total prompts selected: {total}")
    for diff in ["easy", "medium", "hard"]:
        count = difficulty_counts[diff]
        pct = (count / total * 100) if total else 0
        target_pct = difficulty_targets.get(diff, 0) * 100
        target_count = max(1, int(total * difficulty_targets.get(diff, 0)))
        status = "✓" if count >= target_count else "✗"
        print(f"  {status} {diff.capitalize():8} {count:3} ({pct:5.1f}%) - target: {target_count} ({target_pct:.0f}%)")

    # Skill-level summary
    print("\n=== Skill Coverage Summary ===")
    print(f"Target per skill: {min_per_skill} ({min_skill_ratio*100:.0f}%)")
    print()

    skills_at_target = 0
    for skill in sorted(all_skills):
        count = skill_counts.get(skill, 0)
        pct = (count / total * 100) if total else 0
        status = "✓" if count >= min_per_skill else "✗"
        print(f"  {status} {skill}: {count} prompts ({pct:.1f}%)")
        if count >= min_per_skill:
            skills_at_target += 1

    print(f"\nSkills at target: {skills_at_target}/{len(all_skills)}")

    # Subskill-level summary
    print("\n=== Subskill Coverage Summary ===")
    covered = 0
    uncovered = []
    for pair in all_pairs:
        skill, subskill = pair
        label = f"{skill}/{subskill}" if subskill else skill
        count = pair_counts.get(pair, 0)
        if count > 0:
            covered += 1
            print(f"  ✓ {label}: {count} prompts")
        else:
            uncovered.append(label)
            print(f"  ✗ {label}: 0 prompts")

    print(f"\nSubskill coverage: {covered}/{len(all_pairs)} pairs")
    if uncovered:
        print(f"Uncovered: {', '.join(uncovered)}")


def main():
    parser = argparse.ArgumentParser(description="Sample prompts to cover all skills/subskills")
    parser.add_argument(
        "--input",
        type=str,
        default="assets/generation_prompts/v8.1-gpt-5-mini/processed_prompts.json",
        help="Path to input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/generation_prompts/v8.1-gpt-5-mini/sampled_prompts.json",
        help="Path to output sampled prompts JSON file",
    )
    parser.add_argument("-c", "--count", type=int, default=50, help="Target number of prompts to sample")
    parser.add_argument(
        "--min-skill-ratio", type=float, default=0.15, help="Minimum ratio of samples per skill (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--difficulty-easy", type=float, default=0.40, help="Target ratio for easy prompts (default: 0.40 = 40%%)"
    )
    parser.add_argument(
        "--difficulty-medium", type=float, default=0.40, help="Target ratio for medium prompts (default: 0.40 = 40%%)"
    )
    parser.add_argument(
        "--difficulty-hard", type=float, default=0.20, help="Target ratio for hard prompts (default: 0.20 = 20%%)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling (default: None = use prompt_id for ties)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Build difficulty targets dict
    difficulty_targets = {
        "easy": args.difficulty_easy,
        "medium": args.difficulty_medium,
        "hard": args.difficulty_hard,
    }

    print(f"Loading prompts from {input_path}...")
    prompts = load_prompts(input_path)
    print(f"Loaded {len(prompts)} prompts")

    print(f"\nSampling {args.count} prompts (min {args.min_skill_ratio*100:.0f}% per skill)...")
    print(
        f"Difficulty targets: easy={args.difficulty_easy*100:.0f}%, medium={args.difficulty_medium*100:.0f}%, hard={args.difficulty_hard*100:.0f}%"
    )
    selected = sample_prompts(
        prompts,
        target_count=args.count,
        min_skill_ratio=args.min_skill_ratio,
        difficulty_targets=difficulty_targets,
        verbose=not args.quiet,
        seed=args.seed,
    )

    print_coverage_summary(selected, min_skill_ratio=args.min_skill_ratio, difficulty_targets=difficulty_targets)

    # Save selected prompt IDs (new format - IDs only)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract dataset version from input path
    dataset_version = input_path.parent.name
    source_file = input_path.name
    
    # Create ID-only output format
    output_data = {
        "source_file": source_file,
        "dataset_version": dataset_version,
        "prompt_ids": [p["prompt_id"] for p in selected]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(selected)} prompt IDs to {output_path}")


if __name__ == "__main__":
    main()
