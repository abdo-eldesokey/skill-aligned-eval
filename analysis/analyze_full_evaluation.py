#%%

"""
Analyze Full Evaluation: Compute average scores per model and skill.

Scoring Rules:
  - Binary QA (yes/no): yes=1, no=0, unsure=excluded (unsure rate computed separately)
  - Likert (0-5 integer): normalized to 0-1 by dividing by 5
  - Per-word text (dict): compute word accuracy via compute_text_per_word_stats, normalize 0-100 → 0-1
  - Visual artifacts brush mask: painted_pixels / total_pixels ratio (0=clean, 1=fully painted)
  - Aesthetics rating (0-5): normalized to 0-1 by dividing by 5
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import sys
import os
import itertools
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apps.annotation_stats_app import compute_text_per_word_stats
from config import BASE_DIR, ASSETS_DIR, ANNOTATIONS_DIR, GENERATION_PROMPTS_DIR

# ── Configuration ──────────────────────────────────────────────────────────────

TASK_ID = "full_evaluation"
DATASET_VERSION = "v8.1-gpt-5-mini"

ANNOTATIONS_VQA_DIR = ANNOTATIONS_DIR / TASK_ID / "vqa"
ANNOTATIONS_MASK_DIR = ANNOTATIONS_DIR / TASK_ID / "artifact_mask"
PROMPTS_DIR = GENERATION_PROMPTS_DIR / DATASET_VERSION
TASK_CONFIG_PATH = ASSETS_DIR / "annotation_tasks.json"

# ── Load prompt → question → skill mapping ─────────────────────────────────────

def load_prompt_skill_mapping():
    """
    Load processed_prompts.json and build a mapping:
      {prompt_id: {uid_str: {skill, subskill}}}
    
    Also returns the list of prompt_ids in the evaluation task,
    the list of models from the task config, and a mapping to detect
    which UIDs are text_rendering questions that use per-word annotation format.
    """
    # Load the task config to get the prompts file and models list
    with open(TASK_CONFIG_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    
    task_config = None
    for task in tasks:
        if task["id"] == TASK_ID:
            task_config = task
            break
    
    if task_config is None:
        raise ValueError(f"Task '{TASK_ID}' not found in annotation_tasks.json")
    
    # Get models from the task config
    config_models = task_config.get("models", [])
    
    prompts_file = PROMPTS_DIR / task_config["prompts_file"]
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_meta = json.load(f)
    
    eval_prompt_ids = set(prompts_meta["prompt_ids"])
    
    # Load all prompts
    source_file = PROMPTS_DIR / prompts_meta["source_file"]
    with open(source_file, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    
    # Build mapping: prompt_id -> {uid -> {skill, subskill, is_text_rendering}}
    prompt_mapping = {}
    for prompt in all_prompts:
        pid = prompt["prompt_id"]
        if pid not in eval_prompt_ids:
            continue
        
        uid_map = {}
        for ann in prompt["annotations"]:
            uid = ann["uid"]
            is_text_rendering = (ann["skill"] == "text_rendering" and 
                                 ann["subskill"] == "rendering_accuracy")
            uid_map[uid] = {
                "skill": ann["skill"],
                "subskill": ann["subskill"],
                "is_text_rendering_accuracy": is_text_rendering,
            }
        
        prompt_mapping[pid] = uid_map
    
    return prompt_mapping, eval_prompt_ids, config_models


# ── Brush mask scoring ─────────────────────────────────────────────────────────

def compute_mask_artifact_ratio(mask_path: Path) -> float:
    """
    Compute the ratio of painted (non-zero) pixels to total pixels in a mask image.
    Returns a float in [0, 1] where 0 = no artifacts, 1 = fully painted.
    """
    img = Image.open(mask_path).convert("L")  # grayscale
    pixels = np.array(img)
    total = pixels.size
    painted = np.count_nonzero(pixels)
    return painted / total if total > 0 else 0.0


# ── Score one annotator's response ─────────────────────────────────────────────

def score_annotator_responses(annotator_data: dict, uid_map: dict, img_annotations_full: dict):
    """
    Score a single annotator's responses for one image.
    
    Returns:
        scores: dict of {skill: [list of scores]}  (multiple questions per skill possible)
        unsure_counts: dict of {skill: {total: int, unsure: int}}
    """
    scores = {}
    unsure_counts = {}
    
    for uid_str, skill_info in uid_map.items():
        skill = skill_info["skill"]
        is_text_rendering_acc = skill_info["is_text_rendering_accuracy"]
        
        if uid_str not in annotator_data:
            continue
        
        value = annotator_data[uid_str]
        
        # ── Per-word text rendering question ──
        if is_text_rendering_acc and isinstance(value, dict):
            # Use compute_text_per_word_stats to get accuracy
            # Build a mini-annotations dict for just this annotator
            single_annot = {
                "annotator": {uid_str: value}
            }
            stats = compute_text_per_word_stats(single_annot)
            if "annotator" in stats:
                accuracy = stats["annotator"].get("word_accuracy", None)
                if accuracy is not None:
                    score = accuracy / 100.0  # normalize 0-100 → 0-1
                    scores.setdefault(skill, []).append(score)
            continue
        
        # ── Likert score (integer) ──
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = value / 5.0  # normalize 0-5 → 0-1
            scores.setdefault(skill, []).append(score)
            continue
        
        # ── Binary QA (yes/no/unsure) ──
        if isinstance(value, str):
            val_lower = value.strip().lower()
            
            if skill not in unsure_counts:
                unsure_counts[skill] = {"total": 0, "unsure": 0}
            unsure_counts[skill]["total"] += 1
            
            if val_lower == "unsure":
                unsure_counts[skill]["unsure"] += 1
                continue  # Exclude from score
            
            if val_lower == "yes":
                scores.setdefault(skill, []).append(1.0)
            elif val_lower == "no":
                scores.setdefault(skill, []).append(0.0)
            continue
    
    return scores, unsure_counts


# ── Main processing ────────────────────────────────────────────────────────────

def process_all_annotations():
    """
    Process all annotation files and compute average scores per model × skill.
    Only processes models listed in the task config.
    """
    prompt_mapping, eval_prompt_ids, config_models = load_prompt_skill_mapping()
    
    # Only use models from task config
    models = config_models
    print(f"Models from config: {models}")
    print(f"Evaluation prompts: {len(eval_prompt_ids)}")
    
    # Collect all per-image, per-annotator scores
    # Structure: {model: {skill: [all scores across images and annotators]}}
    all_scores = {}
    all_unsure = {}
    all_aesthetics = {}
    all_artifacts = {}
    
    # Store raw scores per annotator for convergence analysis
    # Structure: raw_scores[model][img_id][annotator][skill] = score
    raw_scores = {}
    
    for model in models:
        model_scores = {}
        model_unsure = {}
        model_aesthetics = []
        model_artifacts = []
        
        raw_scores[model] = {}
        
        vqa_dir = ANNOTATIONS_VQA_DIR / model
        mask_dir = ANNOTATIONS_MASK_DIR / model
        
        if not vqa_dir.exists():
            print(f"  WARNING: VQA directory not found for model '{model}', skipping")
            continue
        
        for json_file in sorted(vqa_dir.glob("*.json")):
            img_id = int(json_file.stem)
            
            if img_id not in eval_prompt_ids:
                continue
            
            uid_map = prompt_mapping.get(img_id)
            if uid_map is None:
                print(f"  WARNING: No prompt mapping for image {img_id}")
                continue
            
            with open(json_file, "r", encoding="utf-8") as f:
                img_annotations = json.load(f)
            
            raw_scores[model][img_id] = {}
            
            for annotator, annotator_data in img_annotations.items():
                raw_scores[model][img_id][annotator] = {}
                
                # Score VQA questions
                scores, unsure = score_annotator_responses(
                    annotator_data, uid_map, img_annotations
                )
                
                # Merge scores and store raw score
                for skill, skill_scores in scores.items():
                    model_scores.setdefault(skill, []).extend(skill_scores)
                    if skill_scores:
                        raw_scores[model][img_id][annotator][skill] = np.mean(skill_scores)
                
                # Merge unsure counts
                for skill, counts in unsure.items():
                    if skill not in model_unsure:
                        model_unsure[skill] = {"total": 0, "unsure": 0}
                    model_unsure[skill]["total"] += counts["total"]
                    model_unsure[skill]["unsure"] += counts["unsure"]
                
                # ── Aesthetics rating ──
                if "_aesthetics_rating" in annotator_data:
                    aes_val = annotator_data["_aesthetics_rating"]
                    if isinstance(aes_val, (int, float)):
                        score = aes_val / 5.0
                        model_aesthetics.append(score)
                        raw_scores[model][img_id][annotator]["aesthetic_quality"] = score
                
                # ── Visual artifacts (brush mask) ──
                mask_file = mask_dir / f"{img_id}_{annotator}.png"
                if mask_file.exists():
                    ratio = compute_mask_artifact_ratio(mask_file)
                    score = 1.0 - ratio  # Invert so higher=cleaner
                    model_artifacts.append(score)
                    raw_scores[model][img_id][annotator]["visual_artifacts"] = score
        
        all_scores[model] = model_scores
        all_unsure[model] = model_unsure
        all_aesthetics[model] = model_aesthetics
        all_artifacts[model] = model_artifacts
        
        print(f"  {model}: Processed {len(list(vqa_dir.glob('*.json')))} annotation files")
    
    return all_scores, all_unsure, all_aesthetics, all_artifacts, models, raw_scores


def build_results_table(all_scores, all_unsure, all_aesthetics, all_artifacts, models):
    """
    Build a pandas DataFrame with models as rows and skills as columns.
    Includes an 'Average' row across all models.
    """
    from utils.skills import SKILL_TAXONOMY
    
    # Ordered skill list: taxonomy skills + visual_artifacts + aesthetic_quality
    skill_columns = list(SKILL_TAXONOMY.keys()) + ["visual_artifacts", "aesthetic_quality"]
    
    rows = []
    for model in models:
        row = {"model": model}
        
        model_scores = all_scores.get(model, {})
        
        for skill in skill_columns:
            if skill == "aesthetic_quality":
                vals = all_aesthetics.get(model, [])
                row[skill] = np.mean(vals) if vals else np.nan
            elif skill == "visual_artifacts":
                vals = all_artifacts.get(model, [])
                row[skill] = 1.0 - np.mean(vals) if vals else np.nan
            else:
                vals = model_scores.get(skill, [])
                row[skill] = np.mean(vals) if vals else np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index("model")
    
    # Add average column across all skills for each model
    df["Average"] = df.mean(axis=1)
    
    return df


def build_unsure_table(all_unsure, models):
    """
    Build a table of unsure rates per model × skill.
    Includes an 'Average' row across all models.
    """
    from utils.skills import SKILL_TAXONOMY
    
    skill_columns = list(SKILL_TAXONOMY.keys())
    
    rows = []
    for model in models:
        row = {"model": model}
        model_unsure = all_unsure.get(model, {})
        
        for skill in skill_columns:
            counts = model_unsure.get(skill, {"total": 0, "unsure": 0})
            if counts["total"] > 0:
                row[skill] = counts["unsure"] / counts["total"]
            else:
                row[skill] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index("model")
    
    # Add average column across all skills for each model
    df["Average"] = df.mean(axis=1)
    
    return df


def compute_rank_convergence(raw_scores, models, max_samples=200):
    """
    Compute rank convergence: correlation between model rankings using
    subsets of k annotators vs using all annotators.
    """
    from utils.skills import SKILL_TAXONOMY
    from scipy.stats import spearmanr
    import itertools
    import random
    
    skill_columns = list(SKILL_TAXONOMY.keys()) + ["visual_artifacts", "aesthetic_quality"]
    
    # 1. Flatten into a dataframe: Model, Img, Annotator, Skill -> Score
    records = []
    for model, imgs in raw_scores.items():
        for img_id, annotators in imgs.items():
            for annotator, skills in annotators.items():
                row = {"model": model, "img_id": img_id, "annotator": annotator}
                for skill, score in skills.items():
                    row[skill] = score
                records.append(row)
                
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    all_annotators = sorted(df["annotator"].dropna().unique())
    n_annotators = len(all_annotators)
    
    if n_annotators < 2:
        return pd.DataFrame()
        
    convergence_results = []
    
    # Helper to rank models for a set of annotators
    def get_ranks_for_annotators(ann_subset):
        sub_df = df[df["annotator"].isin(ann_subset)]
        # Average across images and annotators for each model and skill
        model_skill_means = sub_df.groupby("model")[skill_columns].mean()
        # Rank models per skill (descending so higher score = rank 1)
        ranks = model_skill_means.rank(ascending=False, method='min')
        
        # Add Overall Average rank
        model_overall_means = model_skill_means.mean(axis=1)
        ranks['Average'] = model_overall_means.rank(ascending=False, method='min')
        
        return ranks
        
    # Full ground truth ranking
    gt_ranks = get_ranks_for_annotators(all_annotators)
    columns_to_track = skill_columns + ['Average']
    
    # Check k=2 to n_annotators (can't do correlation with 1 annotator easily if variance is 0, but we can try 1)
    for k in range(1, n_annotators + 1):
        combos = list(itertools.combinations(all_annotators, k))
        if len(combos) > max_samples:
            combos = random.sample(combos, max_samples)
            
        corrs_per_col = {c: [] for c in columns_to_track}
        
        for combo in combos:
            k_ranks = get_ranks_for_annotators(combo)
            for col in columns_to_track:
                if col in k_ranks.columns and col in gt_ranks.columns:
                    # Drop NaNs just in case
                    k_col = k_ranks[col].dropna()
                    gt_col = gt_ranks[col].dropna()
                    
                    # Align indices
                    common = k_col.index.intersection(gt_col.index)
                    if len(common) > 1:
                        # compute spearman correlation
                        corr, _ = spearmanr(k_col[common], gt_col[common])
                        # Handle constant array returning NaN
                        if not np.isnan(corr):
                            corrs_per_col[col].append(corr)
                            
        # Average correlation per skill/average for this k
        row = {"Number of Annotators": k}
        for col, corrs in corrs_per_col.items():
            row[col] = np.mean(corrs) if corrs else np.nan
        convergence_results.append(row)
        
    res_df = pd.DataFrame(convergence_results)
    res_df = res_df.set_index("Number of Annotators")
    return res_df


def plot_convergence(df_convergence):
    """
    Generate and save a convergence plot showing the Spearman correlation 
    of model rankings vs the reference full-annotator ranking.
    Average correlation as a thick line, and individual skills as thin lines.
    """
    if df_convergence.empty:
        print("No convergence data to plot.")
        return
        
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n" + "=" * 60)
    print("GENERATING SPEARMAN RANK CONVERGENCE PLOT")
    print("=" * 60)
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
        
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Get valid skills (ones that have non-null data)
    valid_skills = [c for c in df_convergence.columns if not df_convergence[c].isna().all().all() and c != 'Average']
    x = df_convergence.index
    
    # Build a rich visual style palette:
    # 20 distinct colors from 'tab20' paired with 15 highly distinct markers.
    # We use only solid lines because dashed/dotted lines overlap poorly with 19 plots.
    cmap = plt.get_cmap('tab20')
    _colors = [cmap(i) for i in range(20)]
    _markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'h', '<', '>', 'd', 'P', 'H', '8']
    
    def _style(i):
        return (
            _colors[i % 20],
            '-', # Explicitly use solid lines for cleaner plotting
            _markers[i % len(_markers)],
        )
    
    # Plot individual skills
    for i, skill in enumerate(valid_skills):
        y = df_convergence[skill]
        if not y.isna().all():
            color, lstyle, mstyle = _style(i)
            # Lines slightly thinner, alpha slightly higher, larger distinct markers
            ax.plot(x, y, linewidth=1.5, alpha=0.8,
                    color=color, linestyle=lstyle,
                    marker=mstyle, markersize=6, label=skill)
                
    # Plot average (thick black line, always on top)
    if 'Average' in df_convergence.columns:
        y_avg = df_convergence['Average']
        if not y_avg.isna().all():
            ax.plot(x, y_avg, linewidth=4.5, color='black',
                    label="Average across skills", marker='D', markersize=9, zorder=10)
            
    ax.set_xlabel('Number of Annotators Subset ($k$)', fontsize=14)
    ax.set_ylabel('Spearman Correlation\n(vs Full Annotators)', fontsize=14)
    ax.set_title('Ranking Stability: Spearman Rank Correlation', fontsize=16, pad=15)
    
    ax.set_xticks(x)
    
    # Correlation goes from -1 to 1, but we can set lower limit based on data
    min_y = df_convergence.min().min()
    if pd.isna(min_y):
        min_y = 0.0
    ax.set_ylim(max(-1.0, float(min_y) - 0.1), 1.05)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Legend: 2-column list outside to the right; "Average" floated to the top via sort
    handles, labels = ax.get_legend_handles_labels()
    # Move "Average across skills" to the first position
    avg_idx = next((i for i, l in enumerate(labels) if l.startswith("Average")), None)
    if avg_idx is not None:
        handles = [handles[avg_idx]] + [h for j, h in enumerate(handles) if j != avg_idx]
        labels  = [labels[avg_idx]]  + [l for j, l in enumerate(labels)  if j != avg_idx]
    ax.legend(handles, labels, fontsize=9.5, loc='center left',
              bbox_to_anchor=(1.02, 0.5), title="Skills", title_fontsize=11, ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.68)
    
    out_path = BASE_DIR / "spearman_rank_convergence.eps"
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {out_path} and {out_path.with_suffix('.png')}")


def run_analysis():
    print("=" * 60)
    print("FULL EVALUATION ANALYSIS")
    print("=" * 60)
    
    all_scores, all_unsure, all_aesthetics, all_artifacts, models, raw_scores = process_all_annotations()
    
    # ── Scores table ──
    df_scores = build_results_table(all_scores, all_unsure, all_aesthetics, all_artifacts, models)
    
    print("\n" + "=" * 60)
    print("AVERAGE SCORES PER MODEL × SKILL (0-1 scale)")
    print("=" * 60)
    
    # Format for display: transpose to show skills as rows for readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    scores_str = df_scores.T.to_string()
    print("\n" + scores_str)
    
    # ── Unsure rate table ──
    df_unsure = build_unsure_table(all_unsure, models)
    
    # Only show skills that have any unsure data
    df_unsure_clean = df_unsure.dropna(axis=1, how="all")
    unsure_str = ""
    if not df_unsure_clean.empty:
        print("\n" + "=" * 60)
        print("UNSURE RATE PER MODEL × SKILL (Binary QA only)")
        print("=" * 60)
        pd.set_option('display.float_format', '{:.2%}'.format)
        unsure_str = df_unsure_clean.T.to_string()
        print("\n" + unsure_str)
    
    # ── Save to CSV ──
    out_scores_csv = BASE_DIR / "full_evaluation_scores.csv"
    # df_scores.to_csv(out_scores_csv)
    print(f"\nScores CSV saved to {out_scores_csv}")
    
    out_unsure_csv = BASE_DIR / "full_evaluation_unsure_rates.csv"
    # df_unsure.to_csv(out_unsure_csv)
    print(f"Unsure rates CSV saved to {out_unsure_csv}")
    
    # ── Rank Convergence Analysis ──
    df_convergence = compute_rank_convergence(raw_scores, models)
    if not df_convergence.empty:
        df_convergence_t = df_convergence.T
        
        print("\n" + "=" * 60)
        print("SPEARMAN RANK CONVERGENCE (Correlation vs Full Annotators)")
        print("Columns = Number of Annotators subset")
        print("=" * 60)
        pd.set_option('display.float_format', '{:.3f}'.format)
        conv_str = df_convergence_t.to_string()
        print("\n" + conv_str)
        
        out_conv_csv = BASE_DIR / "spearman_rank_convergence.csv"
        df_convergence.to_csv(out_conv_csv)
        print(f"\nSpearman rank convergence CSV saved to {out_conv_csv}")
        
        # Generate plot
        plot_convergence(df_convergence)
    else:
        conv_str = ""
    
    # # ── Save formatted text output ──
    # out_txt = BASE_DIR / "full_evaluation_results.txt"
    # with open(out_txt, "w", encoding="utf-8") as f:
    #     f.write("AVERAGE SCORES PER MODEL × SKILL (0-1 scale)\n")
    #     f.write("=" * 60 + "\n\n")
    #     f.write(scores_str + "\n")
    #     if unsure_str:
    #         f.write("\n\nUNSURE RATE PER MODEL × SKILL (Binary QA only)\n")
    #         f.write("=" * 60 + "\n\n")
    #         f.write(unsure_str + "\n")
    #     if conv_str:
    #         f.write("\n\nRANK CONVERGENCE (Average Rank vs Annotator Count)\n")
    #         f.write("=" * 60 + "\n\n")
    #         f.write(conv_str + "\n")
    # print(f"Formatted results saved to {out_txt}")
    
    return df_scores, df_unsure, df_convergence


if __name__ == "__main__":
    run_analysis()
