"""
Per-Skill Correlation: LLM Predictions vs Average Human Annotations.

Computes Spearman and Pearson correlations between the paired human-mean
and LLM scores per skill, and produces a grouped bar chart of per-skill Spearman ρ.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from config import BASE_DIR, ASSETS_DIR, GENERATION_PROMPTS_DIR
from apps.annotation_stats_app import compute_text_per_word_stats
from utils.skills import SKILL_TAXONOMY

# Human annotations
HUMAN_VQA_DIR = ASSETS_DIR / "annotations" / "full_evaluation" / "vqa"
HUMAN_MASK_DIR = ASSETS_DIR / "annotations" / "full_evaluation" / "artifact_mask"

# LLM annotations
LLM_VQA_DIR = ASSETS_DIR / "annotations" / "full_evaluation_llm" / "vqa"

TASK_CONFIG_PATH = ASSETS_DIR / "annotation_tasks.json"
TASK_ID = "full_evaluation"

# Size of a blank mask in bytes (24664) represents no artifacts
BLANK_MASK_SIZE = 24664

# Ordered skill list matching the taxonomy + extras
# SKILL_ORDER = list(SKILL_TAXONOMY.keys()) + ["visual_artifacts_PAL4VST", "visual_artifacts_LEGION", "aesthetic_quality", "Average"]
SKILL_ORDER = list(SKILL_TAXONOMY.keys()) + ["visual_artifacts_PAL4VST", "aesthetic_quality", "Average"]

# List of models to completely ignore from analysis (e.g. ['gpt-5-mini', 'qwen2.5-max'])
MODELS_TO_EXCLUDE = ["hf:google/gemma-3-4b-it", "hf:google/gemma-3-12b-it", "gpt-5-mini"]


def load_prompt_skill_mapping():
    """
    Load processed_prompts.json and build a mapping:
      {prompt_id: {uid_str: {skill, subskill, is_text_rendering}}}
    """
    with open(TASK_CONFIG_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    task_config = next((t for t in tasks if t["id"] == TASK_ID), None)
    if task_config is None:
        raise ValueError(f"Task '{TASK_ID}' not found in annotation_tasks.json")

    config_models = task_config.get("models", [])
    dataset_version = task_config.get("dataset_version", "")

    prompts_file = GENERATION_PROMPTS_DIR / dataset_version / task_config["prompts_file"]
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_meta = json.load(f)

    eval_prompt_ids = set(prompts_meta["prompt_ids"])

    source_file = GENERATION_PROMPTS_DIR / dataset_version / prompts_meta["source_file"]
    with open(source_file, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    prompt_mapping = {}
    for prompt in all_prompts:
        pid = prompt["prompt_id"]
        if pid not in eval_prompt_ids:
            continue

        uid_map = {}
        for ann in prompt["annotations"]:
            uid = ann["uid"]
            is_text_rendering = ann["skill"] == "text_rendering" and ann["subskill"] == "rendering_accuracy"
            uid_map[uid] = {
                "skill": ann["skill"],
                "subskill": ann["subskill"],
                "is_text_rendering_accuracy": is_text_rendering,
            }

        prompt_mapping[pid] = uid_map

    return prompt_mapping, eval_prompt_ids, config_models


def compute_mask_artifact_ratio(mask_path: Path) -> float:
    """
    If file is exactly BLANK_MASK_SIZE, assume 0 artifacts.
    Otherwise calculate ratio of non-black pixels.
    """
    if mask_path.stat().st_size == BLANK_MASK_SIZE:
        return 0.0

    try:
        from PIL import Image

        with Image.open(mask_path) as img:
            img_array = np.array(img.convert("L"))
            painted_pixels = np.sum(img_array > 0)
            total_pixels = img_array.size
            return painted_pixels / total_pixels if total_pixels > 0 else 0.0
    except Exception as e:
        print(f"    Error reading mask {mask_path.name}: {e}")
        return 0.0


def score_annotator_responses(annotator_data, uid_map):
    """
    Score a single annotator's responses for one image.
    Returns: {skill: [scores]} (all on 0-1 scale)
    """
    scores = {}

    for uid, mapping in uid_map.items():
        skill = mapping["skill"]
        ans = annotator_data.get(uid)

        if ans is None:
            continue

        # 1. Text rendering
        if mapping["is_text_rendering_accuracy"]:
            if isinstance(ans, dict):
                single_annot = {"annotator": {uid: ans}}
                stats = compute_text_per_word_stats(single_annot)
                if "annotator" in stats:
                    accuracy = stats["annotator"].get("word_accuracy", None)
                    if accuracy is not None:
                        score = accuracy / 100.0  # Normalize to 0-1
                        scores.setdefault(skill, []).append(score)
            elif isinstance(ans, str) and ans.lower().strip() == "unsure":
                pass  # skip
            continue

        # 2. Binary QA
        if isinstance(ans, str):
            ans_lower = ans.lower().strip()
            if ans_lower == "yes":
                scores.setdefault(skill, []).append(1.0)
            elif ans_lower == "no":
                scores.setdefault(skill, []).append(0.0)
            # 'unsure' is ignored for score
            continue

        # 3. Likert (0-5)
        if isinstance(ans, (int, float)):
            scores.setdefault(skill, []).append(ans / 5.0)
            continue

    return scores


def process_annotations():
    """
    Process human and LLM annotations, pairing them up per image/uid.
    Returns dictionary mapping:
        llm_name -> { skill -> list of (human_mean, llm_score) tuples }
    """
    prompt_mapping, eval_prompt_ids, models = load_prompt_skill_mapping()

    # Structure: paired_data[llm_name][skill] = [{"human": h_score, "llm": l_score}]
    paired_data = {}

    for model in models:
        human_vqa_dir = HUMAN_VQA_DIR / model
        human_mask_dir = HUMAN_MASK_DIR / model
        llm_vqa_dir = LLM_VQA_DIR / model

        if not human_vqa_dir.exists() or not llm_vqa_dir.exists():
            print(f"Skipping model {model}: missing human or LLM annotations")
            continue

        for json_file in sorted(human_vqa_dir.glob("*.json")):
            img_id = int(json_file.stem)

            if img_id not in eval_prompt_ids:
                continue

            uid_map = prompt_mapping.get(img_id)
            if not uid_map:
                continue

            # 1. Load human data
            with open(json_file, "r", encoding="utf-8") as f:
                human_data = json.load(f)

            # 2. Load LLM data
            llm_json = llm_vqa_dir / json_file.name
            if not llm_json.exists():
                continue

            with open(llm_json, "r", encoding="utf-8") as f:
                llm_data_full = json.load(f)

            if not llm_data_full:
                continue

            # ── Score human responses
            # Aggregate across human annotators for this specific image
            img_human_scores_by_skill = {}
            img_human_artifacts = []
            img_human_aesthetics = []

            for h_annotator, h_annot_data in human_data.items():
                h_scores = score_annotator_responses(h_annot_data, uid_map)

                for skill, vals in h_scores.items():
                    img_human_scores_by_skill.setdefault(skill, []).extend(vals)

                # Artifacts
                mask_file = human_mask_dir / f"{img_id}_{h_annotator}.png"
                if mask_file.exists():
                    ratio = compute_mask_artifact_ratio(mask_file)
                    img_human_artifacts.append(1.0 - ratio)

                # Aesthetics
                if "_aesthetics_rating" in h_annot_data:
                    aes = h_annot_data["_aesthetics_rating"]
                    if isinstance(aes, (int, float)):
                        img_human_aesthetics.append(aes / 5.0)

            # ── Loop over all LLMs that evaluated this image
            for llm_name, llm_data in llm_data_full.items():
                if llm_name not in paired_data:
                    paired_data[llm_name] = {"aesthetic_quality": [], "visual_artifacts_PAL4VST": [], "visual_artifacts_LEGION": []}

                llm_scores = score_annotator_responses(llm_data, uid_map)

                # ── Pair them up
                for skill in img_human_scores_by_skill:
                    h_vals = img_human_scores_by_skill[skill]
                    l_vals = llm_scores.get(skill, [])

                    if h_vals and l_vals:
                        h_mean = np.mean(h_vals)
                        l_mean = np.mean(l_vals)
                        paired_data[llm_name].setdefault(skill, []).append({"human": h_mean, "llm": l_mean})

                # Pair artifacts from the multiple models (PAL4VST, LEGION)
                if img_human_artifacts:
                    h_mean_art = np.mean(img_human_artifacts)

                    llm_mask_dir_base = ASSETS_DIR / "annotations" / "full_evaluation_llm" / "artifact_mask"
                    if not llm_mask_dir_base.exists():
                        llm_mask_dir_base = ASSETS_DIR / "annotations" / "full_evaluation_llm" / "arifact_mask"

                    for artifact_model in ["PAL4VST", "LEGION"]:
                        llm_mask_dir = llm_mask_dir_base / artifact_model
                        llm_mask_file = llm_mask_dir / f"{img_id}_{model}.png"
                        if not llm_mask_file.exists():
                            llm_mask_file = llm_mask_dir / f"{img_id}_{model}.webp"

                        if llm_mask_file.exists():
                            llm_ratio = compute_mask_artifact_ratio(llm_mask_file)
                            llm_art_score = 1.0 - llm_ratio
                        else:
                            llm_art_score = 1.0

                        skill_key = f"visual_artifacts_{artifact_model}"
                        if skill_key not in paired_data[llm_name]:
                            paired_data[llm_name][skill_key] = []
                        paired_data[llm_name][skill_key].append({"human": h_mean_art, "llm": llm_art_score})

                # Pair aesthetics
                if img_human_aesthetics and "_aesthetics_rating" in llm_data:
                    h_mean_aes = np.mean(img_human_aesthetics)
                    llm_aes = llm_data["_aesthetics_rating"]
                    if isinstance(llm_aes, (int, float)):
                        paired_data[llm_name]["aesthetic_quality"].append({"human": h_mean_aes, "llm": llm_aes / 5.0})

    return paired_data


def compute_per_skill_correlation(paired_data_by_llm):
    """
    For each LLM and each skill, compute Spearman and Pearson correlations
    between the paired human-mean and LLM scores.

    Returns:
        records: list of dicts with keys
            {LLM, Skill, Spearman_rho, Spearman_p, Pearson_r, Pearson_p, N}
    """
    records = []

    for llm_name, skills_dict in paired_data_by_llm.items():
        # Shorten LLM name for display
        short = llm_name
        if "gpt5mini" in llm_name:
            short = "gpt-5-mini"
        elif "gpt5" in llm_name:
            short = "ChatGPT-5"
        elif "gpt4" in llm_name:
            short = "gpt-4"
        elif "hf" in llm_name:  # hf:allenai/Molmo2-8B
            short = llm_name.split(":")[-1].split("/")[-1]
        else:
            short = llm_name.split(":")[-1]

        # Filter out models marked to be ignored
        if short in MODELS_TO_EXCLUDE or llm_name in MODELS_TO_EXCLUDE:
            continue

        for skill in SKILL_ORDER:
            pairs = skills_dict.get(skill, [])
            if len(pairs) < 3:
                # Need at least 3 data points for a meaningful correlation
                records.append(
                    {
                        "LLM": short,
                        "Skill": skill,
                        "Spearman_rho": np.nan,
                        "Spearman_p": np.nan,
                        "Pearson_r": np.nan,
                        "Pearson_p": np.nan,
                        "MAE": np.nan,
                        "N": len(pairs),
                        "constant_input": False,
                    }
                )
                continue

            h = np.array([p["human"] for p in pairs])
            l = np.array([p["llm"] for p in pairs])
            mae = np.mean(np.abs(h - l))

            h_const = np.all(h == h[0])
            l_const = np.all(l == l[0])

            # Handle constant-input cases where correlation is undefined
            if h_const or l_const:
                if h_const and l_const and np.isclose(h[0], l[0]):
                    # Both constant and equal → perfect agreement
                    sp_rho, sp_p = 1.0, 0.0
                    pe_r, pe_p = 1.0, 0.0
                else:
                    # One side constant → no discriminative power; use MAE
                    sp_rho, sp_p = np.nan, np.nan
                    pe_r, pe_p = np.nan, np.nan

                records.append(
                    {
                        "LLM": short,
                        "Skill": skill,
                        "Spearman_rho": sp_rho,
                        "Spearman_p": sp_p,
                        "Pearson_r": pe_r,
                        "Pearson_p": pe_p,
                        "MAE": mae,
                        "N": len(pairs),
                        "constant_input": True,
                    }
                )
                continue

            # Normal case: both arrays have variance
            try:
                sp_rho, sp_p = spearmanr(h, l)
            except Exception:
                sp_rho, sp_p = np.nan, np.nan

            try:
                pe_r, pe_p = pearsonr(h, l)
            except Exception:
                pe_r, pe_p = np.nan, np.nan

            records.append(
                {
                    "LLM": short,
                    "Skill": skill,
                    "Spearman_rho": sp_rho,
                    "Spearman_p": sp_p,
                    "Pearson_r": pe_r,
                    "Pearson_p": pe_p,
                    "MAE": mae,
                    "N": len(pairs),
                    "constant_input": False,
                }
            )

    # Compute 'Average' row per LLM (excluding visual_artifacts)
    llms_in_records = set(r["LLM"] for r in records)
    for llm in llms_in_records:
        llm_recs = [r for r in records if r["LLM"] == llm and not r["Skill"].startswith("visual_artifacts")]
        sp_rho_vals = [r["Spearman_rho"] for r in llm_recs if not np.isnan(r["Spearman_rho"])]
        sp_p_vals = [r["Spearman_p"] for r in llm_recs if not np.isnan(r["Spearman_p"])]
        pe_r_vals = [r["Pearson_r"] for r in llm_recs if not np.isnan(r["Pearson_r"])]
        pe_p_vals = [r["Pearson_p"] for r in llm_recs if not np.isnan(r["Pearson_p"])]
        mae_vals = [r["MAE"] for r in llm_recs if not np.isnan(r["MAE"])]

        records.append(
            {
                "LLM": llm,
                "Skill": "Average",
                "Spearman_rho": np.mean(sp_rho_vals) if sp_rho_vals else np.nan,
                "Spearman_p": np.mean(sp_p_vals) if sp_p_vals else np.nan,
                "Pearson_r": np.mean(pe_r_vals) if pe_r_vals else np.nan,
                "Pearson_p": np.mean(pe_p_vals) if pe_p_vals else np.nan,
                "MAE": np.mean(mae_vals) if mae_vals else np.nan,
                "N": sum(r["N"] for r in llm_recs),
                "constant_input": False,
            }
        )

    return records


def plot_per_skill_correlation(df, out_path, include_artifacts=False):
    """
    Grouped bar chart: one group per skill, one bar per LLM.
    Y-axis = Spearman ρ.
    Skills where the LLM output was constant are annotated with † MAE=X.XX.
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    llms = df["LLM"].unique()
    # Extract artifact models for separate plotting
    va_models = [("visual_artifacts_PAL4VST", "#8c564b", "PAL4VST"), ("visual_artifacts_LEGION", "#e377c2", "LEGION")]
    va_data = []

    if include_artifacts:
        for skill_name, color, label in va_models:
            if skill_name in df["Skill"].unique():
                va_df = df[df["Skill"] == skill_name]
                va_rho = va_df["Spearman_rho"].dropna().iloc[0] if not va_df["Spearman_rho"].dropna().empty else np.nan
                va_data.append({"skill_name": skill_name, "rho": va_rho, "color": color, "label": label})

    has_va = len(va_data) > 0

    skills = [
        s
        for s in SKILL_ORDER
        if s in df["Skill"].unique() and not s.startswith("visual_artifacts") and s not in ["weather", "mood_feeling"]
    ]
    n_llms = len(llms)
    n_skills = len(skills)

    if n_skills == 0 and not has_va:
        print("No data to plot.")
        return

    # Find skills that have constant input across all models being plotted
    pivot_const_all = df.pivot_table(index="Skill", columns="LLM", values="constant_input", aggfunc="min")
    # If a skill has True for all models, we exclude it
    constant_skills = []
    for skill in list(skills):
        if skill == "Average":
            continue
        if skill in pivot_const_all.index and pivot_const_all.loc[skill].all():
            constant_skills.append(skill)
            skills.remove(skill)

    n_skills = len(skills)
    if n_skills == 0:
        print("All skills have constant input, nothing to plot.")
        return

    # Pivot tables
    pivot_rho = df.pivot_table(index="Skill", columns="LLM", values="Spearman_rho")
    pivot_rho = pivot_rho.reindex(skills)

    pivot_mae = df.pivot_table(index="Skill", columns="LLM", values="MAE")
    pivot_mae = pivot_mae.reindex(skills)

    pivot_const = df.pivot_table(index="Skill", columns="LLM", values="constant_input", aggfunc="max")
    pivot_const = pivot_const.reindex(skills)

    # Bar positions
    bar_width = 0.8 / max(n_llms, 1)
    x = np.arange(n_skills)

    fig, ax = plt.subplots(figsize=(max(14, n_skills * 0.9), 4))

    cmap = plt.get_cmap("tab10")
    for i, llm in enumerate(llms):
        if llm not in pivot_rho.columns:
            continue
        vals = pivot_rho[llm].values
        offset = (i - (n_llms - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, width=bar_width, label=llm, color=cmap(i % 10), edgecolor="white", linewidth=0.5)

        # Add value labels on top of bars, or MAE annotation for constant-input
        for j, (bar, v) in enumerate(zip(bars, vals)):
            skill = skills[j]
            is_const = llm in pivot_const.columns and pivot_const.loc[skill, llm] == True

            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                y_pos = v + 0.015 if v >= 0 else v - 0.015
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y_pos, f"{v:.2f}", ha="center", va=va, fontsize=10, fontweight="bold", rotation=90
                )

    # Draw visual_artifacts as separate single bars on the far right
    if has_va:
        # Draw separator line
        ax.axvline(n_skills - 0.5, color="black", linewidth=1.2, linestyle=":")

        for idx, va_item in enumerate(va_data):
            va_idx = n_skills + idx
            va_rho = va_item["rho"]

            ax.bar(va_idx, va_rho, width=0.8, color=va_item["color"], edgecolor="white", linewidth=0.5, label=va_item["label"])

            # Add value label
            if not np.isnan(va_rho):
                va_align = "bottom" if va_rho >= 0 else "top"
                y_pos = va_rho + 0.015 if va_rho >= 0 else va_rho - 0.015
                ax.text(va_idx, y_pos, f"{va_rho:.2f}", ha="center", va=va_align, fontsize=10, fontweight="bold", rotation=90)

            skills.append(va_item["skill_name"])

        x = np.arange(len(skills))

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    def _format_skill_label(s):
        if s.startswith("visual_artifacts"):
            return "Visual Artifacts"
        return s.replace("_", " ").title()

    ax.set_xticklabels([_format_skill_label(s) for s in skills], rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Spearman ρ  (LLM vs Avg Human)", fontsize=12)
    ax.set_title("Per-Skill Correlation: LLM vs Human Annotations", fontsize=14, pad=12)
    bottom_lim = min(0, pivot_rho.min().min() - 0.05) if not pivot_rho.empty else 0
    ax.set_ylim(bottom=bottom_lim, top=1.22)
    # Restrict ticks to 1.0 max so we don't display a 1.2 tick for the legend padding
    ticks = np.arange(np.floor(bottom_lim * 10) / 10, 1.1, 0.2)
    ax.set_yticks(ticks)

    # Custom legend ordering so PAL4VST/LEGION is visually separated
    handles, labels = ax.get_legend_handles_labels()
    # Ensure unique labels
    by_label = dict(zip(labels, handles))

    # Move legend to the top as a single row, inside the grid, shifted slightly left
    ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc="upper center", bbox_to_anchor=(0.4, 1.0), ncol=len(by_label))

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Footnote about excluded skills (placed below figure to avoid overlap with x-tick labels)
    footer_text = (
        "Note: Skills excluded due to constant LLM predictions for some of the models (ρ undefined): e.g. mood_feeling, weather, time"
    )
    if has_va:
        footer_text += "\n* Visual Artifacts are evaluated by designated models PAL4VST, not the LLM."

    fig.text(0.5, -0.06, footer_text, ha="center", fontsize=14, fontstyle="italic", color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")


def run():
    print("=" * 60)
    print("PER-SKILL CORRELATION: LLM vs HUMAN")
    print("=" * 60)

    # Process and pair human and LLM annotations
    paired_data = process_annotations()

    if not paired_data:
        print("No paired data found. Exiting.")
        return

    # Compute correlations
    records = compute_per_skill_correlation(paired_data)
    df = pd.DataFrame(records)

    # ── Console output ──
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.3f}".format)

    # Pivot for a nice table: rows=Skill, columns=(LLM, metric)
    for llm in df["LLM"].unique():
        sub = df[df["LLM"] == llm].set_index("Skill")[
            ["Spearman_rho", "Spearman_p", "Pearson_r", "Pearson_p", "MAE", "N", "constant_input"]
        ]
        print(f"\n── {llm} ──")
        print(sub.to_string())

    # # ── Save CSV ──
    # out_csv = BASE_DIR / "llm_human_correlation.csv"
    # df.to_csv(out_csv, index=False)
    # print(f"\nCSV saved to {out_csv}")

    # ── Plot ──
    out_plot = BASE_DIR / "llm_human_correlation.eps"
    plot_per_skill_correlation(df, out_plot, include_artifacts=True)

    out_plot = BASE_DIR / "llm_human_correlation.png"
    plot_per_skill_correlation(df, out_plot, include_artifacts=True)

    return df


if __name__ == "__main__":
    run()
