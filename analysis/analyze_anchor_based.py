#%%
from pathlib import Path
import pandas as pd
import numpy as np
import json

from utils.metrics import krippendorff_alpha, bootstrap_alpha
from config import ANNOTATIONS_DIR, GENERATION_PROMPTS_DIR

def compute_metrics(df, level, label="Global", print_top_disagreement=False):
    # Compute Unsure Rate
    # Unsure is marked as -1
    total_annotations = len(df)
    unsure_count = len(df[df['q_annot'] == -1])
    unsure_rate = unsure_count / total_annotations if total_annotations > 0 else 0
    
    print(f"[{label}] Unsure Rate: {unsure_rate:.2%} ({unsure_count}/{total_annotations})")
    
    # Prepare data for Agreement
    # Treat -1 (unsure) as NaN for agreement calculation
    df_agreement = df.copy()
    df_agreement['q_annot'] = df_agreement['q_annot'].replace(-1, np.nan)
    
    # Pivot: Rows=Items (model, img_id, q_id), Columns=Annotators
    pivot_df = df_agreement.pivot_table(
        index=['model', 'img_id', 'q_id'], 
        columns='annotator', 
        values='q_annot'
    )
    
    if pivot_df.empty:
         print(f"[{label}] No data for agreement computation.")
         print("-" * 30)
         return

    alpha, lo, hi = bootstrap_alpha(pivot_df, level=level)
    print(f"[{label}] Krippendorff's Alpha: {alpha:.4f} [{lo:.4f}, {hi:.4f}]")

    # Identify high disagreement items
    # Calculate std dev per item (row in pivot_df)
    # pivot_df values are 0 or 1 (NaN for unsure/missing)
    if print_top_disagreement:
        item_std = pivot_df.std(axis=1)
        
        # Sort descending
        top_disagreement = item_std.sort_values(ascending=False).head(10)
        
        print(f"\n[{label}] Top 10 Disagreement Items (Std Dev):")
        for (model_idx, img_id_idx, q_id_idx), std_val in top_disagreement.items():
            # Get count of valid annotations for context
            valid_count = pivot_df.loc[(model_idx, img_id_idx, q_id_idx)].count()
            print(f"  Model: {model_idx}, Image: {img_id_idx}, Q: {q_id_idx} - Std: {std_val:.4f} (Count: {valid_count})")
        
        # Optional: Print raw votes if needed
        # votes = pivot_df.loc[(model_idx, img_id_idx, q_id_idx)].to_dict()
        # print(f"    Votes: {votes}")
    
    print("-" * 30)

def process_anchor_annotations(base_path, level, images_to_exclude=None, models_to_exclude=None, annotators_to_exclude=None):
    images_to_exclude = images_to_exclude or []
    models_to_exclude = models_to_exclude or []
    annotators_to_exclude = annotators_to_exclude or []
    
    models = [f.name for f in base_path.glob("*") if f.is_dir()]
    print(f"Available Models in {base_path.name} are: ", models)
    
    anchor_annotations = []
    
    # Text Likert scores 1-5, standard BQA (yes/no/unsure) mappings
    # And we assume the score values 0-5 mapping also covers all numbers for likert
    bqa_to_int = {"unsure": -1, "yes": 1, "no": 0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
    
    for model in models:
        if model in models_to_exclude:
            continue
        for json_file in base_path.glob(f"{model}/*.json"):
            img_id = json_file.stem        
    
            if int(img_id) in images_to_exclude:            
                continue
    
            with open(json_file, 'r', encoding='utf-8') as f:
                img_annotations = json.load(f)
            
            for annotator, q_annotations in img_annotations.items():
                if annotator in annotators_to_exclude:
                    continue
                for q_id, q_annot in q_annotations.items():
                    # For Anchor Likert or standard BQA mapping
                    if q_annot in bqa_to_int:
                        # Exclude Likert 0 (Not Present/Missing) equivalent if needed, but keeping for now if 0 is structurally mapped
                        anchor_annotations.append({
                            'img_id': img_id,
                            'model': model,
                            'annotator': annotator,
                            'q_id': q_id,
                            'q_annot': bqa_to_int[q_annot]
                        })
                        
    return pd.DataFrame(anchor_annotations)

import itertools

def get_convergence_data(df, value_col, level, max_samples=50, ci=0.95, seed=0):
    """
    For each annotator subset size k, enumerate (or sample) annotator
    combinations of size k, compute Krippendorff's alpha on each, and return
    the mean alpha along with percentile bounds across combinations.

    Returns: dict[k] = {'mean': float, 'lo': float, 'hi': float, 'n': int}
    """
    df_agreement = df.copy()
    if value_col == 'q_annot':
        df_agreement[value_col] = df_agreement[value_col].replace(-1, np.nan)

    pivot_df = df_agreement.pivot_table(
        index=['model', 'img_id', 'q_id'],
        columns='annotator',
        values=value_col
    )

    annotators = pivot_df.columns.tolist()
    n_annotators = len(annotators)

    import random
    rng = random.Random(seed)

    results = {}
    lo_pct = 100 * (1 - ci) / 2
    hi_pct = 100 * (1 + ci) / 2

    for k in range(2, n_annotators + 1):
        combos = list(itertools.combinations(annotators, k))
        if len(combos) > max_samples:
            combos = rng.sample(combos, max_samples)

        alphas = []
        for combo in combos:
            sub_df = pivot_df[list(combo)]
            alpha = krippendorff_alpha(sub_df.values, level=level)
            if not np.isnan(alpha):
                alphas.append(alpha)

        if alphas:
            mean = float(np.mean(alphas))
            if len(alphas) >= 2:
                lo = float(np.percentile(alphas, lo_pct))
                hi = float(np.percentile(alphas, hi_pct))
            else:
                lo = hi = mean
            results[k] = {'mean': mean, 'lo': lo, 'hi': hi, 'n': len(alphas)}

    return results

import matplotlib.pyplot as plt

def plot_convergence(df_no_anchor, df_anchor, df_likert):
    print("="*50)
    print("COMPUTING CONVERGENCE PLOT")
    print("="*50)
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
        
    fig, axes = plt.subplots(3, 1, figsize=(10, 3), sharex=True)
    
    def add_plot_to_ax(ax, df, value_col, level, label, marker, color):
        if df is not None and not df.empty:
            conv = get_convergence_data(df, value_col, level)
            if conv:
                x = list(conv.keys())
                y = [conv[k]['mean'] for k in x]
                lo = [conv[k]['lo'] for k in x]
                hi = [conv[k]['hi'] for k in x]
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
                ax.plot(x, y, marker=marker, markersize=8, label=label, linewidth=2.5, alpha=0.85, color=color)
                if len(x) > 0:
                    mid_idx = len(x) // 2
                    ax.text(x[mid_idx], y[mid_idx], f'  {label}', fontsize=16,
                            verticalalignment='bottom', horizontalalignment='left',
                            color=color)
                min_y, max_y = min(lo), max(hi)
                pad = max(0.02, (max_y - min_y) * 0.1)
                ax.set_ylim(max(0.0, min_y - pad), min(1.0, max_y + pad))

                ax.grid(True, linestyle='--', alpha=0.5, axis='y')
                ax.grid(False, axis='x')
                if x:
                     ax.set_xticks(range(2, max(x) + 1))
                return True
        return False

    has_no_anchor = add_plot_to_ax(axes[0], df_no_anchor, "q_annot", "nominal", "No Anchor BQA", "^", '#d62728')
    has_anchor = add_plot_to_ax(axes[1], df_anchor, "q_annot", "nominal", "Anchor BQA", 's', '#ff7f0e')
    has_likert = add_plot_to_ax(axes[2], df_likert, "q_annot", "ordinal", "Anchor Likert", 'o', '#1f77b4')
    
    axes[2].set_xlabel('Number of Annotators', fontsize=16)
    
    if not has_no_anchor: axes[0].set_visible(False)
    if not has_anchor: axes[1].set_visible(False)
    if not has_likert: axes[2].set_visible(False)

    fig.supylabel("Krippendorff's Alpha", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    
    out_path = Path("anchor_convergence_plot.png")
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f"Convergence plot saved to {out_path.absolute()}")


def run_analysis(include_model_stats=False, include_skill_stats=False):
    images_to_exclude = []
    models_to_exclude = []
    annotators_to_exclude = []
    
    strategies = [
        ("No Anchor BQA", ANNOTATIONS_DIR / "no_anchor_bqa" / "vqa", "nominal"),
        ("Anchor BQA", ANNOTATIONS_DIR / "anchor_bqa" / "vqa", "nominal"),
        ("Anchor Likert", ANNOTATIONS_DIR / "anchor_likert" / "vqa", "ordinal"),
    ]
    
    # Load prompts to map to skills
    prompts_file = GENERATION_PROMPTS_DIR / "v8.1-gpt-5-mini" / "processed_prompts.json"
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    # Create mapping: (prompt_id, uid) -> (skill, subskill)
    q_to_skill = {}
    for pdata in prompts_data:
        p_id = str(pdata["prompt_id"])
        for ann in pdata.get("annotations", []):
            q_id = str(ann["uid"])
            skill = ann.get("skill", "unknown")
            subskill = ann.get("subskill", "")
            q_to_skill[(p_id, q_id)] = f"{skill}_{subskill}" if subskill else skill

    def get_skill(row):
        return q_to_skill.get((str(row['img_id']), str(row['q_id'])), "unknown_skill")

    dfs = []
    skills_to_exclude = []

    for name, path_str, level in strategies:
        print(f"\n==================================================")
        print(f"ANALYZING {name.upper()}")
        print(f"==================================================")
        base_path = Path(path_str)
        
        if not base_path.exists():
            print(f"Path not found: {base_path}")
            dfs.append(None)
            continue
            
        df = process_anchor_annotations(base_path, level, images_to_exclude, models_to_exclude, annotators_to_exclude)
        
        if df.empty:
            print("No annotations found.")
            dfs.append(None)
            continue
            
        df['skill'] = df.apply(get_skill, axis=1)
        
        if skills_to_exclude:
            initial_count = len(df)
            df = df[~df['skill'].isin(skills_to_exclude)]
            print(f"\nExcluded {initial_count - len(df)} annotations matching skills: {skills_to_exclude}")
            
        # Global Stats
        compute_metrics(df, level=level, label=f"Global ({name})", print_top_disagreement=False)

        # Per Model Stats
        if include_model_stats:
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                compute_metrics(model_df, level=level, label=f"Model: {model}", print_top_disagreement=False)
            
        dfs.append(df)
        
        # Optional: Print Specific Skill Stats if requested
        if include_skill_stats:
            print(f"\n--- Per Skill Statistics ({name}) ---")
            for skill in df['skill'].unique():
                skill_df = df[df['skill'] == skill]
                if not skill_df.empty:
                    compute_metrics(skill_df, level=level, label=f"Skill: {skill}", print_top_disagreement=False)
                
    # Unpack for plotting
    df_no_anchor, df_anchor, df_likert = dfs
    
    # Plot convergence
    plot_convergence(df_no_anchor, df_anchor, df_likert)


if __name__ == "__main__":
    run_analysis(include_model_stats=False, include_skill_stats=False)
