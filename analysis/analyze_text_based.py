#%%

from pathlib import Path
import pandas as pd
import numpy as np
import json

from utils.metrics import krippendorff_alpha, bootstrap_alpha
from config import ANNOTATIONS_DIR
from apps.annotation_stats_app import compute_text_per_word_stats, extract_text_per_word_judgments

def compute_metrics(df, level, label="Global", print_top_disagreement=False, extreme_diff_threshold=None):
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

    # Average Intra-Item Variance (Stability)
    item_var = pivot_df.var(axis=1)
    avg_var = item_var.mean()
    print(f"[{label}] Average Intra-Item Variance: {avg_var:.4f}")

    # Extreme Disagreement Rate
    if extreme_diff_threshold is not None:
        item_diff = pivot_df.max(axis=1) - pivot_df.min(axis=1)
        extreme_count = (item_diff >= extreme_diff_threshold).sum()
        valid_items = (pivot_df.count(axis=1) >= 2).sum()
        if valid_items > 0:
            extreme_rate = extreme_count / valid_items
            print(f"[{label}] Extreme Disagreement Rate (>= {extreme_diff_threshold}): {extreme_rate:.2%} ({extreme_count}/{valid_items})")
        else:
            print(f"[{label}] Extreme Disagreement Rate: N/A (Not enough overlapping annotations)")

    # Identify high disagreement items
    # Calculate std dev per item (row in pivot_df)
    if print_top_disagreement:
        item_std = pivot_df.std(axis=1)
        
        # Sort descending
        top_disagreement = item_std.sort_values(ascending=False).head(10)
        
        print(f"\n[{label}] Top 10 Disagreement Items (Std Dev):")
        for (model_idx, img_id_idx, q_id_idx), std_val in top_disagreement.items():
            # Get count of valid annotations for context
            valid_count = pivot_df.loc[(model_idx, img_id_idx, q_id_idx)].count()
            print(f"  Model: {model_idx}, Image: {img_id_idx}, Q: {q_id_idx} - Std: {std_val:.4f} (Count: {valid_count})")
        
    print("-" * 30)

def process_text_likert(base_path, images_to_exclude=None, models_to_exclude=None, annotators_to_exclude=None):
    images_to_exclude = images_to_exclude or []
    models_to_exclude = models_to_exclude or []
    annotators_to_exclude = annotators_to_exclude or []
    
    models = [f.name for f in base_path.glob("*") if f.is_dir()]
    print("Available Models are: ", models)
    
    annotations_list = []
    
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
                
                # text_likert stores the score in "_text_likert"
                if "_text_likert" in q_annotations:
                    val = q_annotations["_text_likert"]
                    if val > 0: # valid likert, excluding 0
                        annotations_list.append({
                            'img_id': img_id,
                            'model': model,
                            'annotator': annotator,
                            'q_id': "text_likert",
                            'q_annot': float(val)
                        })

    return pd.DataFrame(annotations_list)

def process_text_per_word(base_path, images_to_exclude=None, models_to_exclude=None, annotators_to_exclude=None):
    images_to_exclude = images_to_exclude or []
    models_to_exclude = models_to_exclude or []
    annotators_to_exclude = annotators_to_exclude or []
    
    models = [f.name for f in base_path.glob("*") if f.is_dir()]
    print("Available Models are: ", models)
    
    annotations_list = []
    word_annotations_list = []
    
    for model in models:
        if model in models_to_exclude:
            continue
        for json_file in base_path.glob(f"{model}/*.json"):
            img_id = json_file.stem        

            if int(img_id) in images_to_exclude:            
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                img_annotations = json.load(f)
                
            # Use inspiration from apps/annotation_stats_app.py
            # compute_text_per_word_stats gives an accuracy (0-100) per annotator for this image
            stats = compute_text_per_word_stats(img_annotations)
            
            for annotator, s in stats.items():
                if annotator in annotators_to_exclude:
                    continue
                
                accuracy = s.get("word_accuracy", None)
                if accuracy is not None and accuracy > 0: # excluding 0
                    annotations_list.append({
                        'img_id': img_id,
                        'model': model,
                        'annotator': annotator,
                        'q_id': "text_accuracy",  # treat accuracy as the continuous value
                        'q_annot': float(accuracy)
                    })
                    
            # Word-level annotations
            for annotator, q_annotations in img_annotations.items():
                if annotator in annotators_to_exclude:
                    continue
                judgments = extract_text_per_word_judgments(q_annotations)
                for word_key, val in judgments.items():
                    word_annotations_list.append({
                        'img_id': img_id,
                        'model': model,
                        'annotator': annotator,
                        'q_id': word_key,
                        'q_annot': float(val)
                    })

    return pd.DataFrame(annotations_list), pd.DataFrame(word_annotations_list)

import matplotlib.pyplot as plt
import itertools
import random

def get_convergence_data(df, level, max_samples=50, ci=0.95, seed=0):
    """Returns dict[k] = {'mean', 'lo', 'hi', 'n'} of alpha across annotator subsets."""
    df_agreement = df.copy()
    df_agreement['q_annot'] = df_agreement['q_annot'].replace(-1, np.nan)
    pivot_df = df_agreement.pivot_table(
        index=['model', 'img_id', 'q_id'],
        columns='annotator',
        values='q_annot'
    )

    annotators = pivot_df.columns.tolist()
    n_annotators = len(annotators)

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

def plot_convergence(df_likert, df_per_word_image, df_per_word_word, df_artificial_bqa=None):
    print("="*50)
    print("COMPUTING CONVERGENCE PLOT")
    print("="*50)
    
    # Prettier style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass # Fallback if style not available
        
    num_plots = 4 if df_artificial_bqa is not None else 3
    fig, axes = plt.subplots(num_plots, 1, figsize=(13, num_plots * 1.15), sharex=True)
    if num_plots == 3:
        axes = list(axes) + [None] # pad so axes[3] doesn't error
    
    def add_plot_to_ax(ax, df, level, label, marker, color):
        if df is not None and not df.empty:
            conv = get_convergence_data(df, level)
            if conv:
                x = list(conv.keys())
                y = [conv[k]['mean'] for k in x]
                lo = [conv[k]['lo'] for k in x]
                hi = [conv[k]['hi'] for k in x]
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
                ax.plot(x, y, marker=marker, markersize=8, label=label, linewidth=2.5, alpha=0.85, color=color)
                if len(x) > 0:
                    mid_idx = len(x) // 2
                    ax.text(x[mid_idx], y[mid_idx], f'  {label}', fontsize=20,
                            verticalalignment='bottom', horizontalalignment='left',
                            color=color)
                min_y, max_y = min(lo), max(hi)
                pad = max(0.02, (max_y - min_y) * 0.1)
                ax.set_ylim(max(0.0, min_y - pad), min(1.0, max_y + pad))

                ax.grid(True, linestyle='--', alpha=0.5, axis='y')
                ax.grid(False, axis='x')
                if x:
                     ax.set_xticks(range(2, max(x) + 1))
                ax.tick_params(axis='both', which='major', labelsize=16)
                return True
        return False

    has_bqa = False
    if axes[3] is not None:
        has_bqa = add_plot_to_ax(axes[0], df_artificial_bqa, "nominal", "BQA", 'd', '#d62728')
        likert_ax, image_ax, word_ax = axes[1], axes[2], axes[3]
    else:
        likert_ax, image_ax, word_ax = axes[0], axes[1], axes[2]
        
    has_likert = add_plot_to_ax(likert_ax, df_likert, "ordinal", "Likert Score", 'o', '#1f77b4')
    has_image = add_plot_to_ax(image_ax, df_per_word_image, "interval", "Word-Level (Whole)", 's', '#ff7f0e')
    has_word = add_plot_to_ax(word_ax, df_per_word_word, "nominal", "Word-Level (Per Word)", '^', '#2ca02c')
    
    # Add X-axis label only to the bottom subplot
    if axes[3] is not None:
        axes[3].set_xlabel('Number of Annotators', fontsize=20)
    else:
        axes[2].set_xlabel('Number of Annotators', fontsize=20)
    
    # Clean up empty subplots if any data is missing
    if axes[3] is not None and not has_bqa: axes[0].set_visible(False)
    if not has_likert: likert_ax.set_visible(False)
    if not has_image: image_ax.set_visible(False)
    if not has_word: word_ax.set_visible(False)

    fig.supylabel("Krippendorff's Alpha", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1) # Make room for supylabel
    
    out_path = Path("text_convergence_plot.png")
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f"Convergence plot saved to {out_path.absolute()}")

def run_analysis(include_model_stats=False, include_artificial_bqa=False):
    images_to_exclude = []
    models_to_exclude = [] 
    annotators_to_exclude = []
    
    # 1. Text Likert Analysis
    print("="*50)
    print("TEXT LIKERT ANALYSIS")
    print("="*50)
    likert_path = ANNOTATIONS_DIR / "text_likert" / "vqa"
    df_likert = None
    df_artificial_bqa = None
    if likert_path.exists():
        df_likert = process_text_likert(likert_path, images_to_exclude, models_to_exclude, annotators_to_exclude)
        if not df_likert.empty:
            print(f"\n--- Statistics (Text Likert) ---")
            compute_metrics(df_likert, level="ordinal", label="Global (Text Likert)", extreme_diff_threshold=2.0)
            
            if include_model_stats:
                for model in df_likert['model'].unique():
                    model_df = df_likert[df_likert['model'] == model]
                    compute_metrics(model_df, level="ordinal", label=f"Model: {model} (Text Likert)", extreme_diff_threshold=2.0)
                    
            if include_artificial_bqa:
                df_artificial_bqa = df_likert.copy()
                df_artificial_bqa['q_annot'] = np.where(df_artificial_bqa['q_annot'] > 3, 1, 0)
                
                print(f"\n--- Statistics (Artificial BQA: Likert > 3 is Yes) ---")
                compute_metrics(df_artificial_bqa, level="nominal", label="Global (BQA)", extreme_diff_threshold=1.0)
                if include_model_stats:
                    for model in df_artificial_bqa['model'].unique():
                        model_df = df_artificial_bqa[df_artificial_bqa['model'] == model]
                        compute_metrics(model_df, level="nominal", label=f"Model: {model} (BQA)", extreme_diff_threshold=1.0)
        else:
            print("No text likert annotations found.")
    else:
        print(f"Path not found: {likert_path}")
        
    print("\n")
    # 2. Text Per Word Analysis
    print("="*50)
    print("TEXT PER WORD ANALYSIS")
    print("="*50)
    per_word_path = ANNOTATIONS_DIR / "text_per_word" / "vqa"
    df_per_word = None
    df_word_level = None
    if per_word_path.exists():
        df_per_word, df_word_level = process_text_per_word(per_word_path, images_to_exclude, models_to_exclude, annotators_to_exclude)
        
        if not df_per_word.empty:
            print(f"\n--- Statistics (Image-Level Accuracy Score) ---")
            # Using interval since accuracy is a percentage score 0-100
            compute_metrics(df_per_word, level="interval", label="Global (Image Level Accuracy Score)", extreme_diff_threshold=40.0, print_top_disagreement=False)
            
            if include_model_stats:
                for model in df_per_word['model'].unique():
                    model_df = df_per_word[df_per_word['model'] == model]
                    compute_metrics(model_df, level="interval", label=f"Model: {model} (Image Level Accuracy Score)", extreme_diff_threshold=40.0)
        else:
            print("No text per word image-level annotations found.")

        if not df_word_level.empty:
            print(f"\n--- Statistics (Word-Level Score) ---")
            # Using nominal since word judgement is binary 0 or 1
            compute_metrics(df_word_level, level="nominal", label="Global (Word-Level)", extreme_diff_threshold=1.0)
            
            if include_model_stats:
                for model in df_word_level['model'].unique():
                    model_df = df_word_level[df_word_level['model'] == model]
                    compute_metrics(model_df, level="nominal", label=f"Model: {model} (Word-Level)", extreme_diff_threshold=1.0)
        else:
            print("No text per word word-level annotations found.")
            
    else:
        print(f"Path not found: {per_word_path}")
        
    # 3. Plot convergence
    plot_convergence(df_likert, df_per_word, df_word_level, df_artificial_bqa)

if __name__ == "__main__":
    run_analysis(include_model_stats=False, include_artificial_bqa=True)
