#%%
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

from utils.metrics import krippendorff_alpha, bootstrap_alpha
from config import ANNOTATIONS_DIR

def compute_pairwise_iou(masks):
    """
    Compute average pairwise IoU for a list of masks.
    masks: list of boolean numpy arrays
    """
    if len(masks) < 2:
        return np.nan
    
    ious = []
    for m1, m2 in itertools.combinations(masks, 2):
        intersection = (m1 & m2).sum()
        union = (m1 | m2).sum()
        
        if union == 0:
            iou = 1.0 # Both empty
        else:
            iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious)

def compute_mask_metrics(df, label="Global", print_top_disagreement=False, extreme_diff_threshold_rate=None):
    print(f"--- {label} ---")
    
    # 1. Pairwise IoU
    iou_per_image = []
    grouped = df.groupby(['model', 'img_id'])
    
    for name, group in grouped:
        model, img_id = name
        masks = group['artifacts_mask'].tolist()
        if len(masks) > 1:
            avg_iou = compute_pairwise_iou(masks)
            iou_per_image.append(avg_iou)
            
    global_iou = np.nanmean(iou_per_image)
    print(f"Average Pairwise IoU: {global_iou:.4f}")

    # 2. Krippendorff's Alpha on Artifact Rate (Interval)
    pivot_rate = df.pivot_table(
        index=['model', 'img_id'], 
        columns='annotator', 
        values='artifacts_rate'
    )
    
    if not pivot_rate.empty:
        alpha_interval, lo, hi = bootstrap_alpha(pivot_rate, level="interval")
        print(f"Krippendorff's Alpha (Rate - Interval): {alpha_interval:.4f} [{lo:.4f}, {hi:.4f}]")
        
        if extreme_diff_threshold_rate is not None:
            item_diff = pivot_rate.max(axis=1) - pivot_rate.min(axis=1)
            extreme_count = (item_diff >= extreme_diff_threshold_rate).sum()
            valid_items = (pivot_rate.count(axis=1) >= 2).sum()
            if valid_items > 0:
                extreme_rate = extreme_count / valid_items
                print(f"Extreme Disagreement Rate (Rate >= {extreme_diff_threshold_rate}%): {extreme_rate:.2%} ({extreme_count}/{valid_items})")

    # Print Disagreement
    if print_top_disagreement and not pivot_rate.empty:
        std_rate = pivot_rate.std(axis=1)
        disagreement_df = std_rate.reset_index(name='std_rate').sort_values(by='std_rate', ascending=False)
        disagreement_df = disagreement_df.dropna(subset=['std_rate'])
        top_disagreement = disagreement_df.head(10)
        
        print(f"\n[{label}] Top 10 Disagreement Items (Highest Std Rate):")
        for _, row in top_disagreement.iterrows():
             print(f"  Model: {row['model']}, Image: {row['img_id']} - Std Rate: {row['std_rate']:.4f}")
    
    print("-" * 30)


def process_artifacts_likert(base_path, images_to_exclude=None, models_to_exclude=None, annotators_to_exclude=None):
    images_to_exclude = images_to_exclude or []
    models_to_exclude = models_to_exclude or []
    annotators_to_exclude = annotators_to_exclude or []
    
    models = [f.name for f in base_path.glob("*") if f.is_dir()]
    print("Available Likert Models are: ", models)
    
    annotations_list = []
    
    for model in models:
        if model in models_to_exclude:
            continue
        for json_file in base_path.glob(f"{model}/*.json"):
            img_id = json_file.stem        

            if str(img_id) in images_to_exclude or int(img_id) in images_to_exclude:            
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                img_annotations = json.load(f)
            for annotator, q_annotations in img_annotations.items():
                if annotator in annotators_to_exclude:
                    continue
                
                if "_aesthetics_rating" in q_annotations:
                    val = q_annotations["_aesthetics_rating"]
                    if val > 0: # valid likert, excluding 0
                        annotations_list.append({
                            'img_id': img_id,
                            'model': model,
                            'annotator': annotator,
                            'q_annot': float(val)
                        })

    return pd.DataFrame(annotations_list)

def compute_likert_metrics(df, label="Global", print_top_disagreement=False, extreme_diff_threshold=None):
    print(f"--- {label} ---")
    
    df_agreement = df.copy()
    df_agreement['q_annot'] = df_agreement['q_annot'].replace(-1, np.nan)
    
    pivot_df = df_agreement.pivot_table(
        index=['model', 'img_id'], 
        columns='annotator', 
        values='q_annot'
    )
    
    if pivot_df.empty:
         print(f"No data for agreement computation.")
         return

    alpha, lo, hi = bootstrap_alpha(pivot_df, level="ordinal")
    print(f"Krippendorff's Alpha (Ordinal): {alpha:.4f} [{lo:.4f}, {hi:.4f}]")

    item_var = pivot_df.var(axis=1)
    avg_var = item_var.mean()
    print(f"Average Intra-Item Variance: {avg_var:.4f}")

    if extreme_diff_threshold is not None:
        item_diff = pivot_df.max(axis=1) - pivot_df.min(axis=1)
        extreme_count = (item_diff >= extreme_diff_threshold).sum()
        valid_items = (pivot_df.count(axis=1) >= 2).sum()
        if valid_items > 0:
            extreme_rate = extreme_count / valid_items
            print(f"Extreme Disagreement Rate (>= {extreme_diff_threshold}): {extreme_rate:.2%} ({extreme_count}/{valid_items})")

    if print_top_disagreement:
        item_std = pivot_df.std(axis=1)
        top_disagreement = item_std.sort_values(ascending=False).head(10)
        
        print(f"\nTop 10 Disagreement Items (Std Dev):")
        for (model_idx, img_id_idx), std_val in top_disagreement.items():
            valid_count = pivot_df.loc[(model_idx, img_id_idx)].count()
            print(f"  Model: {model_idx}, Image: {img_id_idx} - Std: {std_val:.4f} (Count: {valid_count})")
            
    print("-" * 30)

def get_convergence_data(df, value_col, level, max_samples=50, ci=0.95, seed=0):
    """Returns dict[k] = {'mean', 'lo', 'hi', 'n'} of alpha across annotator subsets."""
    df_agreement = df.copy()
    if value_col == 'q_annot':
        df_agreement[value_col] = df_agreement[value_col].replace(-1, np.nan)

    pivot_df = df_agreement.pivot_table(
        index=['model', 'img_id'],
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

def plot_convergence(df_likert, df_masks):
    print("="*50)
    print("COMPUTING CONVERGENCE PLOT")
    print("="*50)
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
        
    fig, axes = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
    
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

    has_likert = add_plot_to_ax(axes[0], df_likert, "q_annot", "ordinal", "Likert Score", 'o', '#1f77b4')
    has_rate = add_plot_to_ax(axes[1], df_masks, "artifacts_rate", "interval", "Brush Annotation", 's', '#ff7f0e')
    
    axes[1].set_xlabel('Number of Annotators', fontsize=16)
    
    if not has_likert: axes[0].set_visible(False)
    if not has_rate: axes[1].set_visible(False)

    fig.supylabel("Krippendorff's Alpha", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    
    out_path = Path("artifacts_convergence_plot.png")
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f"Convergence plot saved to {out_path.absolute()}")

#%%
def process_artifacts_brush(images_to_exclude=None, models_to_exclude=None, annotators_to_exclude=None, verbose=True):
    images_to_exclude = images_to_exclude or []
    models_to_exclude = models_to_exclude or []
    annotators_to_exclude = annotators_to_exclude or []

    base_path = ANNOTATIONS_DIR / "artifacts_brush" / "artifact_mask"

    if not base_path.exists():
        if verbose:
            print(f"Path not found: {base_path}")
        return None

    models = [f.name for f in base_path.glob("*") if f.is_dir()]
    if verbose:
        print("Available Brush Models are: ", models)

    artifacts_masks = []
    for model in models:
        if model in models_to_exclude:
            continue
        for png_file in base_path.glob(f"{model}/*.png"):
            parts = str(png_file.stem).split("_")
            if len(parts) >= 2:
                img_id = parts[0]
                annotator = parts[1]
            else:
                continue

            if annotator in annotators_to_exclude:
                continue
            if img_id in images_to_exclude:
                continue

            artifacts_mask = plt.imread(str(png_file), format='png')
            if artifacts_mask.ndim == 3:
                artifacts_mask = artifacts_mask.mean(-1)

            artifacts_mask = artifacts_mask > 0

            artifacts_masks.append({
                'img_id': img_id,
                'model': model,
                'annotator': annotator,
                'artifacts_mask': artifacts_mask,
                'artifacts_rate': artifacts_mask.sum() / (artifacts_mask.size) * 100
            })

    if verbose:
        print("Loaded {} masks".format(len(artifacts_masks)))

    if not artifacts_masks:
        return None

    return pd.DataFrame(artifacts_masks)


images_to_exclude = []
annotators_to_exclude = []
models_to_exclude = []


#%%
def run_analysis(include_model_stats=False):
    print("="*50)
    print("LOADING BRUSH ARTIFACTS MASKS")
    print("="*50)
    artifacts_dataframe = process_artifacts_brush(images_to_exclude, models_to_exclude, annotators_to_exclude)

    print("="*50)
    print("BRUSH ARTIFACTS MASKS ANALYSIS")
    print("="*50)

    if artifacts_dataframe is not None and not artifacts_dataframe.empty:
        # Global
        compute_mask_metrics(artifacts_dataframe, label="Global (Brush Mask)", print_top_disagreement=False, extreme_diff_threshold_rate=40.0)
        
        # Per Model
        if include_model_stats:
            for model in artifacts_dataframe['model'].unique():
                model_df = artifacts_dataframe[artifacts_dataframe['model'] == model].copy()
                if not model_df.empty:
                    compute_mask_metrics(model_df, label=f"Model: {model} (Brush Mask)", print_top_disagreement=False, extreme_diff_threshold_rate=40.0)
    else:
        print("No brush artifacts loaded.")

    print("\n")
    print("="*50)
    print("ARTIFACTS LIKERT ANALYSIS")
    print("="*50)
    likert_path = ANNOTATIONS_DIR / "artifacts_likert" / "vqa"
    
    df_likert = None
    if likert_path.exists():
        df_likert = process_artifacts_likert(likert_path, images_to_exclude, models_to_exclude, annotators_to_exclude)
        if not df_likert.empty:
            compute_likert_metrics(df_likert, label="Global (Aesthetics Likert)", print_top_disagreement=False, extreme_diff_threshold=2.0)
            
            if include_model_stats:
                for model in df_likert['model'].unique():
                    model_df = df_likert[df_likert['model'] == model]
                    compute_likert_metrics(model_df, label=f"Model: {model} (Aesthetics Likert)", extreme_diff_threshold=2.0)

        else:
            print("No artifacts likert annotations found.")
    else:
        print(f"Path not found: {likert_path}")
        
    # Plot convergence
    plot_convergence(df_likert, artifacts_dataframe)

if __name__ == "__main__":
    run_analysis(include_model_stats=False)
