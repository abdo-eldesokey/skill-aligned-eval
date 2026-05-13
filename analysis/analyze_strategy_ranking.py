"""
Cross-strategy model-ranking analysis.

For each "skill family" (anchor-based, text-based, artifacts-based), compute a
per-model aggregate score under every applicable annotation strategy, rank the
models, and quantify whether the ranking is consistent across strategies.

This addresses Reviewer 1's "agreement != quality" complaint by using ranking
stability under different protocols as a downstream-quality proxy, and provides
bootstrap 95% CIs (resampling items with replacement) to know when ranking
flips are statistically distinguishable from noise on a small sample.

Outputs (under analysis/outputs/):
    - ranking_anchor.tex / ranking_text.tex / ranking_artifacts.tex
    - ranking_grouped_bars.pdf
"""

#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

from config import ANNOTATIONS_DIR
from utils.metrics import bootstrap_alpha
from analysis.analyze_anchor_based import process_anchor_annotations
from analysis.analyze_text_based import process_text_likert, process_text_per_word
from analysis.analyze_artifacts import process_artifacts_likert, process_artifacts_brush


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1) Per-strategy loading -> per-item score DataFrame:
#    [model, item_id, score, coverage]
#    - score:    annotator-mean of valid responses in [0, 1] (higher = better)
#    - coverage: fraction of annotators who returned a valid (non-unsure)
#                response for this item, in [0, 1]
# ---------------------------------------------------------------------------
def _per_item_score(df, item_keys, value_col="q_annot", unsure_value=np.nan):
    """
    Aggregate raw per-annotator rows (NaN = invalid/unsure) to per-item scores.

    If unsure_value is np.nan: drop unsure responses (only valid ones contribute
        to the per-item mean; coverage records the drop rate).
    Otherwise: replace NaN with unsure_value (e.g. 0.0 to penalize unsure as a
        protocol failure, or 0.5 to treat as neutral) before averaging.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["model", "item_id", "score", "coverage"])
    sub = df.copy()
    item_id_col = sub[item_keys].astype(str).agg("|".join, axis=1)
    sub["item_id"] = item_id_col

    rows = []
    for (model, item_id), g in sub.groupby(["model", "item_id"]):
        valid = g[value_col].dropna()
        n_total = len(g)
        n_valid = len(valid)
        coverage = n_valid / n_total if n_total else 0.0
        if np.isnan(unsure_value):
            score = valid.mean() if n_valid > 0 else np.nan
        else:
            replaced = g[value_col].fillna(unsure_value)
            score = replaced.mean()
        rows.append({"model": model, "item_id": item_id, "score": score, "coverage": coverage})
    return pd.DataFrame(rows)


def load_anchor_family(unsure_value=0.0):
    """
    Returns {strategy_name: per_item_df with [model, item_id, score, coverage]}.

    unsure_value controls how 'unsure' BQA responses are folded into the score:
        0.0 (default) - unsure is a protocol failure: model gets no credit.
                        Keeps the same item set across strategies.
        np.nan        - drop unsure responses (the original behavior). Be aware
                        that No-Anchor BQA scores then derive from a much
                        smaller subset of annotators per item.
    """
    out = {}
    specs = [
        ("No-Anchor BQA", "no_anchor_bqa", "bqa"),
        ("Anchor BQA", "anchor_bqa", "bqa"),
        ("Anchor Likert", "anchor_likert", "likert"),
    ]
    for label, subdir, kind in specs:
        path = ANNOTATIONS_DIR / subdir / "vqa"
        if not path.exists():
            print(f"[anchor] missing path: {path}")
            continue
        df = process_anchor_annotations(path, level="nominal" if kind == "bqa" else "ordinal")
        if df is None or df.empty:
            continue
        df = df.copy()
        # Unify "invalid response" -> NaN so coverage is computed consistently.
        df["q_annot"] = df["q_annot"].replace(-1, np.nan)
        if kind == "likert":
            # Likert 0 means "not applicable / skipped" (matches text/artifacts pipelines).
            df.loc[df["q_annot"] == 0, "q_annot"] = np.nan
            df["q_annot"] = (df["q_annot"] - 1.0) / 4.0  # 1..5 -> 0..1
        out[label] = _per_item_score(df, ["img_id", "q_id"], unsure_value=unsure_value)
    return out


def load_text_family():
    out = {}

    likert_path = ANNOTATIONS_DIR / "text_likert" / "vqa"
    if likert_path.exists():
        df = process_text_likert(likert_path)
        if df is not None and not df.empty:
            df = df.copy()
            df["q_annot"] = (df["q_annot"] - 1.0) / 4.0  # 1..5 -> 0..1
            out["Text Likert"] = _per_item_score(df, ["img_id"])

    per_word_path = ANNOTATIONS_DIR / "text_per_word" / "vqa"
    if per_word_path.exists():
        img_df, word_df = process_text_per_word(per_word_path)
        if img_df is not None and not img_df.empty:
            img_df = img_df.copy()
            img_df["q_annot"] = img_df["q_annot"] / 100.0
            out["Per-Word (image)"] = _per_item_score(img_df, ["img_id"])
        if word_df is not None and not word_df.empty:
            out["Per-Word (word)"] = _per_item_score(word_df, ["img_id", "q_id"])
    return out


def load_artifacts_family():
    out = {}

    likert_path = ANNOTATIONS_DIR / "artifacts_likert" / "vqa"
    if likert_path.exists():
        df = process_artifacts_likert(likert_path)
        if df is not None and not df.empty:
            df = df.copy()
            df["q_annot"] = 1.0 - (df["q_annot"] - 1.0) / 4.0  # invert: higher = cleaner
            out["Aesthetics Likert"] = _per_item_score(df, ["img_id"])

    brush_df = process_artifacts_brush(verbose=False)
    if brush_df is not None and not brush_df.empty:
        b = brush_df.copy()
        b["q_annot"] = 1.0 - b["artifacts_rate"] / 100.0  # higher = cleaner
        out["Brush Mask"] = _per_item_score(b, ["img_id"])
    return out


# ---------------------------------------------------------------------------
# 2) Bootstrap per-model means + cross-strategy ranking metrics
# ---------------------------------------------------------------------------
def per_model_ci(per_item_df, n_boot=1000, ci=0.95, seed=0):
    """
    Returns DataFrame indexed by model with columns [mean, lo, hi, coverage, n_items].
    Bootstrap resamples items (rows of per_item_df) with replacement.
    Items with NaN score are kept in the n_items count but skipped during the mean.
    """
    rng = np.random.default_rng(seed)
    rows = []
    has_coverage = "coverage" in per_item_df.columns
    for model, sub in per_item_df.groupby("model"):
        scores = sub["score"].to_numpy()
        coverage = sub["coverage"].to_numpy() if has_coverage else None
        n = len(scores)
        if n == 0:
            continue
        valid_scores = scores[~np.isnan(scores)]
        if len(valid_scores) == 0:
            continue
        boots = rng.choice(valid_scores, size=(n_boot, len(valid_scores)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
        rows.append({
            "model": model,
            "mean": float(valid_scores.mean()),
            "lo": float(lo),
            "hi": float(hi),
            "coverage": float(np.nanmean(coverage)) if coverage is not None else 1.0,
            "n_items": int(n),
        })
    return pd.DataFrame(rows).set_index("model").sort_index()


def ranking_table(family_strategies, n_boot=1000, ci=0.95, seed=0):
    """
    Returns:
        score_df:    rows=models, cols=strategies, values=mean score
        ci_df:       same shape, values="[lo, hi]" strings
        coverage_df: rows=models, cols=strategies, mean per-item coverage
        rank_df:     rows=strategies, cols=[winner, ranking, coverage_avg]
    """
    score_rows, ci_rows, cov_rows = {}, {}, {}
    rank_rows = []
    for label, df in family_strategies.items():
        cdf = per_model_ci(df, n_boot=n_boot, ci=ci, seed=seed)
        score_rows[label] = cdf["mean"]
        ci_rows[label] = cdf.apply(lambda r: f"[{r['lo']:.3f}, {r['hi']:.3f}]", axis=1)
        cov_rows[label] = cdf["coverage"]
        ranked = cdf["mean"].dropna().sort_values(ascending=False)
        if not ranked.empty:
            rank_rows.append({
                "strategy": label,
                "winner": ranked.index[0],
                "ranking": " > ".join(f"{m} ({cdf.loc[m, 'mean']:.3f})" for m in ranked.index),
                "coverage_avg": float(cdf["coverage"].mean()),
            })
    score_df = pd.DataFrame(score_rows)
    ci_strings_df = pd.DataFrame(ci_rows)
    coverage_df = pd.DataFrame(cov_rows)
    rank_df = pd.DataFrame(rank_rows).set_index("strategy")
    return score_df, ci_strings_df, coverage_df, rank_df


def cross_strategy_corr(family_strategies):
    """Spearman rho + Kendall tau between every pair of strategies' per-model means."""
    means = {label: df.groupby("model")["score"].mean() for label, df in family_strategies.items()}
    labels = list(means.keys())
    rho = pd.DataFrame(index=labels, columns=labels, dtype=float)
    tau = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            shared = means[a].index.intersection(means[b].index)
            if len(shared) >= 2:
                ra, _ = spearmanr(means[a].loc[shared], means[b].loc[shared])
                ta, _ = kendalltau(means[a].loc[shared], means[b].loc[shared])
                rho.loc[a, b] = ra
                tau.loc[a, b] = ta
    return rho, tau


def bootstrap_top1_agreement(family_strategies, n_boot=1000, seed=0):
    """
    For each bootstrap iter, resample items per (strategy, model), recompute
    per-model means, and check whether all strategies pick the same winner.
    Returns fraction of iterations with full agreement.
    """
    rng = np.random.default_rng(seed)
    labels = list(family_strategies.keys())
    if not labels:
        return float("nan"), {}

    # Pre-extract per-(strategy, model) score arrays for speed.
    per_sm = {}
    for label, df in family_strategies.items():
        per_sm[label] = {
            m: sub["score"].dropna().to_numpy()
            for m, sub in df.groupby("model")
        }
        per_sm[label] = {m: arr for m, arr in per_sm[label].items() if len(arr) > 0}

    agree_count = 0
    pairwise_flip = {(a, b): 0 for a in labels for b in labels if a < b}
    for _ in range(n_boot):
        winners = {}
        means_per_strategy = {}
        for label in labels:
            means = {}
            for m, scores in per_sm[label].items():
                idx = rng.integers(0, len(scores), size=len(scores))
                means[m] = scores[idx].mean()
            means_per_strategy[label] = means
            winners[label] = max(means, key=means.get)

        if len(set(winners.values())) == 1:
            agree_count += 1

        for a in labels:
            for b in labels:
                if a < b:
                    if winners[a] != winners[b]:
                        pairwise_flip[(a, b)] += 1

    top1_agreement_rate = agree_count / n_boot
    pairwise_flip_rate = {pair: c / n_boot for pair, c in pairwise_flip.items()}
    return top1_agreement_rate, pairwise_flip_rate


# ---------------------------------------------------------------------------
# 3) Output emitters
# ---------------------------------------------------------------------------
def _format_markdown_table(family_label, score_df, ci_strings_df, coverage_df, rank_df,
                           rho, top1_rate, pairwise_flip):
    """Plain-text/markdown summary suitable for the console and a .md sibling file."""
    lines = []
    lines.append(f"## {family_label}")
    lines.append("")
    lines.append("**Per-model scores** — each cell is `mean [95% bootstrap CI]`.")
    lines.append("Higher is better. Score is in [0, 1].")
    lines.append("")
    header = ["Model"] + list(score_df.columns)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for model in score_df.index:
        cells = [model]
        for col in score_df.columns:
            mean = score_df.loc[model, col]
            ci = ci_strings_df.loc[model, col]
            if pd.isna(mean):
                cells.append("—")
            else:
                cells.append(f"{mean:.3f} {ci}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("**Annotator coverage** — fraction of annotators who returned a valid")
    lines.append("(non-unsure, non-skipped) response per item, averaged across items.")
    lines.append("Low coverage means the protocol failed to elicit a confident judgment;")
    lines.append("scores from low-coverage protocols rest on fewer annotators per item.")
    lines.append("")
    cov_header = ["Model"] + list(coverage_df.columns)
    lines.append("| " + " | ".join(cov_header) + " |")
    lines.append("|" + "|".join(["---"] * len(cov_header)) + "|")
    for model in coverage_df.index:
        cells = [model] + [f"{coverage_df.loc[model, c]:.0%}" for c in coverage_df.columns]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("**Ranking under each strategy** (best -> worst).")
    lines.append("")
    for s, row in rank_df.iterrows():
        lines.append(f"- *{s}*: {row['ranking']}  (winner: **{row['winner']}**)")

    lines.append("")
    lines.append("**Cross-strategy ranking similarity** — Spearman rho on per-model means.")
    lines.append("rho = 1 means strategies produce identical rankings.")
    lines.append("rho < 0 means they disagree on at least one model-pair.")
    lines.append("")
    pair_lines = []
    for a in rho.index:
        for b in rho.columns:
            if a < b and not pd.isna(rho.loc[a, b]):
                pair_lines.append(f"- {a} vs {b}: rho = {rho.loc[a, b]:.3f}")
    lines.extend(pair_lines)

    lines.append("")
    lines.append(f"**Top-1 winner agreement across all strategies (bootstrap)**: "
                 f"`{top1_rate:.1%}` of resamples pick the same winning model under every strategy.")
    if pairwise_flip:
        lines.append("")
        lines.append("**Pairwise winner-flip rate (bootstrap)** — fraction of resamples")
        lines.append("in which the two strategies disagree on which model is best:")
        lines.append("")
        for (a, b), rate in pairwise_flip.items():
            tag = " — **significant disagreement**" if rate > 0.05 else ""
            lines.append(f"- {a} vs {b}: `{rate:.1%}`{tag}")
    return "\n".join(lines)


def emit_markdown(family_label, score_df, ci_strings_df, coverage_df, rank_df,
                  rho, top1_rate, pairwise_flip, out_path):
    text = _format_markdown_table(family_label, score_df, ci_strings_df, coverage_df,
                                  rank_df, rho, top1_rate, pairwise_flip)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"  wrote {out_path}")


def emit_latex(family_label, score_df, ci_strings_df, coverage_df, rho, top1_rate, out_path):
    """Compact LaTeX table: mean [CI] in cell, coverage row, ranking-agreement footer."""
    cells = score_df.applymap(lambda v: "" if pd.isna(v) else f"{v:.3f}") + " " + ci_strings_df
    cov_row = coverage_df.mean(axis=0).apply(lambda v: f"{v:.0%}")

    n_strats = len(score_df.columns)
    lines = []
    lines.append("% Auto-generated by analyze_strategy_ranking.py")
    lines.append(f"% Family: {family_label}")
    lines.append("% Each cell: mean score [95% bootstrap CI]; higher = better; range [0, 1].")
    lines.append("\\begin{tabular}{l" + "c" * n_strats + "}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(score_df.columns) + " \\\\")
    lines.append("\\midrule")
    for model, row in cells.iterrows():
        lines.append(f"{model} & " + " & ".join(row.values) + " \\\\")
    lines.append("\\midrule")
    lines.append("Annotator coverage & " + " & ".join(cov_row.values) + " \\\\")

    # Compact ranking-agreement footer.
    rho_strs = []
    for a in rho.index:
        for b in rho.columns:
            if a < b and not pd.isna(rho.loc[a, b]):
                rho_strs.append(f"$\\rho_{{\\textit{{{a}}}, \\textit{{{b}}}}} = {rho.loc[a, b]:.2f}$")
    rho_inline = "; ".join(rho_strs) if rho_strs else "n/a"
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{1 + n_strats}}}{{p{{0.95\\linewidth}}}}{{Spearman $\\rho$ "
                 f"between strategy rankings: {rho_inline}. Bootstrap top-1 winner agreement: "
                 f"{top1_rate:.0%}.}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out_path}")


def emit_grouped_bars(families, out_path):
    """One subplot per family. x=models, hue=strategy, error bars = bootstrap CIs."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass

    fig, axes = plt.subplots(len(families), 1, figsize=(10, 3.0 * len(families)))
    if len(families) == 1:
        axes = [axes]

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for ax, (family_label, family_strats) in zip(axes, families.items()):
        if not family_strats:
            ax.set_visible(False)
            continue

        all_models = sorted({m for df in family_strats.values() for m in df["model"].unique()})
        strategy_labels = list(family_strats.keys())
        n_strat = len(strategy_labels)
        bar_w = 0.8 / max(n_strat, 1)
        x_base = np.arange(len(all_models))

        for i, label in enumerate(strategy_labels):
            df = family_strats[label]
            ci_df = per_model_ci(df, n_boot=1000, seed=0)
            means, lows, highs = [], [], []
            for m in all_models:
                if m in ci_df.index:
                    means.append(ci_df.loc[m, "mean"])
                    lows.append(ci_df.loc[m, "mean"] - ci_df.loc[m, "lo"])
                    highs.append(ci_df.loc[m, "hi"] - ci_df.loc[m, "mean"])
                else:
                    means.append(np.nan); lows.append(0); highs.append(0)

            offset = (i - (n_strat - 1) / 2) * bar_w
            ax.bar(x_base + offset, means, bar_w, label=label,
                   color=palette[i % len(palette)],
                   yerr=[lows, highs], capsize=3, alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x_base)
        ax.set_xticklabels(all_models, fontsize=12)
        ax.set_ylabel("Score (higher = better)", fontsize=11)
        ax.set_title(family_label, fontsize=13, loc='left')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc='lower right', ncol=n_strat)
        ax.grid(True, axis='y', alpha=0.4)
        ax.grid(False, axis='x')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f"  wrote {out_path} (and .png)")


# ---------------------------------------------------------------------------
# 4) Consolidated alpha + bootstrap CI summary across all protocols
# ---------------------------------------------------------------------------
def _alpha_for_pivot(pivot_df, level, n_boot, ci, seed):
    """Wrap bootstrap_alpha; returns (alpha, lo, hi, n_items, n_annotators)."""
    if pivot_df.empty:
        return np.nan, np.nan, np.nan, 0, 0
    alpha, lo, hi = bootstrap_alpha(pivot_df, level=level, n_boot=n_boot, ci=ci, seed=seed)
    return alpha, lo, hi, int(pivot_df.shape[0]), int(pivot_df.shape[1])


def compute_alpha_summary(n_boot=1000, ci=0.95, seed=0):
    """
    Compute Krippendorff's alpha with 95% bootstrap CI for every annotation
    strategy across all three families. Bootstrap resamples items (rows of the
    annotator-pivot) with replacement.

    Returns DataFrame with columns:
        family, strategy, level, alpha, lo, hi, n_items, n_annotators
    Strategies with no data on disk are silently skipped.
    """
    rows = []

    # ---- Anchor family ----
    anchor_specs = [
        ("Anchor", "No-Anchor BQA", "no_anchor_bqa", "nominal"),
        ("Anchor", "Anchor BQA",    "anchor_bqa",    "nominal"),
        ("Anchor", "Anchor Likert", "anchor_likert", "ordinal"),
    ]
    for family, label, subdir, level in anchor_specs:
        path = ANNOTATIONS_DIR / subdir / "vqa"
        if not path.exists():
            continue
        df = process_anchor_annotations(path, level=level)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["q_annot"] = df["q_annot"].replace(-1, np.nan)
        pivot = df.pivot_table(index=["model", "img_id", "q_id"],
                               columns="annotator", values="q_annot")
        alpha, lo, hi, n_items, n_ann = _alpha_for_pivot(pivot, level, n_boot, ci, seed)
        rows.append({"family": family, "strategy": label, "level": level,
                     "alpha": alpha, "lo": lo, "hi": hi,
                     "n_items": n_items, "n_annotators": n_ann})

    # ---- Text family ----
    likert_path = ANNOTATIONS_DIR / "text_likert" / "vqa"
    if likert_path.exists():
        df = process_text_likert(likert_path)
        if df is not None and not df.empty:
            d = df.copy()
            d["q_annot"] = d["q_annot"].replace(-1, np.nan)
            pivot = d.pivot_table(index=["model", "img_id", "q_id"],
                                  columns="annotator", values="q_annot")
            alpha, lo, hi, n_items, n_ann = _alpha_for_pivot(pivot, "ordinal", n_boot, ci, seed)
            rows.append({"family": "Text", "strategy": "Text Likert", "level": "ordinal",
                         "alpha": alpha, "lo": lo, "hi": hi,
                         "n_items": n_items, "n_annotators": n_ann})

            # Artificial BQA: threshold Likert > 3 to {0, 1} (matches the
            # existing analysis in analyze_text_based.py).
            d_bqa = df.copy()
            d_bqa["q_annot"] = np.where(d_bqa["q_annot"] > 3, 1.0, 0.0)
            pivot_bqa = d_bqa.pivot_table(index=["model", "img_id", "q_id"],
                                          columns="annotator", values="q_annot")
            alpha, lo, hi, n_items, n_ann = _alpha_for_pivot(pivot_bqa, "nominal", n_boot, ci, seed)
            rows.append({"family": "Text", "strategy": "Text BQA (Likert > 3)",
                         "level": "nominal",
                         "alpha": alpha, "lo": lo, "hi": hi,
                         "n_items": n_items, "n_annotators": n_ann})

    per_word_path = ANNOTATIONS_DIR / "text_per_word" / "vqa"
    if per_word_path.exists():
        img_df, word_df = process_text_per_word(per_word_path)
        if img_df is not None and not img_df.empty:
            pivot = img_df.pivot_table(index=["model", "img_id", "q_id"],
                                       columns="annotator", values="q_annot")
            alpha, lo, hi, n_items, n_ann = _alpha_for_pivot(pivot, "interval", n_boot, ci, seed)
            rows.append({"family": "Text", "strategy": "Per-Word (image accuracy)",
                         "level": "interval",
                         "alpha": alpha, "lo": lo, "hi": hi,
                         "n_items": n_items, "n_annotators": n_ann})
        if word_df is not None and not word_df.empty:
            pivot = word_df.pivot_table(index=["model", "img_id", "q_id"],
                                        columns="annotator", values="q_annot")
            alpha, lo, hi, n_items, n_ann = _alpha_for_pivot(pivot, "nominal", n_boot, ci, seed)
            rows.append({"family": "Text", "strategy": "Per-Word (binary)",
                         "level": "nominal",
                         "alpha": alpha, "lo": lo, "hi": hi,
                         "n_items": n_items, "n_annotators": n_ann})

    # ---- Artifacts family ----
    likert_path = ANNOTATIONS_DIR / "artifacts_likert" / "vqa"
    if likert_path.exists():
        df = process_artifacts_likert(likert_path)
        if df is not None and not df.empty:
            d = df.copy()
            d["q_annot"] = d["q_annot"].replace(-1, np.nan)
            pivot = d.pivot_table(index=["model", "img_id"],
                                  columns="annotator", values="q_annot")
            alpha, lo, hi, n_items, n_ann = _alpha_for_pivot(pivot, "ordinal", n_boot, ci, seed)
            rows.append({"family": "Artifacts", "strategy": "Aesthetics Likert",
                         "level": "ordinal",
                         "alpha": alpha, "lo": lo, "hi": hi,
                         "n_items": n_items, "n_annotators": n_ann})

    brush_df = process_artifacts_brush(verbose=False)
    if brush_df is not None and not brush_df.empty:
        pivot = brush_df.pivot_table(index=["model", "img_id"],
                                     columns="annotator", values="artifacts_rate")
        alpha, lo, hi, n_items, n_ann = _alpha_for_pivot(pivot, "interval", n_boot, ci, seed)
        rows.append({"family": "Artifacts", "strategy": "Brush Mask (artifact rate)",
                     "level": "interval",
                     "alpha": alpha, "lo": lo, "hi": hi,
                     "n_items": n_items, "n_annotators": n_ann})

    return pd.DataFrame(rows)


def emit_alpha_summary(alpha_df, md_path, tex_path):
    # ----- Markdown -----
    md_lines = [
        "# Inter-annotator Agreement Summary",
        "",
        "Krippendorff's alpha with 95% bootstrap CI for every annotation strategy",
        "across all skill families. CIs are computed by resampling items with",
        "replacement (1000 iterations, seed=0).",
        "",
        "| Family | Strategy | Level | Krippendorff's alpha [95% CI] | n items | n annotators |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in alpha_df.iterrows():
        if pd.isna(r["alpha"]):
            cell = "—"
        else:
            cell = f"{r['alpha']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
        md_lines.append(
            f"| {r['family']} | {r['strategy']} | {r['level']} | {cell} | "
            f"{r['n_items']} | {r['n_annotators']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  wrote {md_path}")

    # ----- LaTeX -----
    tex_lines = [
        "% Auto-generated by analyze_strategy_ranking.py (compute_alpha_summary).",
        "% Krippendorff's alpha with 95% bootstrap CI for every annotation strategy.",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Family & Strategy & Level & Krippendorff's $\\alpha$ [95\\% CI] \\\\",
        "\\midrule",
    ]
    last_family = None
    for _, r in alpha_df.iterrows():
        if r["family"] != last_family and last_family is not None:
            tex_lines.append("\\addlinespace[2pt]")
        family_cell = r["family"] if r["family"] != last_family else ""
        last_family = r["family"]
        if pd.isna(r["alpha"]):
            cell = "---"
        else:
            cell = f"${r['alpha']:.3f}\\,[{r['lo']:.3f},\\,{r['hi']:.3f}]$"
        tex_lines.append(f"{family_cell} & {r['strategy']} & {r['level']} & {cell} \\\\")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"  wrote {tex_path}")


# ---------------------------------------------------------------------------
# 5) Main
# ---------------------------------------------------------------------------
def _emit_family(family_label, family_strats, file_stem, n_boot, seed):
    """Compute everything for one family and write .tex + .md outputs.

    Also returns the family_strats dict (for the figure)."""
    print()
    print("=" * 60)
    print(f"FAMILY: {family_label}")
    print("=" * 60)
    if not family_strats:
        print("  (no strategies loaded, skipping)")
        return None

    score_df, ci_strings_df, coverage_df, rank_df = ranking_table(
        family_strats, n_boot=n_boot, seed=seed)
    rho, _tau = cross_strategy_corr(family_strats)
    top1_rate, pairwise_flip = bootstrap_top1_agreement(family_strats, n_boot=n_boot, seed=seed)

    md_text = _format_markdown_table(family_label, score_df, ci_strings_df, coverage_df,
                                     rank_df, rho, top1_rate, pairwise_flip)
    print()
    print(md_text)

    emit_markdown(family_label, score_df, ci_strings_df, coverage_df, rank_df,
                  rho, top1_rate, pairwise_flip,
                  OUTPUT_DIR / f"{file_stem}.md")
    emit_latex(family_label, score_df, ci_strings_df, coverage_df, rho, top1_rate,
               OUTPUT_DIR / f"{file_stem}.tex")
    return family_strats


def run(seed=0, n_boot=1000):
    np.random.seed(seed)

    print("=" * 60)
    print("LOADING ANCHOR FAMILY (unsure counted as 0; same item set across strategies)")
    print("=" * 60)
    anchor_zero = load_anchor_family(unsure_value=0.0)
    print("  strategies:", list(anchor_zero.keys()))

    print()
    print("=" * 60)
    print("LOADING ANCHOR FAMILY (unsure dropped; sensitivity analysis)")
    print("=" * 60)
    anchor_drop = load_anchor_family(unsure_value=np.nan)
    print("  strategies:", list(anchor_drop.keys()))

    print()
    print("=" * 60)
    print("LOADING TEXT FAMILY")
    print("=" * 60)
    text = load_text_family()
    print("  strategies:", list(text.keys()))

    print()
    print("=" * 60)
    print("LOADING ARTIFACTS FAMILY")
    print("=" * 60)
    artifacts = load_artifacts_family()
    print("  strategies:", list(artifacts.keys()))

    # Emit each family. The "unsure -> 0" treatment is the headline result.
    headline_anchor = _emit_family(
        "Anchor-Based Skills (unsure = 0)", anchor_zero, "ranking_anchor", n_boot, seed)
    _emit_family(
        "Anchor-Based Skills (unsure dropped — sensitivity check)",
        anchor_drop, "ranking_anchor_drop_unsure", n_boot, seed)
    headline_text = _emit_family(
        "Text Rendering", text, "ranking_text", n_boot, seed)
    headline_artifacts = _emit_family(
        "Visual Artifacts / Aesthetics", artifacts, "ranking_artifacts", n_boot, seed)

    headline_families = {}
    if headline_anchor: headline_families["Anchor-Based Skills"] = headline_anchor
    if headline_text: headline_families["Text Rendering"] = headline_text
    if headline_artifacts: headline_families["Visual Artifacts / Aesthetics"] = headline_artifacts
    emit_grouped_bars(headline_families, OUTPUT_DIR / "ranking_grouped_bars.pdf")

    # Consolidated alpha + bootstrap CI table across all protocols.
    print()
    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT SUMMARY (all protocols)")
    print("=" * 60)
    alpha_df = compute_alpha_summary(n_boot=n_boot, ci=0.95, seed=seed)
    if not alpha_df.empty:
        display = alpha_df.copy()
        display["alpha [95% CI]"] = display.apply(
            lambda r: ("nan" if pd.isna(r["alpha"]) else
                       f"{r['alpha']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"),
            axis=1,
        )
        print(display[["family", "strategy", "level", "alpha [95% CI]",
                       "n_items", "n_annotators"]].to_string(index=False))
    emit_alpha_summary(alpha_df,
                       OUTPUT_DIR / "alpha_summary.md",
                       OUTPUT_DIR / "alpha_summary.tex")

    print("\nDone.")


if __name__ == "__main__":
    run()
