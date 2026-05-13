"""
Computes Inter-LLM Agreement across different skills using Krippendorff's Alpha.
The results are formatted as a LaTeX table.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from config import BASE_DIR, ASSETS_DIR
from analysis.analyze_llm_human_correlation import load_prompt_skill_mapping, SKILL_ORDER
from utils.metrics import krippendorff_alpha

LLM_VQA_DIR = ASSETS_DIR / "annotations" / "full_evaluation_llm" / "vqa"

# List of models to completely exclude from agreement analysis
MODELS_TO_EXCLUDE = ["hf:google/gemma-3-4b-it","hf:google/gemma-3-12b-it"]


def extract_llm_scores():
    prompt_mapping, eval_prompt_ids, models = load_prompt_skill_mapping()
    
    # Store data as: skill_data[skill][(img_id, uid)][llm_name] = normalized_score
    skill_data = {}
    
    llms_seen = set()
    
    for model in models:
        llm_vqa_dir = LLM_VQA_DIR / model
        if not llm_vqa_dir.exists():
            continue
            
        for json_file in sorted(llm_vqa_dir.glob("*.json")):
            img_id = int(json_file.stem)
            
            if img_id not in eval_prompt_ids:
                continue
                
            uid_map = prompt_mapping.get(img_id)
            if not uid_map:
                continue
                
            with open(json_file, "r", encoding="utf-8") as f:
                llm_data_full = json.load(f)
                
            if not llm_data_full:
                continue
                
            for llm_name, llm_data in llm_data_full.items():
                short = llm_name.split(":")[-1].split("/")[-1] if "hf" in llm_name else llm_name.split(":")[-1]
                if short in MODELS_TO_EXCLUDE or llm_name in MODELS_TO_EXCLUDE:
                    continue
                    
                llms_seen.add(llm_name)
                
                # We need to map back the scores to specific skills
                # Since we want item-level agreement, we need to handle uids correctly.
                # However, score_annotator_responses returns an aggregated list per skill.
                # We should extract per-uid scores instead for item-level agreement.
                
                for uid, mapping in uid_map.items():
                    skill = mapping["skill"]
                    ans = llm_data.get(uid)
                    
                    if ans is None:
                        continue
                        
                    score = None
                    
                    # 1. Text rendering
                    if mapping["is_text_rendering_accuracy"]:
                        if isinstance(ans, dict):
                            # compute_text_per_word_stats expects specific format
                            from apps.annotation_stats_app import compute_text_per_word_stats
                            single_annot = {"annotator": {uid: ans}}
                            stats = compute_text_per_word_stats(single_annot)
                            if "annotator" in stats:
                                accuracy = stats["annotator"].get("word_accuracy", None)
                                if accuracy is not None:
                                    score = accuracy / 100.0
                        elif isinstance(ans, str) and ans.lower().strip() == "unsure":
                            pass # Keep None
                    
                    # 2. Binary QA
                    elif isinstance(ans, str):
                        ans_lower = ans.lower().strip()
                        if ans_lower == "yes":
                            score = 1.0
                        elif ans_lower == "no":
                            score = 0.0
                    
                    # 3. Likert (0-5)
                    elif isinstance(ans, (int, float)):
                        score = ans / 5.0
                        
                    if score is not None:
                        if skill not in skill_data:
                            skill_data[skill] = {}
                        item_key = f"{img_id}_{uid}"
                        if item_key not in skill_data[skill]:
                            skill_data[skill][item_key] = {}
                        skill_data[skill][item_key][llm_name] = score
                        
                # 4. Aesthetics (Single score per image, not per UID)
                if "_aesthetics_rating" in llm_data:
                    aes = llm_data["_aesthetics_rating"]
                    if isinstance(aes, (int, float)):
                        score = aes / 5.0
                        if "aesthetic_quality" not in skill_data:
                            skill_data["aesthetic_quality"] = {}
                        item_key = f"{img_id}_aesthetics"
                        if item_key not in skill_data["aesthetic_quality"]:
                            skill_data["aesthetic_quality"][item_key] = {}
                        skill_data["aesthetic_quality"][item_key][llm_name] = score
                        
    return skill_data, list(llms_seen)


def compute_inter_llm_agreement(skill_data, all_llms):
    records = []
    
    # Pre-sort skills based on SKILL_ORDER where possible
    skills_present = list(skill_data.keys())
    sorted_skills = [s for s in SKILL_ORDER if s in skills_present and not s.startswith("visual_artifacts") and s != "Average"]
    other_skills = [s for s in skills_present if s not in sorted_skills and not s.startswith("visual_artifacts") and s != "Average"]
    
    for skill in sorted_skills + other_skills:
        items_dict = skill_data[skill]
        items_list = list(items_dict.keys())
        
        n_items = len(items_list)
        n_annotators = len(all_llms)
        
        if n_items == 0:
            continue
            
        data_matrix = np.full((n_items, n_annotators), np.nan)
        
        for i, item_key in enumerate(items_list):
            for j, llm_name in enumerate(all_llms):
                if llm_name in items_dict[item_key]:
                    data_matrix[i, j] = items_dict[item_key][llm_name]
                    
        # Compute alpha
        alpha = krippendorff_alpha(data_matrix, level="interval")
        
        records.append({
            "Skill": skill.replace("_", " ").title(),
            "Items": n_items,
            "Krippendorff's Alpha": alpha
        })
        
    df = pd.DataFrame(records)
    return df

def run():
    print("Extracting LLM predictions...")
    skill_data, llms = extract_llm_scores()
    
    print(f"Found {len(llms)} distinct LLM models: {llms}")
    
    if not skill_data:
        print("No valid data found.")
        return
        
    print("Computing Krippendorff's Alpha for each skill...")
    df = compute_inter_llm_agreement(skill_data, llms)
    
    # Compute Average Alpha
    valid_alphas = df["Krippendorff's Alpha"].dropna()
    avg_alpha = valid_alphas.mean() if not valid_alphas.empty else np.nan
    
    # Add an Average row to the dataframe
    avg_row = pd.DataFrame([{
        "Skill": "Average",
        "Items": df["Items"].sum(),
        "Krippendorff's Alpha": avg_alpha
    }])
    df = pd.concat([df, avg_row], ignore_index=True)
    
    print("\n--- Inter-LLM Agreement ---")
    print(df.to_string(index=False))
    
    # Format for LaTeX
    # Rename columns to be LaTeX friendly if needed
    latex_df = df.copy()
    latex_df["Krippendorff's Alpha"] = latex_df["Krippendorff's Alpha"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
    
    latex_str = latex_df.to_latex(index=False, column_format="lcc", caption="Inter-LLM Agreement (Krippendorff's Alpha) per Skill", label="tab:inter_llm_agreement")
    
    out_latex_file = BASE_DIR / "inter_llm_agreement.tex"
    with open(out_latex_file, "w", encoding="utf-8") as f:
        f.write(latex_str)
        
    print(f"\nLaTeX table saved to {out_latex_file}")

if __name__ == "__main__":
    run()
