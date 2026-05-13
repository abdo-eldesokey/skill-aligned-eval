import json
from typing import List, Dict, Optional
from pydantic import BaseModel, ConfigDict, field_validator

# -------------------------
# Skill Taxonomy v6.0
# -------------------------
SKILL_TAXONOMY: Dict[str, Optional[List[str]]] = {
    "entities": ["singular", "count_exact", "count_plural", "uncountable"],
    "attributes": ["color", "texture", "material", "shape", "absolute_scale"],
    "relative_comparison": ["scale", "tone", "distance", "count", "other"],
    "action": ["pose", "standard", "unusual"],
    "spatial_arrangement": None,
    "environment_scene": ["landmark", "general"],
    "style": ["artistic_style", "visual_medium"],
    "lighting": None,
    "weather": None,
    "view": None,
    "text_rendering": ["rendering_accuracy", "style", "numerical", "position"],
    "mood_feeling": None,
    "named_entities": ["character", "vehicle", "product", "artwork"],
    "language_complexity": ["negation", "color_stroop"],
    "time": ["time_of_day", "season", "year_era"],
    "camera": None,
}


class Annotation(BaseModel):
    """Single annotation item with uid, skill, subskill, phrase, question, node_type, and dependencies."""

    model_config = ConfigDict(extra="forbid", strict=True)
    uid: str
    skill: str
    subskill: str  # Empty string if no subskill
    phrase: str
    question: str
    node_type: str  # "presence", "property", or "relation"
    depends_on: List[str]  # List of parent UIDs

    @field_validator("uid")
    def validate_uid(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError(f"UID must be a numeric string, got: '{v}'")
        return v

    @field_validator("skill")
    def validate_skill(cls, v: str) -> str:
        if v not in SKILL_TAXONOMY:
            raise ValueError(f"Unknown skill: '{v}'")
        return v

    @field_validator("subskill")
    def validate_subskill(cls, v: str, info) -> str:
        skill = info.data.get("skill")
        if not skill:
            return v

        allowed_subskills = SKILL_TAXONOMY.get(skill)

        # If subskill is provided (non-empty)
        if v:
            if not allowed_subskills:
                raise ValueError(f"Skill '{skill}' does not have subskills, but '{v}' was provided.")
            if v not in allowed_subskills:
                raise ValueError(f"Invalid subskill '{v}' for skill '{skill}'. Allowed: {allowed_subskills}")

        return v

    # @field_validator("question")
    # def validate_question(cls, v: str) -> str:
    #     # Check not empty
    #     if not v or not v.strip():
    #         raise ValueError("Question cannot be empty.")

    #     # Check word count (≤ 15 words)
    #     word_count = len(v.split())
    #     if word_count > 15:
    #         raise ValueError(f"Question has {word_count} words, exceeds limit of 15 words: '{v}'")

    #     # Check if it's a binary question (should end with '?')
    #     if not v.strip().endswith("?"):
    #         raise ValueError(f"Question must end with '?': '{v}'")

    #     return v

    @field_validator("phrase")
    def validate_phrase(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Phrase cannot be empty.")
        return v

    @field_validator("node_type")
    def validate_node_type(cls, v: str) -> str:
        allowed_types = ["presence", "property", "relation"]
        if v not in allowed_types:
            raise ValueError(f"Invalid node_type '{v}'. Allowed: {allowed_types}")
        return v


class TaggedPromptWithVQA(BaseModel):
    """New format with annotations array."""

    model_config = ConfigDict(extra="forbid", strict=True, title="tagged_prompt_with_vqa")
    annotations: List[Annotation]

    @field_validator("annotations")
    def validate_annotations(cls, v: List[Annotation]) -> List[Annotation]:
        if not v:
            raise ValueError("Annotations list cannot be empty.")

        # Check for unique UIDs
        uids = [ann.uid for ann in v]
        if len(uids) != len(set(uids)):
            raise ValueError("All UIDs must be unique within the annotations list.")

        return v

    def pretty(self) -> str:
        headers = ["UID", "Skill", "Sub-Skill", "Phrase", "Question", "Node Type", "Depends On"]
        rows = []
        for ann in self.annotations:
            depends_str = ", ".join(ann.depends_on) if ann.depends_on else ""
            rows.append(
                [ann.uid, ann.skill, ann.subskill or "", f'"{ann.phrase}"', ann.question, ann.node_type, depends_str]
            )

        if not rows:
            return "<No annotations found>"

        # compute column widths
        col_widths = [len(h) for h in headers]
        for r in rows:
            for j, cell in enumerate(r):
                col_widths[j] = max(col_widths[j], len(str(cell)))

        def fmt_row(cols):
            return " | ".join(str(c).ljust(col_widths[j]) for j, c in enumerate(cols))

        lines = [fmt_row(headers), fmt_row(["-" * w for w in col_widths])]
        lines += [fmt_row(r) for r in rows]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()

    def save_to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)
