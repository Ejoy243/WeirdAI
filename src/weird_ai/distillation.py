
"""
Distillation dataset utilities for Weird AI.

Chapter 8 introduces model distillation:

    teacher model -> synthetic training examples -> student model

For Weird AI, this means preparing teacher-generated parody lyrics and teacher
notes as reusable supervised training data. This module does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import random
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class TeacherExample:
    prompt: str
    teacher_lyrics: str
    teacher_notes: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationRecord:
    prompt: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_teacher_example(example: dict) -> bool:
    """
    Validate that a raw teacher example contains usable fields.

    Required fields:
        prompt, teacher_lyrics, teacher_notes, quality_score
    """
    # TODO:
    # 1. Check that all required fields are present.
    # 2. Check that prompt, teacher_lyrics, and teacher_notes are non-empty strings.
    # 3. Check that quality_score can be converted to float.
    # 4. Return True only if all checks pass.

    if not isinstance(example, dict):
        return False

    required_fields = ["prompt", "teacher_lyrics", "teacher_notes", "quality_score"]
    for field in required_fields:
        if field not in example:
            return False

    if not isinstance(example["prompt"], str) or not example["prompt"].strip():
        return False
    if not isinstance(example["teacher_lyrics"], str) or not example["teacher_lyrics"].strip():
        return False
    if not isinstance(example["teacher_notes"], str) or not example["teacher_notes"].strip():
        return False
    if not isinstance(example["quality_score"], (int, float)):
        return False

    return True


def teacher_example_from_dict(example: dict) -> TeacherExample:
    """Convert a valid dictionary into a TeacherExample."""
    # TODO:
    # 1. Validate the example.
    # 2. Raise ValueError if invalid.
    # 3. Convert quality_score to float.
    # 4. Put any extra keys into metadata.
    # 5. Return a TeacherExample.
    if not validate_teacher_example(example):
        raise ValueError("Invalid teacher example.")

    quality_score = float(example["quality_score"])
    metadata = {k: v for k, v in example.items() if k not in ["prompt", "teacher_lyrics", "teacher_notes", "quality_score"]}

    return TeacherExample(
        prompt=example["prompt"].strip(),
        teacher_lyrics=example["teacher_lyrics"].strip(),
        teacher_notes=example["teacher_notes"].strip(),
        quality_score=quality_score,
        metadata=metadata
    )


def format_teacher_response(
    teacher_lyrics: str,
    teacher_notes: str,
    *,
    include_thinking: bool = True,
) -> str:
    """Format teacher notes and lyrics as a distillation target."""
    # TODO:
    # 1. Strip teacher_lyrics and teacher_notes.
    # 2. If include_thinking=True, return notes inside <think>...</think> tags,
    #    followed by a blank line and the lyrics.
    # 3. If include_thinking=False, return only the lyrics.
    # 4. Avoid extra leading/trailing whitespace.
    if include_thinking:
        return f"<think>{teacher_notes.strip()}</think>\n\n{teacher_lyrics.strip()}"
    else:
        return teacher_lyrics.strip()


def format_training_prompt(prompt: str) -> str:
    """Format a Weird AI prompt for distillation training."""
    # TODO:
    # 1. Strip the prompt.
    # 2. Return a consistent format such as:
    #    "User: Write a parody...\\nAssistant:"
    return f"User: {prompt.strip()}\nAssistant:"

def build_full_text(prompt: str, target_text: str) -> str:
    """Combine formatted prompt and target text."""
    # TODO:
    # 1. Use format_training_prompt(prompt).
    # 2. Append target_text after the assistant marker.
    # 3. Return the full training string.
    formatted_prompt = format_training_prompt(prompt)
    return f"{formatted_prompt} {target_text}"

def tokenize_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using an object with encode(text) -> list[int]."""
    # TODO:
    # 1. Call tokenizer.encode(text).
    # 2. Validate that the result is a list of integers.
    # 3. Return the token IDs.
    token_ids = tokenizer.encode(text)
    if not isinstance(token_ids, list) or not all(isinstance(token, int) for token in token_ids):
        raise ValueError("Tokenized output is not a list of integers.")
    return token_ids


def create_labels(
    input_ids: list[int],
    prompt_length: int,
    *,
    mask_prompt: bool = True,
    ignore_index: int = -100,
) -> list[int]:
    """
    Create training labels.

    If mask_prompt=True, prompt token labels should be replaced with ignore_index.
    """
    # TODO:
    # 1. Copy input_ids into labels.
    # 2. If mask_prompt=True, replace labels before prompt_length with ignore_index.
    # 3. Return labels.
    labels = input_ids.copy()
    if mask_prompt:
        for i in range(prompt_length):
            labels[i] = ignore_index
    return labels


def build_distillation_record(
    example: TeacherExample,
    tokenizer: TokenizerLike,
    *,
    include_thinking: bool = True,
    mask_prompt: bool = True,
) -> DistillationRecord:
    """Build one processed distillation record."""
    # TODO:
    # 1. Format the teacher response.
    # 2. Format the prompt.
    # 3. Build the full text.
    # 4. Tokenize the prompt and full text.
    # 5. Create labels.
    # 6. Return a DistillationRecord.
    target_text = format_teacher_response(
        example.teacher_lyrics,
        example.teacher_notes,
        include_thinking=include_thinking,
    )
    prompt_text = format_training_prompt(example.prompt)
    full_text = build_full_text(example.prompt, target_text)   # pass the RAW prompt

    prompt_ids = tokenize_text(tokenizer, prompt_text)
    input_ids = tokenize_text(tokenizer, full_text)
    prompt_length = len(prompt_ids)

    labels = create_labels(input_ids, prompt_length, mask_prompt=mask_prompt)

    return DistillationRecord(
        prompt=example.prompt,
        target_text=target_text,
        full_text=full_text,
        input_ids=input_ids,
        labels=labels,
        prompt_length=prompt_length,
        metadata=dict(example.metadata),
    )

def filter_teacher_examples(
    examples: list[TeacherExample],
    *,
    min_quality_score: float = 0.75,
    min_lyric_lines: int = 4,
) -> list[TeacherExample]:
    """Keep only high-quality teacher examples."""
    # TODO:
    # 1. Keep examples with quality_score >= min_quality_score.
    # 2. Keep examples with at least min_lyric_lines non-empty lyric lines.
    # 3. Return the filtered list.
    filtered_examples = []
    for example in examples:
        if example.quality_score < min_quality_score:
            continue
        lyrics = example.teacher_lyrics.replace("\\n", "\n")   # turn typed-out "\n" into real line breaks
        lyric_lines = [line for line in lyrics.splitlines() if line.strip()]   
        if len(lyric_lines) < min_lyric_lines:
            continue
        filtered_examples.append(example)
    return filtered_examples


def split_records(
    records: list[DistillationRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DistillationRecord], list[DistillationRecord]]:
    """Split records into train and validation sets."""
    # TODO:
    # 1. Validate validation_fraction is between 0 and 1.
    # 2. Shuffle a copy of records using seed.
    # 3. Split into train and validation records.
    # 4. Return both lists.
    if not (0 <= validation_fraction <= 1):
        raise ValueError("validation_fraction must be between 0 and 1.")
    
    random.seed(seed)
    shuffled_records = records.copy()
    random.shuffle(shuffled_records)
    split_index = int(len(shuffled_records) * (1 - validation_fraction))
    train_records = shuffled_records[:split_index]
    validation_records = shuffled_records[split_index:]
    return train_records, validation_records

def record_to_json_dict(record: DistillationRecord) -> dict:
    """Convert a DistillationRecord into a JSON-serializable dictionary."""
    # TODO:
    # 1. Convert the dataclass to a dictionary.
    # 2. Return the dictionary.
    return asdict(record)

def export_jsonl(records: list[DistillationRecord], path: str | Path) -> None:
    """Export records to a JSONL file."""
    # TODO:
    # 1. Open path for writing with UTF-8 encoding.
    # 2. Write one JSON object per line.
    # 3. Use record_to_json_dict for each record.
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record_to_json_dict(record), f)
            f.write("\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into dictionaries."""
    # TODO:
    # 1. Open path with UTF-8 encoding.
    # 2. Read each non-empty line.
    # 3. Parse JSON and append to a list.
    # 4. Return the list.
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_distillation_records(records: list[DistillationRecord]) -> str:
    """Create a readable summary of distillation records."""
    # TODO:
    # 1. Handle empty records.
    # 2. Include number of records.
    # 3. Include average input length.
    # 4. Include average prompt length.
    # 5. Include average target length.
    if not records:
        return "No records to summarize."

    avg_input_length = sum(len(record.input_ids) for record in records) / len(records)

    total_prompt_length = sum(len(record.prompt) for record in records)
    avg_prompt_length = total_prompt_length / len(records)

    total_target_length = sum(len(record.input_ids) for record in records)
    avg_target_length = total_target_length / len(records)

    return (
        f"Records: {len(records)}\n"
        f"Average input length: {avg_input_length:.1f} tokens\n"
        f"Average prompt length: {avg_prompt_length:.1f} tokens\n"
        f"Average target length: {avg_target_length:.1f} tokens"
    )