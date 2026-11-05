"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)


def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    # TODO:
    # 1. Read score_key from the evaluation dictionary.
    # 2. Convert it to a float.
    # 3. If the key is missing or cannot be converted, return 0.0.
    try:
        return float(evaluation.get(score_key, 0.0))
    except (ValueError, TypeError):
        return 0.0


def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    # TODO:
    # 1. Include the original prompt or topic.
    # 2. Include the current lyrics.
    # 3. Include the current overall score.
    # 4. Include any available evaluation details, such as rhyme, syllables, and structure.
    # 5. Include focus items, such as stronger rhymes or more consistent syllables.
    # 6. Include preserve items, such as topic, tone, and approximate length.
    # 7. Ask the model to return only the revised lyrics.
    #
    # Keep the prompt specific. "Make it better" is too vague.

    prompt_lines = [
        f"Original prompt: {original_prompt}",
        f"Current lyrics:\n{current_text}",
        f"Current overall score: {evaluation.get(score_key, 0.0)}",
        f"Evaluation details: {evaluation}",
        f"Focus on: {', '.join(focus) if focus else 'None'}",
        f"Preserve: {', '.join(preserve) if preserve else 'None'}",
        "Please revise the lyrics to improve them while keeping the focus and preserve items in mind.",
        "Return only the revised lyrics without any additional commentary or explanation."
    ]

    return "\n".join(prompt_lines)


def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    # TODO:
    # 1. Evaluate current_text.
    # 2. Build a refinement prompt using build_refinement_prompt.
    # 3. Generate revised text with generation_func.
    # 4. Evaluate the revised text.
    # 5. Compare the old score and revised score.
    # 6. Return a RefinementStep.
    #
    # This function should not decide whether to continue the full loop.
    # It only performs one attempt.

    current_evaluation = evaluation_func(current_text)
    current_score = get_score(current_evaluation, score_key=score_key)
    revised_text = generation_func(
        build_refinement_prompt(
            original_prompt,
            current_text,
            current_evaluation,
            focus=focus,
            preserve=preserve,
            score_key=score_key,
        ),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    revised_evaluation = evaluation_func(revised_text)
    revised_score = get_score(revised_evaluation, score_key=score_key)
    return RefinementStep(
        iteration=0,  # This will be set by the caller in iterative_refinement
        prompt=original_prompt,
        previous_text=current_text,
        revised_text=revised_text,
        previous_score=current_score,
        revised_score=revised_score,
        accepted=revised_score > current_score,
        evaluation=revised_evaluation,
    )

def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    # TODO:
    # 1. Compare the current score and revised score.
    # 2. If require_improvement is True, accept only strictly higher scores.
    # 3. If require_improvement is False, accept equal scores as well.
    # 4. Return the selected text, selected evaluation, and whether the revision was accepted.

    current_score = get_score(current_evaluation, score_key=score_key)
    revised_score = get_score(revised_evaluation, score_key=score_key)

    if require_improvement:
        if revised_score > current_score:
            return revised_text, revised_evaluation, True
    else:
        if revised_score >= current_score:
            return revised_text, revised_evaluation, True

    return current_text, current_evaluation, False

def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    # TODO:
    # 1. Stop if iteration has reached max_iterations.
    # 2. Stop if target_score is set and current_score is high enough.
    # 3. Stop if stop_on_no_improvement is True and last_improved is False.
    # 4. Otherwise continue.

    if iteration >= max_iterations:
        return False
    if target_score is not None and current_score >= target_score:
        return False
    if stop_on_no_improvement and not last_improved:
        return False
    return True


def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    # TODO:
    # 1. Evaluate initial_text.
    # 2. Repeatedly call refine_once until should_continue says to stop.
    # 3. After each step, keep the better text using choose_better_text.
    # 4. Track every RefinementStep in a list.
    # 5. Return a result dictionary with the initial and final information.
    #
    # Be careful to avoid infinite loops.

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    evaluation = evaluation_func(initial_text)
    initial_score = get_score(evaluation, score_key=score_key)
    current_text = initial_text
    steps = []
    while should_continue(
        iteration=len(steps),
        max_iterations=max_iterations,
        current_score=get_score(evaluation, score_key=score_key),
        target_score=target_score,
        last_improved=steps[-1].accepted if steps else True,
        stop_on_no_improvement=stop_on_no_improvement,
    ):
        step = refine_once(
            original_prompt,
            current_text,
            generation_func,
            evaluation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            focus=focus,
            preserve=preserve,
            **kwargs,
        )
        current_text, evaluation, accepted = choose_better_text(
            current_text=current_text,
            revised_text=step.revised_text,
            current_evaluation=evaluation,
            revised_evaluation=step.evaluation,
            score_key=score_key,
            require_improvement=require_improvement,
        )
        step.accepted = accepted
        steps.append(step)

    final_score = get_score(evaluation, score_key=score_key)
    last_improved = steps[-1].accepted if steps else True

    if target_score is not None and final_score >= target_score:
        stopped_reason = "target_score"
    elif len(steps) >= max_iterations:
        stopped_reason = "max_iterations"
    elif stop_on_no_improvement and not last_improved:
        stopped_reason = "no_improvement"

    return {
        "initial_text": initial_text,
        "final_text": current_text,
        "iterations": steps,
        "initial_score": initial_score,
        "final_score": final_score,
        "stopped_reason": stopped_reason,
    }

def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""

    # TODO:
    # 1. Include the initial score and final score.
    # 2. Include the number of iterations.
    # 3. Include whether each iteration was accepted or rejected.
    # 4. Include the stopped reason.
    #
    # This is useful for debugging and for showing the tradeoff between extra
    # inference compute and output improvement.

    iterations = result.get("iterations", [])

    lines = [
        f"Initial score: {result.get('initial_score', 0.0):.2f}",
        f"Final score: {result.get('final_score', 0.0):.2f}",
        f"Iterations: {len(iterations)}",
        f"Stopped reason: {result.get('stopped_reason', 'unknown')}",
    ]

    for step in iterations:
        status = "accepted" if step.accepted else "rejected"
        lines.append(
            f"  Iteration {step.iteration}: "
            f"{step.previous_score:.2f} -> {step.revised_score:.2f} ({status})"
        )

    return "\n".join(lines)