"""
Reasoning prompt helpers for Weird AI.

This module introduces the first lightweight reasoning-related utilities for
Weird AI. These functions do not train a reasoning model yet. Instead, they
help students explore how prompt structure can encourage a model to produce
intermediate reasoning steps before generating a final answer.

Lesson 8 connects the Weird AI parody assistant to the reasoning model portion
of the course by focusing on inference-time reasoning prompts.
"""


def build_reasoning_prompt(user_prompt):
    """
    Convert a normal user prompt into a general reasoning prompt.

    The returned prompt should encourage the model to think through the task
    before giving its final answer.

    Args:
        user_prompt: The original user request as a string.

    Returns:
        A formatted prompt string that includes the original task and asks for
        step-by-step reasoning.
    """

    
    reasoning_prompt = "You are Weird AI.\n" \
    "Think step-by-step before writing the final answer:\n" \
    "1. Understand the user's request.\n" \
    "2. Identify important constraints.\n" \
    "3. Plan the response.\n" \
    "4. Produce the final answer.\n\n" \
    f"User prompt: {user_prompt}\n\n" \
    "Final Answer.\n\n" \

    return reasoning_prompt


def build_parody_reasoning_prompt(topic):
    """
    Build a reasoning prompt specifically for parody lyric generation.

    This prompt should guide Weird AI through a simple creative planning process
    before asking it to write the final parody lyrics.

    Args:
        topic: The parody topic or user request as a string.

    Returns:
        A formatted prompt string with step-by-step parody planning sections.
    """

    reasoning_parody_prompt = "You are Weird AI.\n" \
    "Think step-by-step before writing the final parody:\n" \
    "1. Identify the topic.\n" \
    "2. Identify funny ideas or exaggerations.\n" \
    "3. Identify possible rhymes.\n" \
    "4. Write the final parody lyrics.\n\n" \
    f"Topic: {topic}\n\n" \
    "Final Parody Lyrics:\n"    
    return reasoning_parody_prompt


def build_closed_world_prompt(premises, question):
    """
    Build a prompt for closed-world reasoning.

    Closed-world reasoning means the model should answer using only the facts
    explicitly provided in the prompt. It should not use outside knowledge.

    Args:
        premises: A list of premise strings.
        question: The question to answer from those premises.

    Returns:
        A formatted prompt string for closed-world reasoning.
    """
    closed_world_prompt = "You are Weird AI.\n" \
    "Use only the provided premises to answer the question.\n" \
    "Premises:\n"

    for i, premise in enumerate(premises, start=1):
        closed_world_prompt += f"{i}. {premise}\n"

    closed_world_prompt += f"\nQuestion: {question}\n\n" \
    "Reasoning:\n" \
    "Answer:\n"

    return closed_world_prompt


def build_open_world_prompt(premises, question):
    """
    Build a prompt for open-world reasoning.

    Open-world reasoning allows the model to consider both the provided premises
    and relevant background knowledge. This can reveal contradictions between
    the prompt and the model's background knowledge.

    Args:
        premises: A list of premise strings.
        question: The question to answer.

    Returns:
        A formatted prompt string for open-world reasoning.
    """


    open_world_prompt = "You are Weird AI.\n" \
    "You may use relevant background knowledge to answer the question.\n" \
    "However, you must also consider whether the provided premises conflict with known facts.\n" \
    "Premises:\n"

    for i, premise in enumerate(premises, start=1):
        open_world_prompt += f"{i}. {premise}\n"

    open_world_prompt += f"\nQuestion: {question}\n\n" \
    "Reasoning:\n" \
    "Answer:\n"

    return open_world_prompt


