"""The prompt shared by the zero-shot LLMs and the LLM fine-tunes.

Fixed before any run; never tuned on a test set.
"""

import re

ANSWERS = ("real", "ai")

INSTRUCTION = (
    "You check product reviews for an online shop. Some reviews were written by real customers, others by an "
    "AI model. Decide which one this review is."
)


def category_name(category: str) -> str:
    """'Tools_and_Home_Improvement_5' -> 'Tools and Home Improvement'."""
    return category.removesuffix("_5").replace("_", " ")


def review_block(text: str, category: str, rating) -> str:
    return f"Category: {category_name(category)}\nRating: {int(rating)}/5\nReview: {text}"


def classify_messages(text: str, category: str, rating) -> list[dict]:
    """Chat messages asking for a one-word answer, `real` or `ai`."""
    content = f"{INSTRUCTION}\n\n{review_block(text, category, rating)}\n\nAnswer with one word: real or ai."
    return [{"role": "user", "content": content}]


EXPLAIN_SUFFIX = (
    "First give one or two short sentences on the clues you see, then the verdict on its own last line, exactly "
    "`Answer: real` or `Answer: ai`."
)
ANSWER_RE = re.compile(r"Answer:\**\s*`?(real|ai)`?\.?\s*\Z", re.IGNORECASE)


def explain_messages(text: str, category: str, rating) -> list[dict]:
    """Chat messages asking for a short explanation, then `Answer: real|ai` (used by GRPO and the demo)."""
    content = f"{INSTRUCTION}\n\n{review_block(text, category, rating)}\n\n{EXPLAIN_SUFFIX}"
    return [{"role": "user", "content": content}]


def parse_answer(completion: str) -> str | None:
    """`real` or `ai` if the completion ends with a well-formed answer line, else None."""
    m = ANSWER_RE.search(completion.strip())
    return m.group(1).lower() if m else None
