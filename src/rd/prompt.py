"""The prompt shared by the zero-shot LLMs and the LLM fine-tunes.

Fixed before any run; never tuned on a test set.
"""

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
