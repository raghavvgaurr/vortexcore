"""The prompts must match the notebook word for word, so accuracy stays the same."""
from src.generation import generation_prompt
from src.query_processing import decomposition_prompt, rewrite_prompt


def test_prompt_variables():
    assert set(rewrite_prompt.input_variables) == {"question"}
    assert set(decomposition_prompt.input_variables) == {"question"}
    assert set(generation_prompt.input_variables) == {"context", "question"}