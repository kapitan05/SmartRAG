from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langsmith.schemas import Example, Run

# Создаем кастомного судью с применением Critique Shadowing
# Мы жестко прописываем критерии и даем примеры того, что считать ошибкой.
custom_business_metric = GEval(
    name="Business Accuracy",
    criteria="Determine if the answer contains correct financial metrics strictly according to the SEC filings.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[
        "1. Read the expected financial metrics.",
        "2. Check if the actual output includes these EXACT numbers.",
        "3. FEW-SHOT EXAMPLE: If expected is '$1.2 Billion' and actual is '$1.2B', score it 1.0.",
        "4. FEW-SHOT EXAMPLE: If expected is 'Net income increased by 10%' and actual is 'Revenue increased by 10%', score it 0.0 (Income vs Revenue is a critical mistake).",
        "5. Ignore polite conversational filler (e.g., 'According to the report...').",
    ],
    model="gpt-4o",
)


def eval_custom_business_logic(run: Run, example: Example) -> dict[str, Any]:
    """LangSmith обертка для кастомной GEval метрики с Few-Shot примерами."""
    if not run.outputs or not example.outputs or not example.inputs:
        return {"key": "Business Accuracy", "score": 0, "comment": "Missing data"}

    test_case = LLMTestCase(
        input=example.inputs.get("question", ""),
        actual_output=run.outputs.get("answer", ""),
        expected_output=example.outputs.get("expected_answer", ""),
    )

    custom_business_metric.measure(test_case)
    return {
        "key": "Business Accuracy",
        "score": custom_business_metric.score,
        "comment": custom_business_metric.reason,
    }
