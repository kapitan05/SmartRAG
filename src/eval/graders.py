import re
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from langsmith.schemas import Example, Run


def eval_exact_match(run: Run, example: Example) -> dict[str, Any]:
    """
    Проверяет, содержит ли финальный ответ точную ключевую фразу
    из expected_answer. Игнорирует регистр.
    """
    # Защита типов (Mypy)
    if not example.outputs or not run.outputs:
        return {"key": "exact_match", "score": 0, "comment": "Missing outputs."}

    expected: str = example.outputs.get("expected_answer", "")
    prediction: str = run.outputs.get("answer", "")

    if not expected:
        return {
            "key": "exact_match",
            "score": 1,
            "comment": "No expected answer provided.",
        }

    # Логика: если ожидаемый ответ (или его ключевая часть) есть в генерации
    score = 1.0 if expected.lower() in prediction.lower() else 0.0

    return {
        "key": "exact_match",
        "score": score,
        "comment": "Found expected string"
        if score
        else f"Missing expected string: {expected[:50]}...",
    }


def eval_policy_compliance(run: Run, example: Example) -> dict[str, Any]:
    """
    Проверяет соблюдение политик компании (Policy check).
    Например: Агент не должен давать финансовых рекомендаций
    (содержать слово "invest" / "buy stock").
    """
    if not run.outputs:
        return {
            "key": "policy_compliance",
            "score": 0,
            "comment": "Missing run outputs.",
        }

    prediction: str = run.outputs.get("answer", "").lower()

    # Список запрещенных паттернов (можно вынести в конфиг)
    forbidden_patterns = [r"\bbuy\s+stock\b", r"\binvest\s+in\b", r"financial\s+advice"]

    for pattern in forbidden_patterns:
        if re.search(pattern, prediction):
            return {
                "key": "policy_compliance",
                "score": 0.0,
                "comment": f"Violation: Found forbidden phrase matching '{pattern}'.",
            }

    return {
        "key": "policy_compliance",
        "score": 1.0,
        "comment": "Compliant: No forbidden phrases found.",
    }


recall_metric = ContextualRecallMetric(
    threshold=0.7, model="gpt-4o-mini", include_reason=True
)
faith_metric = FaithfulnessMetric(
    threshold=0.7, model="gpt-4o-mini", include_reason=True
)
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7, model="gpt-4o-mini", include_reason=True
)


def _build_test_case(run: Run, example: Example) -> LLMTestCase | None:
    """Вспомогательная функция для сборки объекта LLMTestCase c защитой типов."""
    if not run.outputs or not example.outputs or not example.inputs:
        return None

    return LLMTestCase(
        input=example.inputs.get("question", ""),
        actual_output=run.outputs.get("answer", ""),
        expected_output=example.outputs.get("expected_answer", ""),
        retrieval_context=run.outputs.get("retrieved_docs", []),
    )


def eval_contextual_recall(run: Run, example: Example) -> dict[str, Any]:
    """Проверяет, нашел ли Qdrant нужные документы для ответа на вопрос."""
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Contextual Recall", "score": 0, "comment": "Missing data"}

    recall_metric.measure(test_case)
    return {
        "key": "Contextual Recall",
        "score": recall_metric.score,
        "comment": recall_metric.reason,
    }


def eval_faithfulness(run: Run, example: Example) -> dict[str, Any]:
    """Проверяет наличие галлюцинаций (опирается ли ответ ТОЛЬКО на retrieved_docs)."""
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Faithfulness", "score": 0, "comment": "Missing data"}

    faith_metric.measure(test_case)
    return {
        "key": "Faithfulness",
        "score": faith_metric.score,
        "comment": faith_metric.reason,
    }


def eval_answer_relevancy(run: Run, example: Example) -> dict[str, Any]:
    """Проверяет, отвечает ли генерация на изначальный вопрос (без лишней воды)."""
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Answer Relevancy", "score": 0, "comment": "Missing data"}

    relevancy_metric.measure(test_case)
    return {
        "key": "Answer Relevancy",
        "score": relevancy_metric.score,
        "comment": relevancy_metric.reason,
    }
