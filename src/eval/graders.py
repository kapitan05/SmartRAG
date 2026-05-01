import re
from typing import Any

from deepeval.test_case import LLMTestCase
from langsmith.schemas import Example, Run

from src.eval.custom_metrics import (
    PrecisionAtKMetric,
    RecallAtKMetric,
    WordF1Metric,
    custom_business_metric,
    faith_metric,
    recall_metric,
    relevancy_metric,
)


def extract_expected_sources(answer_text: str) -> list[str]:
    """Extracts expected sources from the end of the answer."""
    match = re.search(r"SOURCE\(S\):\s*(.+)", answer_text, flags=re.IGNORECASE)
    if match:
        sources_str = match.group(1)
        return [s.strip() for s in sources_str.split(",")]
    return []


# wrapper for custom metrics with better type safety and error handling
def _build_test_case(run: Run, example: Example) -> LLMTestCase | None:
    """Вспомогательная функция для сборки объекта LLMTestCase c защитой типов."""
    if not run.outputs or not example.outputs or not example.inputs:
        return None

    expected_answer = str(example.outputs.get("expected_answer", ""))

    expected_docs = example.outputs.get("expected_context", [])
    if not expected_docs:
        expected_docs = extract_expected_sources(expected_answer)

    return LLMTestCase(
        input=example.inputs.get("question", ""),
        actual_output=run.outputs.get("answer", ""),
        expected_output=expected_answer,
        retrieval_context=run.outputs.get("retrieved_docs", []),
        context=expected_docs,
    )


# ==========================================
#  PRECISION METRICS
# ==========================================


def evaluate_precision_at_1(run: Run, example: Example) -> dict[str, Any]:
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Precision@1", "score": 0.0}

    metric = PrecisionAtKMetric(k=1, threshold=0.0)
    metric.measure(test_case)
    return {"key": "Precision@1", "score": metric.score}


def evaluate_precision_at_3(run: Run, example: Example) -> dict[str, Any]:
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Precision@3", "score": 0.0}

    metric = PrecisionAtKMetric(k=3, threshold=0.0)
    metric.measure(test_case)
    return {"key": "Precision@3", "score": metric.score}


def evaluate_precision_at_10(run: Run, example: Example) -> dict[str, Any]:
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Precision@10", "score": 0.0}

    metric = PrecisionAtKMetric(k=10, threshold=0.0)
    metric.measure(test_case)
    return {"key": "Precision@10", "score": metric.score}


# ==========================================
# RECALL METRICS
# ==========================================


def evaluate_recall_at_1(run: Run, example: Example) -> dict[str, Any]:
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Recall@1", "score": 0.0}

    metric = RecallAtKMetric(k=1, threshold=0.0)
    metric.measure(test_case)
    return {"key": "Recall@1", "score": metric.score, "comment": metric.reason}


def evaluate_recall_at_3(run: Run, example: Example) -> dict[str, Any]:
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Recall@3", "score": 0.0}

    metric = RecallAtKMetric(k=3, threshold=0.0)
    metric.measure(test_case)
    return {"key": "Recall@3", "score": metric.score, "comment": metric.reason}


def evaluate_recall_at_10(run: Run, example: Example) -> dict[str, Any]:
    test_case = _build_test_case(run, example)
    if not test_case:
        return {"key": "Recall@10", "score": 0.0}

    metric = RecallAtKMetric(k=10, threshold=0.0)
    metric.measure(test_case)
    return {"key": "Recall@10", "score": metric.score, "comment": metric.reason}


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


# ==========================================
# GENERATION METRICS
# ==========================================


def evaluate_word_f1(run: Run, example: Example) -> dict[str, Any]:
    test_case = _build_test_case(run, example)
    if not test_case:
        return {
            "key": "Word F1",
            "score": 0.0,
            "comment": "Missing data in run/example",
        }

    metric = WordF1Metric(threshold=0.5)
    metric.measure(test_case)
    return {"key": "Word F1", "score": metric.score}


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
