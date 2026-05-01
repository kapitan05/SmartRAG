import re

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class WordF1Metric(BaseMetric):  # type: ignore
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold: float = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.expected_output or not test_case.actual_output:
            self.score = 0.0
            self.success = False
            self.reason = "Missing expected_output or actual_output."
            return self.score

        expected_words = set(re.findall(r"\w+", test_case.expected_output.lower()))
        actual_words = set(re.findall(r"\w+", test_case.actual_output.lower()))

        common_words = expected_words.intersection(actual_words)

        if not expected_words or not actual_words:
            self.score = 0.0
        else:
            precision = len(common_words) / len(actual_words)
            recall = len(common_words) / len(expected_words)

            if precision + recall == 0:
                self.score = 0.0
            else:
                self.score = 2 * (precision * recall) / (precision + recall)

        self.success = self.score >= self.threshold
        self.reason = f"Word F1 Score is {self.score:.2f}."

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Word F1"


class PrecisionAtKMetric(BaseMetric):  # type: ignore
    def __init__(self, k: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.k: int = k
        self.threshold: float = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        retrieved_k: list[str] = (
            test_case.retrieval_context[: self.k] if test_case.retrieval_context else []
        )
        expected: list[str] = test_case.context if test_case.context else []

        if not expected or not retrieved_k:
            self.score = 0.0
            self.reason = "Missing retrieval_context or expected_context."
            self.success = False
            return self.score

        hits = 0
        for exp in expected:
            if any(exp.lower() in ret.lower() for ret in retrieved_k):
                hits += 1

        self.score = hits / self.k
        self.success = self.score >= self.threshold
        self.reason = f"Found {hits} expected contexts in Top-{self.k} retrieved docs."

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return f"Precision@{self.k}"


class RecallAtKMetric(BaseMetric):  # type: ignore
    def __init__(self, k: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.k: int = k
        self.threshold: float = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        retrieved_k: list[str] = (
            test_case.retrieval_context[: self.k] if test_case.retrieval_context else []
        )
        expected: list[str] = test_case.context if test_case.context else []

        if not expected:
            self.score = 0.0
            self.reason = "Missing expected_context."
            self.success = False
            return self.score

        if not retrieved_k:
            self.score = 0.0
            self.reason = "Qdrant returned nothing."
            self.success = False
            return self.score

        hits = 0
        for exp in expected:
            if any(exp.lower() in ret.lower() for ret in retrieved_k):
                hits += 1

        self.score = hits / len(expected)
        self.success = self.score >= self.threshold
        self.reason = (
            f"Found {hits} out of {len(expected)} expected docs in Top-{self.k}."
        )

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return f"Recall@{self.k}"


recall_metric = ContextualRecallMetric(
    threshold=0.7, model="gpt-4o-mini", include_reason=True
)
faith_metric = FaithfulnessMetric(
    threshold=0.7, model="gpt-4o-mini", include_reason=True
)
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7, model="gpt-4o-mini", include_reason=True
)


# Critique Shadowing judge with strict criteria for financial metrics accuracy.
# Few-Shot examples are included in the evaluation steps to guide the model
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
