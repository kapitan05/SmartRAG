import re

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def extract_sources(text: str) -> set[str]:
    """
    Searches for [SOURCE: filename.pdf] tags in the text
    and returns a list of unique filenames.
    """
    matches = re.findall(r"\[SOURCE:\s*(.+?)\]", text)
    # Приводим к нижнему регистру и удаляем пробелы для надежности
    return set(m.strip().lower() for m in matches)


class WordF1Metric(BaseMetric):  # type: ignore
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold: float = threshold
        # Add basic stop words to ignore
        self.stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "of",
            "in",
            "to",
            "and",
        }

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.expected_output or not test_case.actual_output:
            self.score, self.success = 0.0, False
            return self.score

        # Extract words, lowercased, ignoring stop words
        expected_words = set(
            w
            for w in re.findall(r"\w+", test_case.expected_output.lower())
            if w not in self.stop_words
        )
        actual_words = set(
            w
            for w in re.findall(r"\w+", test_case.actual_output.lower())
            if w not in self.stop_words
        )

        common_words = expected_words.intersection(actual_words)

        if not expected_words or not actual_words:
            self.score = 0.0
        else:
            precision = len(common_words) / len(actual_words)
            recall = len(common_words) / len(expected_words)
            self.score = (
                0.0
                if precision + recall == 0
                else 2 * (precision * recall) / (precision + recall)
            )

        self.success = self.score >= self.threshold
        self.reason = f"Word F1 Score is {self.score:.2f} (excluding stop words)."
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Word F1"


class DocumentPrecisionMetric(BaseMetric):  # type: ignore
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold: float = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        # Берем ВЕСЬ контекст, без среза по K
        retrieved_calls: list[str] = (
            test_case.retrieval_context if test_case.retrieval_context else []
        )
        expected: list[str] = test_case.context if test_case.context else []

        if not expected or not retrieved_calls:
            self.score = 0.0
            self.reason = "Missing retrieval_context or expected_context."
            self.success = False
            return self.score

        # 1. Собираем ожидаемые файлы
        expected_sources = set()
        for exp in expected:
            extracted = extract_sources(exp)
            if extracted:
                expected_sources.update(extracted)
            else:
                expected_sources.add(exp.strip().lower())

        # 2. Собираем ВСЕ уникальные файлы, которые Агент нашел за все вызовы
        retrieved_sources = set()
        for ret in retrieved_calls:
            retrieved_sources.update(extract_sources(ret))

        if not retrieved_sources:
            self.score = 0.0
            self.success = False
            self.reason = "No sources could be extracted from retrieval_context."
            return self.score

        # 3. Считаем пересечение
        hits = len(expected_sources.intersection(retrieved_sources))

        # Precision: Какая доля из найденных Агентом документов была правильной?
        self.score = hits / len(retrieved_sources)
        self.success = self.score >= self.threshold
        self.reason = f"Out of {len(retrieved_sources)} unique docs retrieved by Agent, {hits} were correct."

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Doc_Precision"


class DocumentRecallMetric(BaseMetric):  # type: ignore
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold: float = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        # Берем ВЕСЬ контекст, без среза по K
        retrieved_calls: list[str] = (
            test_case.retrieval_context if test_case.retrieval_context else []
        )
        expected: list[str] = test_case.context if test_case.context else []

        if not expected:
            self.score = 0.0
            self.reason = "Missing expected_context."
            self.success = False
            return self.score

        if not retrieved_calls:
            self.score = 0.0
            self.reason = "Agent returned nothing."
            self.success = False
            return self.score

        # 1. Собираем ожидаемые файлы
        expected_sources = set()
        for exp in expected:
            extracted = extract_sources(exp)
            if extracted:
                expected_sources.update(extracted)
            else:
                expected_sources.add(exp.strip().lower())

        # 2. Собираем ВСЕ уникальные файлы, которые Агент нашел за все вызовы
        retrieved_sources = set()
        for ret in retrieved_calls:
            retrieved_sources.update(extract_sources(ret))

        # 3. Считаем пересечение
        hits = len(expected_sources.intersection(retrieved_sources))

        # Recall: Какую долю из ожидаемых документов Агент смог найти?
        self.score = hits / len(expected_sources) if expected_sources else 0.0
        self.success = self.score >= self.threshold
        self.reason = f"Found {hits} out of {len(expected_sources)} expected docs across all Agent retrievals."

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Doc_Recall"


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
