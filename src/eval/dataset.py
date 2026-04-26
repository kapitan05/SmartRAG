import csv
import io
import logging
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

# Раскомментируй импорт google.cloud в продакшне
# from google.cloud import storage
from langsmith import Client

logger = logging.getLogger(__name__)
load_dotenv()


def get_benchmark_from_local(file_path: str) -> str:
    """Читает локальный CSV файл и возвращает его содержимое как строку."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Локальный файл не найден: {file_path}")

    # Path.read_text автоматически открывает и безопасно закрывает файл
    return path.read_text(encoding="utf-8")


def get_benchmark_from_gcs(bucket_name: str, file_path: str) -> str:
    """Скачивает CSV из GCS как строку (для production)."""
    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(file_path)
    # raw_bytes = cast(bytes, blob.download_as_string())
    # return raw_bytes.decode("utf-8")
    raise NotImplementedError("GCS загрузка временно отключена для локальных тестов")


def sync_csv_to_langsmith(dataset_name: str, csv_content: str) -> None:
    """Парсит CSV и загружает в LangSmith с метаданными."""
    client = Client()

    # 1. Парсим CSV
    reader = csv.DictReader(io.StringIO(csv_content))

    inputs = []
    outputs = []
    metadata = []

    for row in reader:
        # LangSmith Inputs
        inputs.append({"question": row.get("Question", "")})
        # LangSmith Outputs
        outputs.append({"expected_answer": row.get("Answer", "")})
        # LangSmith Metadata (Супер-фича для фильтрации в дашборде!)
        metadata.append(
            {
                "source_docs": row.get("Source Docs", ""),
                "question_type": row.get("Question Type", ""),
                "source_chunk_type": row.get("Source Chunk Type", ""),
            }
        )

    # 2. Пересоздаем датасет
    if client.has_dataset(dataset_name=dataset_name):
        logger.info(f"Deleting old dataset {dataset_name}...")
        client.delete_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(
        dataset_name=dataset_name, description="Gold Benchmark (CSV)"
    )

    # 3. Загружаем все батчем
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=dataset.id,
    )
    logger.info(f"Synced {len(inputs)} complex examples to LangSmith.")


if __name__ == "__main__":
    # Настраиваем логирование, чтобы видеть процесс в терминале
    logging.basicConfig(level=logging.INFO)

    # 1. Задай имя твоего тестового датасета
    DATASET_NAME = "RAG_Gold_Benchmark_v1"

    # 2. Укажи путь к локальному CSV файлу (например, создай папку data/ в корне проекта)
    LOCAL_CSV_PATH = "evaluation_data/ground_truth_test.csv"

    logger.info(f"Начинаем загрузку локального датасета из {LOCAL_CSV_PATH}...")

    try:
        # Читаем локальный файл
        csv_data = get_benchmark_from_local(LOCAL_CSV_PATH)

        # Отправляем в LangSmith
        sync_csv_to_langsmith(DATASET_NAME, csv_data)

        logger.info("✅ Загрузка успешно завершена! Проверь дашборд LangSmith.")

    except FileNotFoundError as e:
        logger.error(f"❌ Ошибка: {e}")
        logger.info("Пожалуйста, создай файл с тестовыми данными по указанному пути.")
    except Exception as e:
        logger.error(f"❌ Произошла непредвиденная ошибка: {e}")
