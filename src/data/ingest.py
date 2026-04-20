import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR: str = "sec-10K-filings"
COLLECTION_NAME: str = "sec_reports"
DOCLING_API_URL: str = "http://localhost:5001/v1/convert/file"


def parse_document_with_docling(file_path: str) -> str:
    """
    Sends a local file to the Docling microservice via HTTP POST and extracts
    the parsed markdown representation of the document.
    """
    # 1. Define the optimization options based on the documentation
    options = {
        "do_ocr": "false",  # Disable OCR
        "table_mode": "fast",  # Fast table extraction
        "image_export_mode": "placeholder",  # Avoid Base64 strings for images
        "include_images": "false",  # Skip image extraction entirely
        "to_formats": "md",  # Markdown output
    }

    with open(file_path, "rb") as f:
        files: dict[str, tuple[str, Any, str]] = {
            "files": (os.path.basename(file_path), f, "application/pdf")
        }

        print(
            f"⏳ Sending {os.path.basename(file_path)} to Docling API with optimized settings..."
        )

        # 2. Pass 'options' to the 'data' parameter to send them as form fields
        response = httpx.post(DOCLING_API_URL, files=files, data=options, timeout=None)

    if response.status_code != 200:
        raise Exception(
            f"Docling error (Status {response.status_code}): {response.text}"
        )

    json_response = response.json()

    # 3. Extract the markdown based on the documented response format
    # {"document": {"md_content": "...", ...}, "status": "success"}
    document_data = json_response.get("document", {})
    markdown_text: str = document_data.get("md_content", "")

    # Fallback just in case you are using an older version of the container
    if not markdown_text and "markdown" in json_response:
        markdown_text = json_response.get("markdown", "")

    if not markdown_text:
        print("⚠️ Warning: No markdown content was returned from Docling.")

    return markdown_text


def run_ingestion() -> None:
    """
    Executes the complete ingestion pipeline: ensures Qdrant collection exists,
    parses a PDF via Docling, splits Markdown text, and uploads embeddings.
    """
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    # for now parse only one document
    file_path = f"{DATA_DIR}/NVIDIA_SEC_10K_2023.pdf"
    md_file_path = file_path.replace(".pdf", ".md")

    # Check if the markdown file already exists
    if os.path.exists(md_file_path):
        print(
            f"⏭️  Markdown file {md_file_path} already exists. Skipping Docling parsing and Qdrant upload."
        )
        return

    print(f"⏳ Parsing {file_path} via Docling API...")
    markdown_content = parse_document_with_docling(file_path)

    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"💾 Saved parsed markdown to {md_file_path}")

    doc = Document(
        page_content=markdown_content,
        metadata={"company": "Nvidia", "year": 2023, "source": file_path},
    )

    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([doc])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=f"http://{qdrant_host}:{qdrant_port}",
        collection_name=COLLECTION_NAME,
        force_recreate=False,
    )
    print("✅ Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
