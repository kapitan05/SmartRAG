import json
import logging
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
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
DOCLING_API_URL: str = os.getenv(
    "DOCLING_API_URL", "http://docling:5001/v1/convert/file"
)


def parse_filename(filename: str) -> dict[str, str]:
    """
    Extracts Year, Quarter, and Ticker from names like '2022 Q1 NVDA.pdf'
    """
    base_name = os.path.basename(filename).replace(".pdf", "")
    # Regex to catch pattern: YYYY QX TICKER
    match = re.search(r"(\d{4})\s+(Q[1-4])\s+([A-Za-z]+)", base_name)
    if match:
        return {
            "year": match.group(1),
            "quarter": match.group(2),
            "ticker": match.group(3).upper(),
        }
    return {"year": "Unknown", "quarter": "Unknown", "ticker": "Unknown"}


def condense_markdown_tables(md_text: str) -> str:
    """Removes extra spaces inside markdown tables to save characters."""
    # Replace multiple spaces surrounding a pipe with a single space
    condensed = re.sub(r"\s+\|\s+", " | ", md_text)
    # Replace multiple spaces before a pipe with a single space
    condensed = re.sub(r"\s+\|", " |", condensed)
    # Replace multiple spaces after a pipe with a single space
    condensed = re.sub(r"\|\s+", "| ", condensed)
    return condensed


def save_chunks_to_json(chunks: list[Document], output_path: str) -> None:
    """Saves LangChain Document objects to a clean JSON file."""
    data_to_save = []
    for i, chunk in enumerate(chunks):
        data_to_save.append(
            {
                "chunk_id": i,
                "metadata": chunk.metadata,
                "text": chunk.page_content,
                "char_count": len(chunk.page_content),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    logger.info(f"💾 Saved {len(chunks)} enriched chunks to {output_path}")


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


def summarize_chunk(chunk_text: str, document_context: str, llm: ChatOpenAI) -> str:
    """
    Optional: Calls an LLM to generate a strict 1-sentence summary of the chunk.
    """
    prompt = f"""You are a financial analyst. Give a 1-sentence summary of the following excerpt from a SEC filing.
    Document Context: {document_context}
    
    Excerpt:
    {chunk_text}
    
    Summary:"""

    response = llm.invoke(prompt)
    return str(response.content).strip()


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
    # llm_summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    for file_name in os.listdir(DATA_DIR):
        if not file_name.endswith(".pdf"):
            continue

        logger.info(f"\n{'=' * 40}\n🚀 STARTING FILE: {file_name}\n{'=' * 40}")

        file_path = os.path.join(DATA_DIR, file_name)
        md_file_path = file_path.replace(".pdf", ".md")
        json_file_path = file_path.replace(".pdf", "_chunks.json")

        doc_meta = parse_filename(file_path)
        logger.info(f"Extracted Metadata: {doc_meta}")

        # Check if the markdown file already exists
        if os.path.exists(md_file_path):
            logger.info(f"Loading existing markdown from {md_file_path}")
            with open(md_file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
        else:
            logger.info(f"Parsing {file_path} via Docling API...")
            markdown_content = parse_document_with_docling(file_path)
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"💾 Saved parsed markdown to {md_file_path}")

        logger.info("Condensing markdown tables to optimize chunking...")
        markdown_content = condense_markdown_tables(markdown_content)

        # Split: Markdown Headers
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        header_splits = markdown_splitter.split_text(markdown_content)

        # Split: Chunk Size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=400
        )
        final_chunks = text_splitter.split_documents(header_splits)

        # Contextual Enrichment
        logger.info(f"Enriching {len(final_chunks)} chunks with context...")
        for i, chunk in enumerate(final_chunks):
            h1 = chunk.metadata.get("Header_1", "")
            h2 = chunk.metadata.get("Header_2", "")

            section_path = " > ".join([h for h in [h1, h2] if h])

            context_header = f"{doc_meta['ticker']} Financial Report, {doc_meta['year']} {doc_meta['quarter']}."
            if section_path:
                context_header += f"\nSection: {section_path}"

            # 2. Append text (Header at top)
            chunk.page_content = f"{context_header}\n\n{chunk.page_content}"

            # (OPTIONAL) Enable this if you want the LLM summary.
            # WARNING: For a full 10-Q, this will make ~200 API calls per document.
            #  (4000 chars per chunk)
            # document_context = f"{doc_meta['ticker']} 10-Q filing for {doc_meta['year']} {doc_meta['quarter']}."
            # summary = summarize_chunk(chunk.page_content, document_context, llm)
            # context_header += f"\n[SUMMARY: {summary}]"

            # Ensure metadata is cleanly attached to Qdrant payload
            chunk.metadata.update(
                {
                    "ticker": doc_meta["ticker"],
                    "year": doc_meta["year"],
                    "quarter": doc_meta["quarter"],
                    "source": file_name,
                }
            )

        save_chunks_to_json(final_chunks, json_file_path)

        logger.info(
            f"Generating embeddings and uploading {len(final_chunks)} chunks to Qdrant"
        )
        QdrantVectorStore.from_documents(
            final_chunks,
            embeddings,
            url=f"http://{qdrant_host}:{qdrant_port}",
            collection_name=COLLECTION_NAME,
            force_recreate=False,
        )
        logger.info(f"✅ Finished processing {file_name}")

    logger.info("🎉 ALL FILES INGESTED SUCCESSFULLY!")


if __name__ == "__main__":
    run_ingestion()
