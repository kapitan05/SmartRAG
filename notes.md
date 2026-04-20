data/ingest.py

def parse_document_with_docling:

-httpx.post(DOCLING_API_URL, files=files, timeout=120.0) (files=files, so post knows, tht it need to transfer bites
now, so http request like ... multipart/form-data )

-        files: dict[str, tuple[str, Any, str]] = {
            "file": (os.path.basename(file_path), f, "application/pdf")
        } # "file" - because Docling expects exactly this name (form field name), so to put it to httml
             <input type="file" name="file">;
- #application/pdf - (MIME type) global naming to distinguish what bites represent
