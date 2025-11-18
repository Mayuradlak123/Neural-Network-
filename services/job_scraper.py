from langchain_community.document_loaders import WebBaseLoader
from config.logger import logger
from typing import Optional, Dict
import re


class JobScraperService:
    """Service to scrape and extract job descriptions from URLs using LangChain WebBaseLoader."""

    def __init__(self):
        self.loader = None

    def extract_job_description(self, url: str) -> Optional[Dict[str, str]]:
        logger.info(f"[START] Extracting job description from: {url}")

        try:
            # Initialize WebBaseLoader
            self.loader = WebBaseLoader(url)
            logger.debug(f"Initialized WebBaseLoader for: {url}")

            # Load and extract documents
            logger.debug("Loading web page content...")
            documents = self.loader.load()

            if not documents:
                status_code = 400
                logger.warning(
                    f"[FAILED] No content extracted from URL: {url} "
                    f"(Status Code: {status_code})"
                )
                return None

            content = ""
            metadata = {}

            for doc in documents:
                content += doc.page_content
                metadata.update(doc.metadata)

            logger.info(
                f"[SUCCESS] Extracted {len(content)} characters from {url} "
                f"(Status Code: 200)"
            )
            logger.debug(f"Document metadata: {metadata}")

            # Clean and process the content
            cleaned_content = self._clean_content(content)

            result = {
                "url": url,
                "raw_content": content,
                "cleaned_content": cleaned_content,
                "metadata": metadata,
                "char_count": len(cleaned_content),
                "status_code": 200,
            }

            logger.debug(f"Extraction result summary: { {k: v for k, v in result.items() if k != 'raw_content'} }")
            return result

        except Exception as e:
            status_code = 500
            logger.exception(
                f"[EXCEPTION] Error extracting job description from {url}: {e} "
                f"(Status Code: {status_code})"
            )
            raise

        finally:
            logger.debug(f"[END] Job description extraction attempt completed for {url}")

    def _clean_content(self, text: str) -> str:
        """Removes excessive whitespace, line breaks, and HTML artifacts."""
        try:
            logger.debug("Cleaning extracted text content...")
            cleaned = re.sub(r"\s+", " ", text)
            cleaned = re.sub(r"<!--.*?-->", "", cleaned)
            cleaned = cleaned.strip()
            logger.debug("Text content cleaned successfully.")
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning content: {e}")
            return text
