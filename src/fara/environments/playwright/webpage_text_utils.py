"""Utility class for extracting text/markdown from web pages via Playwright."""

import io
import logging
import os
import tempfile
from typing import Any, Optional

import tiktoken
from playwright.async_api import Page

logger = logging.getLogger(__name__)


class WebpageTextUtilsPlaywright:
    def __init__(self):
        self._markdown_converter: Optional[Any] = None
        self._page_script: str = ""

        with open(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"),
            "rt",
            encoding="utf-8",
        ) as fh:
            self._page_script = fh.read()

    async def get_all_webpage_text(self, page: Page, n_lines: int = 50) -> str:
        """Retrieve the text content of the web page.

        Args:
            page: The Playwright page object.
            n_lines: The number of lines to return from the page inner text.

        Returns:
            The text content of the page.
        """
        try:
            text_in_viewport = await page.evaluate("""() => {
                return document.body.innerText;
            }""")
            text_in_viewport = "\n".join(text_in_viewport.split("\n")[:n_lines])
            text_in_viewport = "\n".join(
                [line for line in text_in_viewport.split("\n") if line.strip()]
            )
            assert isinstance(text_in_viewport, str)
            return text_in_viewport
        except Exception:
            return ""

    async def get_visible_text(self, page: Page) -> str:
        """Retrieve the text content of the browser viewport (approximately).

        Args:
            page: The Playwright page object.

        Returns:
            The text content of the page.
        """
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        result = await page.evaluate("WebSurfer.getVisibleText();")
        assert isinstance(result, str)
        return result

    async def get_page_markdown(self, page: Page, max_tokens: int = -1) -> str:
        """Retrieve the markdown content of the web page.
        Automatically detects and handles PDF content.

        Args:
            page: The Playwright page object.
            max_tokens: The maximum number of tokens to return. -1 means no limit.

        Returns:
            The markdown content of the page or extracted PDF content.
        """
        is_pdf = await self._is_pdf_page(page)

        if is_pdf:
            pdf_content = await self._extract_pdf_content(page)
            if max_tokens == -1:
                return pdf_content
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
            tokens = tokenizer.encode(pdf_content)
            return tokenizer.decode(tokens[:max_tokens])

        if self._markdown_converter is None:
            from markitdown import MarkItDown

            self._markdown_converter = MarkItDown()
        html = await page.evaluate("document.documentElement.outerHTML;")
        res = self._markdown_converter.convert_stream(
            io.BytesIO(html.encode("utf-8")), file_extension=".html", url=page.url
        )
        text_content = res.text_content

        if max_tokens == -1:
            return text_content
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        tokens = tokenizer.encode(text_content)
        return tokenizer.decode(tokens[:max_tokens])

    async def _is_pdf_page(self, page: Page) -> bool:
        """Check if the current page is a PDF document."""
        if page.url.lower().endswith(".pdf"):
            return True

        result = await page.evaluate("""() => {
            if (document.contentType === 'application/pdf') return true;
            if (document.querySelector('embed[type="application/pdf"]') ||
                document.querySelector('object[type="application/pdf"]')) return true;
            if (window.PDFViewerApplication || document.querySelector('#viewer.pdfViewer')) return true;
            return false;
        }""")

        return result

    async def _extract_pdf_content(self, page: Page) -> str:
        """Extract text content from a PDF page."""
        url = page.url

        try:
            browser_text = await self._extract_pdf_browser(page)
            if browser_text and len(browser_text) > 100:
                return browser_text

            logger.info("Using MarkItDown for better PDF extraction...")
            pdf_buffer = await page.context.request.get(url)
            pdf_data = await pdf_buffer.body()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(pdf_data)

            if self._markdown_converter is None:
                from markitdown import MarkItDown

                self._markdown_converter = MarkItDown()
            result = self._markdown_converter.convert(temp_file_path)

            os.unlink(temp_file_path)

            return result.text_content

        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return f"Error extracting PDF content: {str(e)}"

    async def _extract_pdf_browser(self, page: Page) -> str:
        """Extract text content from a PDF page using browser methods."""
        pdf_text = await page.evaluate("""() => {
            if (window.PDFViewerApplication) {
                const textContent = document.querySelectorAll('.textLayer div');
                if (textContent.length > 0) {
                    return Array.from(textContent).map(div => div.textContent).join('\\n');
                }
            }
            const textElements = Array.from(document.querySelectorAll('p, span, div'))
                .filter(el => {
                    const style = window.getComputedStyle(el);
                    return style.display !== 'none' &&
                           style.visibility !== 'hidden' &&
                           el.textContent.trim() !== '';
                });
            return textElements.map(el => el.textContent).join('\\n');
        }""")

        return pdf_text
