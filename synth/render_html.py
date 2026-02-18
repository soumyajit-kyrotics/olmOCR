"""
render_html.py — Render HTML files to PDF and PNG using Playwright.

Uses headless Chromium via Playwright to render the generated HTML
files into:
  - PDF files (for use as input to the existing OCR pipeline)
  - PNG screenshots (for visual comparison and GRPO training)

These rendered images become the synthetic training inputs —
the model sees the rendered page and must produce text that
passes the unit tests extracted from the source HTML.
"""

import argparse
import asyncio
import glob
import json
import os
import sys


async def render_html_to_png(html_path: str, png_path: str, width: int = 800, height: int = 1000) -> None:
    """
    Render an HTML file to a PNG screenshot using Playwright.

    Args:
        html_path: path to the HTML file
        png_path: output path for the PNG screenshot
        width: viewport width in pixels
        height: viewport height in pixels
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": width, "height": height})

        # Load the HTML file
        file_url = f"file://{os.path.abspath(html_path)}"
        await page.goto(file_url, wait_until="networkidle")

        # Take a full-page screenshot
        await page.screenshot(path=png_path, full_page=False)
        await browser.close()


async def render_html_to_pdf(html_path: str, pdf_path: str, width: int = 800, height: int = 1000) -> None:
    """
    Render an HTML file to a PDF using Playwright.

    Args:
        html_path: path to the HTML file
        pdf_path: output path for the PDF
        width: viewport width in pixels
        height: viewport height in pixels
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": width, "height": height})

        file_url = f"file://{os.path.abspath(html_path)}"
        await page.goto(file_url, wait_until="networkidle")

        await page.pdf(
            path=pdf_path,
            width=f"{width}px",
            height=f"{height}px",
            print_background=True,
        )
        await browser.close()


async def render_directory(html_dir: str, output_dir: str, fmt: str = "png") -> int:
    """
    Render all HTML files in a directory to PNG or PDF.

    Reads the corresponding .json metadata file for each HTML file
    to get the correct viewport dimensions.

    Returns the number of files rendered.
    """
    from playwright.async_api import async_playwright

    os.makedirs(output_dir, exist_ok=True)
    html_files = sorted(glob.glob(os.path.join(html_dir, "*.html")))

    if not html_files:
        print(f"No HTML files found in {html_dir}")
        return 0

    count = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        for html_path in html_files:
            basename = os.path.splitext(os.path.basename(html_path))[0]
            ext = "png" if fmt == "png" else "pdf"
            out_path = os.path.join(output_dir, f"{basename}.{ext}")

            # Try to read dimensions from metadata
            meta_path = html_path.replace(".html", ".json")
            width, height = 800, 1000
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    width = meta.get("width", 800)
                    height = meta.get("height", 1000)

            # Force A4 aspect ratio (1:1.414) for consistent portrait pages
            height = round(width * 1.414)

            page = await browser.new_page(viewport={"width": width, "height": height})
            file_url = f"file://{os.path.abspath(html_path)}"
            await page.goto(file_url, wait_until="networkidle")

            if fmt == "png":
                await page.screenshot(path=out_path, full_page=False)
            else:
                await page.pdf(
                    path=out_path,
                    width=f"{width}px",
                    height=f"{height}px",
                    print_background=True,
                )

            await page.close()
            count += 1
            print(f"  Rendered: {basename}.{ext} ({width}x{height})")

        await browser.close()

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render HTML files to PNG/PDF using Playwright"
    )
    parser.add_argument("--html-dir", required=True, help="Directory containing HTML files")
    parser.add_argument("--output-dir", default="synth_output/rendered", help="Output directory")
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="png", help="Output format")
    args = parser.parse_args()

    total = 0
    if args.format in ("png", "both"):
        png_dir = os.path.join(args.output_dir, "png") if args.format == "both" else args.output_dir
        n = asyncio.run(render_directory(args.html_dir, png_dir, "png"))
        total += n
        print(f"Rendered {n} PNG files → {png_dir}")

    if args.format in ("pdf", "both"):
        pdf_dir = os.path.join(args.output_dir, "pdf") if args.format == "both" else args.output_dir
        n = asyncio.run(render_directory(args.html_dir, pdf_dir, "pdf"))
        total += n
        print(f"Rendered {n} PDF files → {pdf_dir}")

    print(f"\nDone! {total} files rendered.")


if __name__ == "__main__":
    main()
