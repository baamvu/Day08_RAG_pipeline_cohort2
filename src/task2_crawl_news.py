"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-chu-bin-bi-tam-giu-vi-lien-quan-ma-tuy-4755275.html",
    "https://vnexpress.net/trum-ma-tuy-dung-sau-duong-day-lien-quan-4-tiep-vien-hang-khong-5059153.html",
    "https://vnexpress.net/227-nguoi-bi-truy-to-trong-vu-4-tiep-vien-hang-khong-xach-ma-tuy-5057648.html",
    "https://ngoisao.vnexpress.net/co-tien-nguyen-do-truc-phuong-bi-bat-vi-lien-quan-ma-tuy-4816047.html",
    "https://vnexpress.net/chu-de/chau-viet-cuong-2883",
    "https://vnexpress.net/lan-ra-duong-day-ma-tuy-lon-nhat-lich-su-tu-vu-4-tiep-vien-hang-khong-4702921.html",
]


def slugify(text: str, max_len: int = 60) -> str:
    """Tạo slug từ text, bỏ dấu tiếng Việt."""
    text = text.lower().strip()
    text = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", text)
    text = re.sub(r"[èéẹẻẽêềếệểễ]", "e", text)
    text = re.sub(r"[ìíịỉĩ]", "i", text)
    text = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", text)
    text = re.sub(r"[ùúụủũưừứựửữ]", "u", text)
    text = re.sub(r"[ỳýỵỷỹ]", "y", text)
    text = re.sub(r"đ", "d", text)
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


async def crawl_article(url: str) -> dict | None:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            if not result.success:
                print(f"  [FAIL] Failed to crawl: {url}")
                return None

            title = result.metadata.get("title", "") if result.metadata else ""
            if not title:
                title = url.split("/")[-1].replace("-", " ").replace(".html", "")

            return {
                "url": url,
                "title": title.strip(),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": result.markdown or "",
            }
    except Exception as e:
        print(f"  [ERROR] crawling {url}: {e}")
        return None


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    total = len(ARTICLE_URLS)
    success = 0

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{total}] Crawling: {url}")
        article = await crawl_article(url)

        if article is None:
            continue

        slug = slugify(article["title"]) or f"article-{i:02d}"
        filename = f"{slug}.json"
        filepath = DATA_DIR / filename

        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath.name}")
        success += 1

    print(f"\n[DONE] Crawled {success}/{total} articles -> {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
