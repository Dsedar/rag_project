"""
Парсер статей из Fandom Wiki по вселенной Ведьмака.
Рекурсивно обходит указанную категорию, очищает текст статей
и сохраняет результат в JSON-файл для последующей индексации.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Добавляем корень проекта в sys.path для импорта rag_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_config import settings

try:
    import html2text
    import requests
    from bs4 import BeautifulSoup, Comment
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry
except ImportError:
    html2text = None
    requests = None
    BeautifulSoup = None
    Comment = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wiki_parser")


def get_http_session(retries: int = 3, backoff_factor: float = 0.5) -> Any:
    """Создаёт HTTP-сессию с автоматическими повторными попытками."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "Witcher3RAG-Crawler/1.0 (Educational research project)"}
    )
    return session


def clean_postprocessing(text: str) -> str:
    """Очищает Markdown/HTML текст от вики-разметки и лишних пробелов."""
    if not text:
        return ""
    text = re.sub(r"''+'(.*?)''+'", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&mdash;", "—", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\.{2,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.M)
    text = re.sub(r"^[.,;:!?—\-—\s]+$", "", text, flags=re.M)
    text = re.sub(r"\.{2,}", "...", text)
    return text.strip()


def get_page_content(
    title: str, wiki_domain: str, subcategory: str, session: Any
) -> Optional[Dict[str, str]]:
    """Загружает содержимое статьи через MediaWiki API, удаляет мусор и конвертирует в текст."""
    url = f"https://{wiki_domain}/api.php"
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": True,
    }
    try:
        response = session.get(url, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning(f"Ошибка запроса для '{title}': {e}")
        return None

    if "error" in data:
        logger.warning(f"Пропущено (ошибка API) '{title}': {data['error'].get('info')}")
        return None

    try:
        html_content = data["parse"]["text"]["*"]
    except KeyError:
        logger.warning(f"Нет HTML для статьи: {title}")
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    # Удаление неинформативных тегов и блоков
    selectors_to_remove = [
        "script",
        "style",
        "nav",
        "footer",
        "sup",
        "table",
        "aside",
        "img",
        "figcaption",
        ".mw-editsection",
        ".reference",
        ".navbox",
        ".toc",
        ".gallery",
    ]
    for selector in selectors_to_remove:
        for el in soup.select(selector):
            el.decompose()

    # Удаление HTML-комментариев
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # Конвертация в чистый Markdown/текст
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.ignore_images = True
    text_maker.body_width = 0
    raw_text = text_maker.handle(str(soup))
    clean_text = clean_postprocessing(raw_text)

    if not clean_text:
        return None

    return {
        "title": data["parse"].get("title", title),
        "text": clean_text,
        "url": f"https://{wiki_domain}/wiki/{title.replace(' ', '_')}",
        "subcategory": subcategory,
    }


def get_pages_from_category(
    category: str, wiki_domain: str, session: Any, limit: int = 500
) -> Tuple[List[str], List[str]]:
    """Получает список статей (ns=0) и подкатегорий (ns=14) внутри категории."""
    url = f"https://{wiki_domain}/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": limit,
        "format": "json",
    }
    pages: List[str] = []
    subcats: List[str] = []

    while True:
        try:
            response = session.get(url, params=params, timeout=12)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Ошибка API при получении категории '{category}': {e}")
            break

        for member in data.get("query", {}).get("categorymembers", []):
            ns = member.get("ns")
            member_title = member.get("title")
            if ns == 14:  # Подкатегория
                subcats.append(member_title)
            elif ns == 0:  # Основная статья
                pages.append(member_title)

        if "continue" in data:
            params.update(data["continue"])
            time.sleep(settings.SCRAPER_DELAY)
        else:
            break

    return pages, subcats


def get_all_pages_recursive(
    category: str,
    wiki_domain: str,
    session: Any,
    visited: Optional[Set[str]] = None,
    max_depth: int = 5,
    depth: int = 0,
) -> Dict[str, Set[str]]:
    """Рекурсивный обход подкатегорий с ограничением по глубине."""
    if visited is None:
        visited = set()
    if depth >= max_depth or category in visited:
        return {}

    indent = "  " * depth
    logger.info(f"{indent}🔍 Обход категории (глубина {depth}): {category}")
    visited.add(category)

    pages, subcats = get_pages_from_category(category, wiki_domain, session)
    result: Dict[str, Set[str]] = {}

    for page in pages:
        if page not in result:
            result[page] = set()
        result[page].add(category)

    for subcat in subcats:
        if subcat not in visited:
            sub_result = get_all_pages_recursive(
                category=subcat,
                wiki_domain=wiki_domain,
                session=session,
                visited=visited,
                max_depth=max_depth,
                depth=depth + 1,
            )
            for title, cats in sub_result.items():
                if title not in result:
                    result[title] = set()
                result[title].update(cats)

    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Парсер статей Вики по Ведьмаку 3.")
    parser.add_argument(
        "--category",
        type=str,
        default=settings.ROOT_CATEGORY,
        help=f"Корневая категория для парсинга (по умолчанию: '{settings.ROOT_CATEGORY}')",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=settings.WIKI_DOMAIN,
        help=f"Домен Вики (по умолчанию: '{settings.WIKI_DOMAIN}')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.DATA_FILE),
        help=f"Путь к выходному JSON-файлу (по умолчанию: '{settings.DATA_FILE}')",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=settings.SCRAPER_DELAY,
        help="Задержка между запросами в секундах (по умолчанию: 0.5)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Максимальная глубина рекурсии категорий (по умолчанию: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if html2text is None or requests is None or BeautifulSoup is None:
        logger.error(
            "❌ Не установлены необходимые пакеты (requests, beautifulsoup4, html2text).\n"
            "Установите зависимости: pip install -r requirements.txt"
        )
        sys.exit(1)

    logger.info(f"🚀 Запуск парсера Fandom Wiki ({args.domain})")
    logger.info(f"📂 Корневая категория: '{args.category}'")
    logger.info(f"💾 Выходной файл: '{args.output}'")

    session = get_http_session()

    # 1. Поиск всех страниц
    title_to_subcats = get_all_pages_recursive(
        category=args.category,
        wiki_domain=args.domain,
        session=session,
        max_depth=args.max_depth,
    )
    all_titles = list(title_to_subcats.keys())
    logger.info(f"✅ Найдено {len(all_titles)} уникальных статей для выгрузки.")

    # 2. Загрузка контента статей
    knowledge_base: List[Dict[str, str]] = []
    total = len(all_titles)

    for i, (title, subcats_set) in enumerate(title_to_subcats.items(), start=1):
        subcategory = next(iter(subcats_set)) if subcats_set else "Общее"
        if i % 50 == 0 or i == 1 or i == total:
            logger.info(f"[{i}/{total}] Загрузка: {title} ({subcategory})")

        content = get_page_content(title, args.domain, subcategory, session)
        if content:
            knowledge_base.append(content)

        time.sleep(args.delay)

    # 3. Сохранение
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    logger.info(f"🎉 Готово! Сохранено {len(knowledge_base)} статей в '{args.output}'")


if __name__ == "__main__":
    main()
