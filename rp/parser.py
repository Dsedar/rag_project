import requests
from bs4 import BeautifulSoup, Comment
import html2text
import re
import json
import time
from typing import List, Dict, Set, Tuple

# ----------------------------
# 1. НАСТРОЙКИ
# ----------------------------
WIKI_DOMAIN = "vedmak.fandom.com"
ROOT_CATEGORY = "Категория:Ведьмак 3"
OUTPUT_FILE = "witcher3_knowledge_base.json"
DELAY = 0.5

# ----------------------------
# 2. ОЧИСТКА ТЕКСТА
# ----------------------------
def clean_postprocessing(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"''+'(.*?)''+'", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&mdash;", "—", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\.{2,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.M)
    text = re.sub(r"^[.,;:!?—\-—\s]+$", "", text, flags=re.M)
    text = re.sub(r"\.{2,}", "...", text)
    return text.strip()

# ----------------------------
# 3. ПАРСИНГ СТАТЬИ
# ----------------------------
def get_page_content(title: str, wiki_domain: str, subcategory: str) -> Dict[str, str]:
    url = f"https://{wiki_domain}/api.php"
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": True,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Ошибка запроса для '{title}': {e}")
        return None
    if "error" in data:
        print(f"⚠️ Пропущено (ошибка API): {title} — {data['error']['info']}")
        return None
    try:
        html_content = data["parse"]["text"]["*"]
    except KeyError:
        print(f"⚠️ Нет HTML для: {title}")
        return None
    soup = BeautifulSoup(html_content, "html.parser")
    # Удаление ненужных элементов
    for selector in ["script", "style", "nav", "footer", "sup", "table", "aside",
                     "img", "figcaption", ".mw-editsection", ".reference", ".navbox"]:
        for el in soup.select(selector):
            el.decompose()
    # Удаление комментариев
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    # Конвертация в текст
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.ignore_images = True
    text_maker.body_width = 0
    raw_text = text_maker.handle(str(soup))
    clean_text = clean_postprocessing(raw_text)
    return {
        "title": data["parse"]["title"],
        "text": clean_text,
        "url": f"https://{wiki_domain}/wiki/{title.replace(' ', '_')}",
        "subcategory": subcategory  # Используем реальную подкатегорию
    }

# ----------------------------
# 4. РЕКУРСИВНЫЙ ПАРСИНГ КАТЕГОРИЙ
# ----------------------------
def get_pages_from_category(category: str, wiki_domain: str, limit=500) -> Tuple[List[str], List[str]]:
    url = f"https://{wiki_domain}/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": limit,
        "format": "json"
    }
    pages = []
    subcats = []
    while True:
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        except Exception as e:
            print(f"Ошибка API для категории {category}: {e}")
            break
        for member in data["query"]["categorymembers"]:
            title = member["title"]
            ns = member["ns"]
            if ns == 14:  # Подкатегория
                subcats.append(title)
            elif ns == 0:  # Статья
                pages.append(title)
        if "continue" in data:
            params.update(data["continue"])
            time.sleep(DELAY)
        else:
            break
    return pages, subcats

def get_all_pages_recursive(category: str, wiki_domain: str, visited: Set[str] = None, max_depth=5, depth=0) -> Dict[str, Set[str]]:
    if visited is None:
        visited = set()
    if depth >= max_depth:
        return {}
    print(f"{'  ' * depth}🔍 Обработка категории: {category}")
    if category in visited:
        return {}
    visited.add(category)
    pages, subcats = get_pages_from_category(category, wiki_domain)

    # Теперь храним словарь: title -> set(subcategories)
    result = {}

    print(f"{'  ' * depth}Нашли {len(pages)} статей в категории: {category}")
    for page in pages:
        if page not in result:
            result[page] = set()
        result[page].add(category)  # Добавляем подкатегорию

    for subcat in subcats:
        if subcat not in visited:
            print(f"{'  ' * depth}Переходим к подкатегории: {subcat}")
            sub_result = get_all_pages_recursive(subcat, wiki_domain, visited, max_depth, depth + 1)
            for title, subcats_set in sub_result.items():
                if title not in result:
                    result[title] = set()
                result[title].update(subcats_set)

    return result


# ----------------------------
# 5. ОСНОВНОЙ ЗАПУСК
# ----------------------------
if __name__ == "__main__":
    print(f"🚀 Начинаем парсинг категории: {ROOT_CATEGORY}")

    title_to_subcats = get_all_pages_recursive(ROOT_CATEGORY, WIKI_DOMAIN)
    all_titles = list(title_to_subcats.keys())
    print(f"\n✅ Найдено {len(all_titles)} уникальных статей.\n")

    # Для отладки: выводим первые 20 статей с их подкатегориями
    print("📃 Содержимое title_to_subcats (первые 20):")
    for i, (title, subcats) in enumerate(list(title_to_subcats.items())[:20]):
        print(f"{i+1}. {title} — Подкатегории: {', '.join(subcats)}")

    # 2. Парсим каждую статью
    knowledge_base = []
    for i, (title, subcats_set) in enumerate(title_to_subcats.items()):
        subcategory = next(iter(subcats_set))  # Берём первую подкатегорию
        print(f"[{i+1}/{len(title_to_subcats)}] Парсинг: {title} (подкатегория: {subcategory})")
        content = get_page_content(title, WIKI_DOMAIN, subcategory)
        if content:
            knowledge_base.append(content)
        time.sleep(DELAY)

    # 3. Сохраняем результат
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 Готово! База знаний сохранена в '{OUTPUT_FILE}'")
    print(f"📦 {len(knowledge_base)} статей успешно обработано.")

