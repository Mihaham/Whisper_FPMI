# Лекторий ФПМИ

Тексты лекций [лектория ФПМИ](https://vkvideo.ru/@lectorium_fpmi) и инструменты, которые на этих текстах стоят: каталог, таблица по видео, облако слов, главы из описаний, граф терминов, поиск и ответы с цитатами.

В репозитории лежит **текст**, не видео. Исходники роликов скачиваются на время расшифровки и удаляются.

## Откуда что берётся

| Шаг | Результат |
|-----|-----------|
| [Идея](idea.md) | Зачем расшифровывать лекторий |
| [Расшифровка](pipeline.md) | VK → Whisper `large-v3` → `result/` и `download/` |
| [Датасет](dataset.md) | `data/lectures.csv`: длительность, лектор, главы, служебные слова |
| [Облако слов](wordcloud.md) | Картина частых лемм, обновляется перед пушем |
| [Таймкоды](timecodes.md) | Главы из описания VK, в том числе длиннее часа |
| [Граф терминов](graph.md) | Интерактивная карта слов по каждому курсу |
| [Поиск и RAG](search.md) | BM25 без модели и ответ цитатами |
| [Метрики](metrics.md) | Насколько поиск попадает в главу и в лекцию |

Каталог всех сохранённых лекций: [CATALOG.md](https://github.com/Mihaham/Whisper_FPMI/blob/main/CATALOG.md).

## Быстрый старт

```powershell
python -m pip install -r requirements.txt
python -m whisper_fpmi run          # докачать и расшифровать недостающее
python -m whisper_fpmi build        # таблица, облако, граф, индекс, метрики
python -m whisper_fpmi search "жорданова форма"
python -m whisper_fpmi serve        # поиск и граф на localhost
```

Документация этого сайта собирается отдельно:

```powershell
python -m pip install -r requirements-docs.txt
mkdocs serve
mkdocs build
```
