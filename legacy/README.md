# Устаревший пайплайн

Скрипты в этой папке **не используйте**. Они качали лекции с YouTube через `pytube`/`scrapetube` и расшифровывали моделью Whisper `medium`.

Актуальный код:

```text
python -m whisper_fpmi run
```

Документация — в корневом `README.md`.

| Файл | Зачем был |
|------|-----------|
| `main.py` | Цикл: список YouTube → скачать → whisper |
| `download.py` | `pytube` в максимальном разрешении в `high/` |
| `whisper.py` | CLI `whisper --model medium` |
| `cache.py` | JSONL-кэш названий |
| `timecodes.py` | Нарезка роликов по словам «друзья»/«товарищ» |
| `clip.py` | Разовый склей клипов MoviePy |
| `requirements.txt` | Старые зависимости, включая CUDA-колёса PyTorch |
