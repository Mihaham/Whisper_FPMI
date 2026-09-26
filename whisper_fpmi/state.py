from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from whisper_fpmi.paths import DATA_DIR, STATE_PATH


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"videos": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mark_done(
    state: dict[str, Any],
    video_id: str,
    *,
    title: str,
    url: str,
    slug: str,
    model: str,
) -> None:
    videos = state.setdefault("videos", {})
    videos[video_id] = {
        "title": title,
        "url": url,
        "slug": slug,
        "plain": f"result/{slug}.txt",
        "timed": f"download/{slug}.mp4.txt",
        "model": model,
        "status": "done",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
