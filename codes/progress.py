"""Lessons on disk and each learner's progress card per lesson.

Pure Python (no torch / FastAPI) so it can be unit-tested without a GPU.

Lessons:  lessons/lesson1.json, lessons/lesson2.json, ...  The number in the file name sets the
          order; add a lesson by dropping in the next lessonN.json (same format) and restarting.
Progress: progress/<user>.json, one file per learner, holding one card per lesson id:
  {
    "status": "not_started" | "in_progress" | "completed",
    "step": 2,                 # where to resume (index into the lesson's steps)
    "total": 5,                # steps in the lesson
    "results": {"greet": "first_try", "how_are_you": "passed"},   # this run, see LessonSession.results
    "first_try": 1,            # steps passed first time in this run
    "best_first_try": 4,       # best over finished runs
    "completions": 1,          # how many times the lesson was finished
    "started_at": "...", "last_played": "...", "completed_at": "..."
  }
Learners identify themselves by name only (no password): fine for a family or classroom machine,
not for anything exposed to the internet.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone

_LESSON_FILE = re.compile(r"^lesson(\d+)\.json$")
_UNSAFE = re.compile(r"[^\w-]+", re.UNICODE)


def load_lessons(lesson_dir: str) -> list[dict]:
    """All lessonN.json files in lesson_dir, ordered by N. Adds "num" (N) to each lesson."""
    found = []
    for name in os.listdir(lesson_dir):
        m = _LESSON_FILE.match(name)
        if m:
            with open(os.path.join(lesson_dir, name), encoding="utf-8") as f:
                lesson = json.load(f)
            lesson["num"] = int(m.group(1))
            found.append(lesson)
    found.sort(key=lambda l: l["num"])
    ids = [l["id"] for l in found]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Two lesson files in {lesson_dir} share an id: {ids}")
    if not found:
        raise FileNotFoundError(f"No lesson1.json, lesson2.json, ... in {lesson_dir}")
    return found


def lesson_summary(lesson: dict) -> dict:
    """What the lesson picker shows."""
    return {"id": lesson["id"], "num": lesson["num"], "title": lesson.get("title", lesson["id"]),
            "summary": lesson.get("summary_en", ""), "steps": len(lesson["steps"])}


def user_key(name: str) -> str:
    """Display name -> file-safe key. 'Tirthankar B.' -> 'tirthankar_b'. Kannada letters are kept."""
    key = _UNSAFE.sub("_", (name or "").strip().lower()).strip("_")
    return key[:40]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_card(lesson: dict) -> dict:
    return {"status": "not_started", "step": 0, "total": len(lesson["steps"]), "results": {},
            "first_try": 0, "best_first_try": 0, "completions": 0,
            "started_at": None, "last_played": None, "completed_at": None}


class ProgressStore:
    """Reads and writes progress/<user>.json. One lock for all users: writes are tiny and rare."""

    def __init__(self, root: str, lessons: list[dict]):
        self.root = root
        self.lessons = {l["id"]: l for l in lessons}
        self.order = [l["id"] for l in lessons]
        self.lock = threading.Lock()
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.json")

    def _read(self, key: str) -> dict:
        try:
            with open(self._path(key), encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"lessons": {}}
        except (OSError, ValueError) as e:  # a damaged file must not lock the learner out
            print(f"[Progress] could not read {self._path(key)}: {e}; starting fresh")
            return {"lessons": {}}

    def _write(self, key: str, data: dict) -> None:
        os.makedirs(self.root, exist_ok=True)  # in case the folder was removed while running
        tmp = self._path(key) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._path(key))  # atomic: a crash never leaves half a file

    def cards(self, name: str) -> dict:
        """{lesson_id: card} for every lesson, in lesson order (not-started lessons included)."""
        key = user_key(name)
        with self.lock:
            saved = self._read(key)["lessons"] if key else {}
        out = {}
        for lid in self.order:
            card = empty_card(self.lessons[lid])
            card.update(saved.get(lid, {}))
            card["total"] = len(self.lessons[lid]["steps"])  # the lesson file may have changed
            card["step"] = min(card["step"], card["total"] - 1)
            out[lid] = card
        return out

    def next_lesson(self, name: str) -> str:
        """The lesson to suggest: the first one in progress, else the first never finished."""
        cards = self.cards(name)
        for status in ("in_progress", "not_started"):
            for lid in self.order:
                if cards[lid]["status"] == status:
                    return lid
        return self.order[0]

    def start(self, name: str, lesson_id: str, restart: bool = False) -> dict:
        """Card for a session that is starting. Resumes an unfinished run; a finished lesson
        (or restart=True) starts a new run at step 0, keeping completions and the best score."""
        card = self.cards(name)[lesson_id]
        if restart or card["status"] != "in_progress":
            card.update(step=0, results={}, first_try=0, started_at=_now())
        card["status"] = "in_progress"  # a replay of a finished lesson keeps its completions count
        card["last_played"] = _now()
        self._save(name, lesson_id, card)
        return card

    def update(self, name: str, lesson_id: str, session) -> dict:
        """Card after a turn of a LessonSession (tutor.py)."""
        card = self.cards(name)[lesson_id]
        card["results"] = dict(session.results)
        card["first_try"] = sum(1 for r in session.results.values() if r == "first_try")
        card["last_played"] = _now()
        if session.done:
            card["step"] = card["total"] - 1
            if card["status"] != "completed":  # count each finished run once
                card["status"] = "completed"
                card["completions"] += 1
                card["completed_at"] = card["last_played"]
                card["best_first_try"] = max(card["best_first_try"], card["first_try"])
        else:
            card["step"] = session.step_idx
        self._save(name, lesson_id, card)
        return card

    def _save(self, name: str, lesson_id: str, card: dict) -> None:
        key = user_key(name)
        if not key:
            return
        with self.lock:
            data = self._read(key)
            data["name"] = name.strip()
            data.setdefault("lessons", {})[lesson_id] = card
            self._write(key, data)
