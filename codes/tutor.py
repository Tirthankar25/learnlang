"""LeLa lesson controller.

Pure Python (no torch / FastAPI) so it can be unit-tested without a GPU.

Split of responsibilities:
  * The controller owns the lesson plan (lessons.json), the current step,
    attempt counts and the conversation history. It decides when to move on.
  * The LLM is the "agent" inside each step: it classifies the student's turn
    (answer / clarification question), judges it, and writes what LeLa says.
  * Guardrails in code keep a small LLM from derailing the lesson: a fuzzy
    match on the expected Kannada phrase can force "advance", attempts are
    capped, and a bad LLM reply falls back to a deterministic hint.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from translit import has_kannada, transliterate

# LLM call signature: (messages) -> raw assistant text. Injected so tests can fake it.
LLMFn = Callable[[list[dict]], Awaitable[str]]

ACTIONS = ("advance", "retry", "explain")
HISTORY_TURNS = 12          # last N lines of dialogue sent to the LLM
FUZZY_PASS = 0.80           # similarity at which code forces "advance"
# Vetted praise LeLa speaks when the LLM's own Kannada isn't trusted (see llm_kannada below).
PRAISE = [("ಚೆನ್ನಾಗಿದೆ!", "Well done!"), ("ತುಂಬಾ ಒಳ್ಳೆಯದು!", "Very good!"), ("ಸರಿಯಾಗಿದೆ!", "That's right!")]

SYSTEM_PROMPT = """You are LeLa, a warm, patient Kannada tutor in a spoken 1-on-1 lesson with a beginner whose first language is English.
The teacher leads: LeLa asks, the student answers. The lesson plan is fixed by the app; you handle ONE student turn at a time.

You receive: the current lesson step (what LeLa just asked, the goal, an example answer), how many attempts the student has made, the recent conversation, and the student's latest turn. Speech recognition can be imperfect, so judge the intent, not the spelling.

Choose exactly one action:
- "advance": the turn is an acceptable answer to the current step (small grammar or pronunciation slips are fine; praise briefly).
- "retry": the student tried to answer but it is wrong, incomplete, in English, or they stayed silent on the goal. Coach them: tell them exactly what to say, in Kannada with the English meaning.
- "explain": the student asked for help (meaning of a word, to repeat, to slow down, "what should I say?"), usually in English. Answer the question briefly in English, give the Kannada phrase they need.

Write LeLa's reply in two parallel fields (the app adds the English-letter pronunciation itself):
- "say_kn": what LeLa says aloud. Kannada script ONLY (no Latin letters, no digits), at most two short sentences.
- "say_en": English shown on screen: a translation of say_kn, plus any explanation.
Do NOT ask the next lesson question yourself; the app asks it after "advance".

Reply with JSON only:
{"action": "advance|retry|explain", "say_kn": "...", "say_en": "...", "note": "<one-line assessment for the log>"}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "say_kn": {"type": "string"},
        "say_en": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["action", "say_kn", "say_en"],
}


# ---------------------------------------------------------------- text helpers

_KN_KEEP = re.compile(r"[^ಀ-೿\s.,!?]")
_PUNCT = re.compile(r"[\s.,!?।'\"-]+")


def kannada_only(text: str) -> str:
    """Strip anything the Kannada TTS should not read (Latin, digits, markup)."""
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", _KN_KEEP.sub(" ", text))
    return re.sub(r"\s+([.,!?])", r"\1", text).strip()


def _norm(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFC", text or ""))


def match_score(student: str, expected: list[str], mode: str) -> float:
    """0..1 similarity of the student's Kannada to any expected phrase."""
    s = _norm(student)
    if not s or not expected or mode == "llm":
        return 0.0
    best = 0.0
    for exp in expected:
        e = _norm(exp)
        if not e:
            continue
        if mode == "prefix":
            cand = [s[: len(e)]]
        else:  # "contains": best window of the same length anywhere in the reply
            cand = [s[i : i + len(e)] for i in range(max(1, len(s) - len(e) + 1))]
        for c in cand:
            best = max(best, difflib.SequenceMatcher(None, c, e).ratio())
    return best


def parse_llm_json(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of the model output and validate it."""
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if data.get("action") not in ACTIONS:
        return None
    data["say_kn"] = kannada_only(data.get("say_kn", ""))
    data["say_en"] = str(data.get("say_en", "")).strip()
    return data


def student_display(text: Optional[str], lang: str) -> Optional[str]:
    """What the student sees under "You said": English letters for Kannada, English as is."""
    if not text:
        return text
    return transliterate(text) if lang == "kn" or has_kannada(text) else text


# ---------------------------------------------------------------- data types

@dataclass
class TutorTurn:
    """One thing LeLa says: shown on screen and (Kannada part) spoken."""
    kn: str
    roman: str
    en: str
    state: str
    action: str = "prompt"
    passed: Optional[bool] = None
    student: Optional[str] = None
    student_lang: str = "kn"
    lesson_done: bool = False

    def to_json(self) -> dict:
        return {
            "type": "tutor",
            "teacher_kannada": self.kn,
            "romanized": self.roman,
            "english_help": self.en,
            "state": self.state,
            "action": self.action,
            "passed": self.passed,
            "student_transcript": self.student,
            "student_display": student_display(self.student, self.student_lang),
            "lesson_done": self.lesson_done,
        }


@dataclass
class LessonSession:
    lesson: dict
    llm: LLMFn
    step_idx: int = 0
    attempts: int = 0
    history: list[str] = field(default_factory=list)
    done: bool = False
    # False: LeLa only SPEAKS hand-checked Kannada (lesson lines, hints, praise); the LLM still
    # judges each turn and writes the English explanations. An 8B model's Kannada is often
    # ungrammatical, which IndicF5 then reads out as garble. Turn on with a stronger model.
    llm_kannada: bool = False
    turns: int = 0

    @property
    def step(self) -> dict:
        return self.lesson["steps"][self.step_idx]

    def _log(self, who: str, text: str) -> None:
        self.history.append(f"{who}: {text}")
        del self.history[:-HISTORY_TURNS]

    def _prompt_turn(self, prefix: Optional[TutorTurn] = None) -> TutorTurn:
        """LeLa asks the current step's question, optionally after praise/feedback."""
        st = self.step
        turn = TutorTurn(st["tutor_kn"], transliterate(st["tutor_kn"]), st["tutor_en"], st["id"])
        if prefix:
            turn = TutorTurn(
                f"{prefix.kn} {turn.kn}".strip(),
                f"{prefix.roman} {turn.roman}".strip(),
                f"{prefix.en} {turn.en}".strip(),
                st["id"], prefix.action, prefix.passed, prefix.student, prefix.student_lang,
            )
        self._log("LeLa", turn.kn)
        return turn

    def opening(self) -> TutorTurn:
        """The tutor starts the class with a greeting."""
        return self._prompt_turn()

    def _build_messages(self, student_text: str, lang: str, score: float) -> list[dict]:
        st = self.step
        example = (st.get("expected_kn") or [""])[0]
        context = {
            "current_step": {
                "id": st["id"],
                "lela_asked_kn": st["tutor_kn"],
                "lela_asked_en": st["tutor_en"],
                "goal": st["goal_en"],
                "example_answer_kn": example,
                "hint_en": st["hint_en"],
            },
            "attempts_so_far": self.attempts,
            "max_attempts": st.get("max_attempts", 3),
            "recent_conversation": self.history,
            "student_turn": {
                "language": "English" if lang == "en" else "Kannada",
                "text": student_text,
            },
        }
        # Only meaningful when the step has fixed expected phrases; a 0 for an open answer
        # ("where are you from?") made the LLM reject correct replies.
        if st.get("expected_kn") and st.get("match") != "llm":
            context["student_turn"]["similarity_to_expected"] = round(score, 2)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, indent=1)},
        ]

    def _fallback(self, score: float) -> dict:
        """Deterministic reply when the LLM is down or returns junk."""
        st = self.step
        if score >= FUZZY_PASS:
            return {"action": "advance", "say_kn": "ಚೆನ್ನಾಗಿದೆ!", "say_en": "Well done!"}
        return {"action": "retry", "say_kn": st["hint_kn"], "say_en": st["hint_en"]}

    async def handle_student(self, student_text: str, lang: str = "kn") -> TutorTurn:
        if self.done:
            return TutorTurn("ಮತ್ತೆ ಸಿಗೋಣ!", transliterate("ಮತ್ತೆ ಸಿಗೋಣ!"), "See you next time!", "END", lesson_done=True)

        student_text = (student_text or "").strip()
        self._log("Student", f"{student_text} [{lang}]")
        st = self.step
        score = match_score(student_text, st.get("expected_kn", []), st.get("match", "contains")) if lang == "kn" else 0.0

        decision = None
        try:
            decision = parse_llm_json(await self.llm(self._build_messages(student_text, lang, score)))
        except Exception as err:  # network / server errors: keep the lesson going
            print(f"[LLM] call failed: {err}")
        if decision is None:
            decision = self._fallback(score)

        # Guardrails: a clear match always passes; a clarification never costs an attempt.
        action = decision["action"]
        if score >= FUZZY_PASS and action == "retry":
            action = "advance"
            decision.update(say_kn="ಚೆನ್ನಾಗಿದೆ!", say_en="Well done!")
        if action == "retry":
            self.attempts += 1
            if self.attempts >= st.get("max_attempts", 3):
                # Model the answer and move on rather than looping forever.
                action = "advance"
                decision.update(
                    say_kn=f"ಪರವಾಗಿಲ್ಲ. {st['hint_kn']}",
                    say_en=f"No problem. {st['hint_en']} Let's continue.",
                )

        llm_kn = decision["say_kn"]
        if not self.llm_kannada:
            self.turns += 1
            if action == "advance" and not decision["say_kn"].startswith("ಪರವಾಗಿಲ್ಲ"):
                decision["say_kn"], decision["say_en"] = PRAISE[self.turns % len(PRAISE)]
            elif action == "retry":
                decision["say_kn"], decision["say_en"] = st["hint_kn"], st["hint_en"]
            elif action == "explain":
                decision["say_kn"] = ""  # English explanation on screen, then the question is re-asked
        print(f"[Turn] step={st['id']} heard='{student_text}' [{lang}] score={score:.2f} action={action} "
              f"llm_kn='{llm_kn}' spoken='{decision['say_kn']}'")

        say_kn = decision["say_kn"] if (decision["say_kn"] or action == "explain") else kannada_only(st["hint_kn"])
        feedback = TutorTurn(
            say_kn, transliterate(say_kn), decision["say_en"], st["id"],
            action=action, passed=(action == "advance"), student=student_text, student_lang=lang,
        )

        if action == "advance":
            self.attempts = 0
            if self.step_idx + 1 >= len(self.lesson["steps"]):
                self.done = True
                feedback.lesson_done = True
                feedback.state = "END"
                self._log("LeLa", feedback.kn)
                return feedback
            self.step_idx += 1
            return self._prompt_turn(prefix=feedback)

        if action == "explain":
            # Answer the question, then ask the same question again.
            return self._prompt_turn(prefix=feedback)

        self._log("LeLa", feedback.kn)
        return feedback
