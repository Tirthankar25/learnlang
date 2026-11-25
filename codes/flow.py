"""AuTut's four-part lesson flow: Concept Intro -> Guided Repetition -> Micro-Roleplay -> Mastery Check.

Pure Python (no torch / FastAPI). load_lessons() (progress.py) runs build_flow() on every lesson
file, so LessonSession (tutor.py) and the progress cards both see one flat list of steps, each
tagged with its "phase". The lesson files themselves don't have to change:

  * concept  one "intro" step: AuTut names the lesson's key phrases (Kannada spoken, English
             letters + meaning on screen). Any reply that isn't a question moves on; questions
             are answered first.
  * repeat   one "rep_<id>" step per key phrase: "say after me", graded on that exact phrase.
  * roleplay the lesson's own dialogue steps (everything except "goodbye"), unchanged.
  * mastery  up to MASTERY_MAX "check_<id>" steps: roleplay questions asked again with fewer
             attempts and no "say after me".
  * close    the lesson's "goodbye" step. After it AuTut asks: next lesson, repeat, or stop.

Key phrases come from each roleplay step's model answer in hint_kn ("<phrase> ಎಂದು ಹೇಳಿ."); see
key_phrase(). Open-answer steps ("ನಿಮ್ಮ ಹೆಸರು ಹೇಳಿ.", match "llm") get no repetition step.

Optional fields in a lesson file to take control by hand:
  "key_phrases": [{"kn": "...", "en": "meaning"}]   replaces the derived phrase list (intro + repetition)
  "mastery": ["step_id", ...]                        which roleplay steps the mastery check re-asks
  "repeat_kn" on a step                              the phrase to repeat for that step
  any step with a "phase" field                     the file is already structured: used as is
"""

from __future__ import annotations

import re

from translit import transliterate

FLOW_VERSION = 2            # progress cards from before the flow resume at step 0
PHASES = ("concept", "repeat", "roleplay", "mastery", "close")
PHASE_LABELS = {
    "concept": "Concept intro",
    "repeat": "Guided repetition",
    "roleplay": "Micro-roleplay",
    "mastery": "Mastery check",
    "close": "Wrap-up",
}
MASTERY_MAX = 3

# What AuTut says when a new phase starts (spoken before the phase's first question).
PHASE_START = {
    "repeat": ("ಮೊದಲು ಅಭ್ಯಾಸ ಮಾಡೋಣ.", "Part 2 of 4, guided repetition: say each phrase after me."),
    "roleplay": ("ಈಗ ಒಂದು ಸಣ್ಣ ಸಂಭಾಷಣೆ ಮಾಡೋಣ.", "Part 3 of 4, micro-roleplay: now let's have a short conversation."),
    "mastery": ("ಈಗ ಒಂದು ಚಿಕ್ಕ ಪರೀಕ್ಷೆ. ನೀವೇ ಉತ್ತರ ಹೇಳಿ.", "Part 4 of 4, mastery check: answer on your own this time."),
    "close": ("ಪರೀಕ್ಷೆ ಮುಗಿಯಿತು.", "The check is done."),
}

# The intro's last line: shown with the Kannada word, but spoken in English only (the English voice
# says the word itself, so the Kannada voice does not cut in mid-sentence).
READY_LINE_EN = "Say ಸರಿ (sari) when you are ready"
READY_LINE_SPOKEN = "Say in Kannada the word, saari, when you are ready"
READY_KN = ["ಸರಿ", "ಹೌದು", "ಆಯ್ತು", "ಓಕೆ", "ಸಿದ್ಧ", "ರೆಡಿ", "ಶುರು ಮಾಡೋಣ"]
_SAY = re.compile(r"\s*ಎಂದು\s+(ಹೇಳಿ|ಕೇಳಿ)")
_LEAD = re.compile(r"^(ಉದಾಹರಣೆಗೆ|ಅಥವಾ)[,\s]*")
_GLOSS = re.compile(r"'(.+?)'(?=[\s.,;:)]|$)")
_KN_ONLY = re.compile(r"[^ಀ-೿\s.,!?]")


def _clean_kn(text: str) -> str:
    text = re.sub(r"\s+", " ", _KN_ONLY.sub(" ", text or ""))
    return re.sub(r"\s+([.,!?])", r"\1", text).strip()


def key_phrase(step: dict):
    """(phrase_kn, meaning_en) a roleplay step teaches, or None for an open answer.

    Order: the step's own "repeat_kn" if set; else the model answer in hint_kn ("<phrase> ಎಂದು
    ಹೇಳಿ." -> <phrase>, last sentence only, so "ಮುಂದಿನ ಸಂಖ್ಯೆ ನಾಲ್ಕು. ನಾಲ್ಕು" -> "ನಾಲ್ಕು"); else
    hint_kn itself when it is a model sentence rather than an instruction ("ನಿಮ್ಮ ಹೆಸರು ಹೇಳಿ.")."""
    hint = step.get("hint_kn", "")
    m = _SAY.search(hint)
    if step.get("repeat_kn"):
        phrase = step["repeat_kn"]
    elif m:
        phrase = _LEAD.sub("", re.split(r"\.\s+", hint[: m.start()].strip())[-1])
    elif step.get("match") == "llm" or not step.get("expected_kn") or re.search(r"(ಹೇಳಿ|ಕೇಳಿ)[.!]?\s*$", hint):
        return None
    else:
        phrase = hint
    phrase = _LEAD.sub("", _clean_kn(phrase)).rstrip(" .,!")
    if not phrase:
        return None
    hint_en = step.get("hint_en", "")
    at = hint_en.find(phrase)
    at = at if at >= 0 else hint_en.find(phrase.split()[0])
    gloss = _GLOSS.search(hint_en, max(at, 0)) or _GLOSS.search(hint_en)
    return phrase, (gloss.group(1).strip() if gloss else "")


def _phrases(lesson: dict, roleplay: list[dict]) -> list[tuple[str, str, dict]]:
    """[(kn, en, source roleplay step or {})], de-duplicated, in lesson order."""
    out, seen = [], set()
    if lesson.get("key_phrases"):
        for p in lesson["key_phrases"]:
            kn = _clean_kn(p["kn"]).rstrip(" .,!")
            if kn and kn not in seen:
                seen.add(kn)
                out.append((kn, p.get("en", ""), {}))
        return out
    for st in roleplay:
        kp = key_phrase(st)
        if kp and kp[0] not in seen:
            seen.add(kp[0])
            out.append((kp[0], kp[1], st))
    return out


def _intro_step(lesson: dict, phrases) -> dict:
    spoken = " ".join(f"{kn}." if not kn.endswith("?") else kn for kn, _, _ in phrases)
    kn = _clean_kn(f"ನಮಸ್ಕಾರ! ಇವತ್ತಿನ ಪಾಠದ ಮುಖ್ಯ ಮಾತುಗಳು. {spoken} ಸಿದ್ಧವೇ?")
    listed = "; ".join(f"{transliterate(p)}" + (f" = {en}" if en else "") for p, en, _ in phrases)
    en = (f"Part 1 of 4, concept intro. Today: {lesson.get('summary_en') or lesson.get('title', '')}\n"
          f"Key phrases: {listed}.\n{READY_LINE_EN}, or ask me anything in English or Kannada.")
    return {
        "id": "intro", "phase": "concept", "tutor_kn": kn, "tutor_en": en,
        "goal_en": "The student says they are ready (any reply that is not a question).",
        "expected_kn": READY_KN, "match": "any",
        "hint_kn": "ಸರಿ ಎಂದು ಹೇಳಿ.", "hint_en": "Say ಸರಿ (sari), 'okay', when you are ready.",
        "max_attempts": 1,
        "reprompt_kn": "ಸಿದ್ಧವೇ?", "reprompt_en": "Ready? Say ಸರಿ (sari) to begin.",
    }


def _repeat_step(kn: str, en: str, src: dict, n: int) -> dict:
    roman = transliterate(kn)
    meaning = f" ({en})" if en else ""
    sid = f"rep_{src['id']}" if src.get("id") else f"rep_{n}"
    return {
        "id": sid, "phase": "repeat",
        "tutor_kn": _clean_kn(f"ನನ್ನ ನಂತರ ಹೇಳಿ. {kn}"),
        "tutor_en": f"Say after me: {roman}{meaning}.",
        "goal_en": f"The student repeats the phrase {kn}. Only this phrase counts; small pronunciation slips are fine.",
        "expected_kn": [kn], "match": "contains", "phrase_en": en,
        "hint_kn": _clean_kn(f"{kn} ಎಂದು ಹೇಳಿ."),
        "hint_en": src.get("hint_en") or f"Say {kn} ({roman}){meaning}.",
        "max_attempts": 2,
        "reprompt_kn": _clean_kn(f"ನನ್ನ ನಂತರ ಹೇಳಿ. {kn}"), "reprompt_en": f"Say after me: {roman}.",
    }


def _mastery_steps(lesson: dict, roleplay: list[dict], phrases) -> list[dict]:
    by_id = {st["id"]: st for st in roleplay}
    if lesson.get("mastery"):
        picked = [by_id[i] for i in lesson["mastery"] if i in by_id]
    else:
        cands = [src for _, _, src in phrases if src]
        if len(cands) > MASTERY_MAX:  # spread over the lesson: first, middle(s), last
            idx = sorted({round(i * (len(cands) - 1) / (MASTERY_MAX - 1)) for i in range(MASTERY_MAX)})
            cands = [cands[i] for i in idx]
        picked = cands
    out = []
    for st in picked:
        check = dict(st, id=f"check_{st['id']}", phase="mastery", max_attempts=min(2, st.get("max_attempts", 3)))
        check.pop("reprompt_kn", None)
        check.pop("reprompt_en", None)
        out.append(check)
    return out


def build_flow(lesson: dict) -> list[dict]:
    """The lesson's steps as AuTut's four-part flow (see module docstring). Idempotent."""
    steps = lesson["steps"]
    if any("phase" in st for st in steps):
        return steps
    roleplay = [dict(st, phase="roleplay") for st in steps if st["id"] != "goodbye"]
    close = [dict(st, phase="close") for st in steps if st["id"] == "goodbye"]
    for st in close:  # after a question, AuTut must say what to do next, not just wait
        if not st.get("reprompt_kn"):
            st["reprompt_kn"] = "ಪಾಠ ಮುಗಿಸಲು ಧನ್ಯವಾದಗಳು ಎಂದು ಹೇಳಿ."
            st["reprompt_en"] = "To finish the lesson, say ಧನ್ಯವಾದಗಳು (dhanyavādagaḷu), 'thank you'."
    phrases = _phrases(lesson, roleplay)
    flow = [_intro_step(lesson, phrases)] if phrases else []
    flow += [_repeat_step(kn, en, src, i) for i, (kn, en, src) in enumerate(phrases)]
    flow += roleplay
    flow += _mastery_steps(lesson, roleplay, phrases)
    flow += close
    ids = [st["id"] for st in flow]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Lesson {lesson.get('id')}: step ids clash after building the flow: {ids}")
    return flow


def phase_of(step: dict) -> str:
    return step.get("phase", "roleplay")
