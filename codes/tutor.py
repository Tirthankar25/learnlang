"""LeLa lesson controller. The tutor persona is AuTut, who speaks Kannada and English.

Pure Python (no torch / FastAPI) so it can be unit-tested without a GPU.

Split of responsibilities:
  * The controller owns the lesson plan (lessons/lessonN.json), the current step,
    attempt counts and the conversation history. It decides when to move on.
  * The LLM is the "agent" inside each step: it classifies the student's turn
    (answer / clarification question), judges it, and writes what AuTut says.
  * Guardrails in code keep a small LLM from derailing the lesson: a fuzzy
    match on the expected Kannada phrase can force "advance", attempts are
    capped, and a bad LLM reply falls back to a deterministic hint.

Lesson flow (see flow.py): every lesson runs as Concept Intro -> Guided Repetition ->
Micro-Roleplay -> Mastery Check, then a wrap-up. Each step carries its "phase"; AuTut announces
each new phase. The student can interrupt at any step, in English or Kannada, with a question:
AuTut answers it, then asks the current step again (the step and its attempt count are kept).
When the lesson ends AuTut asks: next lesson, repeat this lesson, or stop for now (END_CHOICES).
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from flow import PHASE_LABELS, PHASE_START, phase_of
from translit import has_kannada, transliterate

# LLM call signature: (messages) -> raw assistant text. Injected so tests can fake it.
LLMFn = Callable[[list[dict]], Awaitable[str]]
# English -> Kannada translator (e.g. Sarvam). Optional; without it the LLM's own Kannada is used.
TranslateFn = Callable[[str], Awaitable[str]]

ACTIONS = ("advance", "retry", "explain", "answer")
TUTOR = "AuTut"
HISTORY_TURNS = 12          # last N lines of dialogue sent to the LLM
FUZZY_PASS = 0.80           # similarity at which code forces "advance"
# Vetted praise AuTut speaks when the LLM's own Kannada isn't trusted (see llm_kannada below).
PRAISE = [("ಚೆನ್ನಾಗಿದೆ!", "Well done!"), ("ತುಂಬಾ ಒಳ್ಳೆಯದು!", "Very good!"), ("ಸರಿಯಾಗಿದೆ!", "That's right!")]

SYSTEM_PROMPT = """You are AuTut, a warm, patient tutor who speaks Kannada and English, in a spoken 1-on-1 Kannada lesson with a beginner whose first language is English.
The teacher leads: AuTut asks, the student answers. The lesson plan is fixed by the app; you handle ONE student turn at a time.

Each lesson has four parts: concept intro, guided repetition (the student repeats one phrase after AuTut), micro-roleplay (a short conversation) and mastery check (the student answers alone). You receive the current part as "phase".
You receive: the current lesson step (what AuTut just asked, the goal, an example answer), how many attempts the student has made, the recent conversation, and the student's latest turn. Speech recognition can be imperfect, so judge the intent, not the spelling.

Choose exactly one action:
- "advance": the turn is an acceptable answer to the current step (small grammar or pronunciation slips are fine; praise briefly).
- "retry": the student tried to answer the step but it is wrong, incomplete, or an English answer (e.g. "hello" or "I am fine" in English). Coach them: tell them exactly what to say, in Kannada with the English meaning.
- "explain": the student asked for help with THIS step, in English or in Kannada (e.g. "ಅರ್ಥ ಏನು?", "ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ") (meaning of a word, to repeat, to slow down, "what should I say?"). Answer the question briefly in English, give the Kannada phrase they need.
- "answer": the student asked any other question or made a remark that is not an attempt at the step (e.g. "When is the next class?", "Are you a real person?", "How do I say water?"). Answer it.

For "answer": say_en is AuTut's actual answer in plain, simple English, one or two short sentences, with no Kannada script, no romanization and no digits (write numbers as words), so it translates cleanly. say_kn is the same answer in Kannada. Be truthful: AuTut is an AI tutor in this app; there is no fixed class timetable, the student can start the next lesson whenever they like by picking it from the lesson list. If you do not know something, say so kindly.

Write AuTut's reply in two parallel fields (the app adds the English-letter pronunciation itself):
- "say_kn": what AuTut says aloud. Kannada script ONLY (no Latin letters, no digits), at most two short sentences.
- "say_en": English shown on screen: a translation of say_kn, plus any explanation.
Do NOT ask the next lesson question yourself; the app asks it after "advance".

Reply with JSON only:
{"action": "advance|retry|explain", "say_kn": "...", "say_en": "...", "note": "<one-line assessment for the log>"}"""

ANSWER_PROMPT = """You are AuTut, a warm Kannada and English tutor (an AI tutor in a lesson app) talking to a beginner whose first language is English.
The student just asked a question or made a remark in English. Answer it as AuTut.
Facts: there is no fixed class timetable; the student can start the next lesson whenever they like by picking it from the lesson list. If you do not know something, say so kindly.
- "say_en": the answer in plain, simple English, one or two short sentences, no Kannada script, no romanization, no digits (write numbers as words).
- "say_kn": the same answer in Kannada script.
Reply with JSON only:
{"action": "answer", "say_kn": "...", "say_en": "...", "note": "..."}"""

# Plain-text prompt for answering a side question. A free-text answer is far more reliable from a
# 4-bit 8B model than one squeezed into the JSON schema (that produced empty or junk answers).
ANSWER_TEXT_PROMPT = """You are AuTut, a friendly Kannada and English tutor (an AI tutor inside a lesson app). The student asked you something in English.
Reply with the answer only: one or two short, simple English sentences. Do not use Kannada script, romanization, digits (write numbers as words), lists or quotes.
Answer general-knowledge questions directly and correctly.
About classes: there is no fixed timetable. The student can start the next lesson whenever they like, for example tomorrow, by picking it from the lesson list."""

ANSWER_EXAMPLES = [
    ("When is the next class?", "There is no fixed timetable. You can start the next lesson whenever you like, even tomorrow."),
    ("What is the capital of Karnataka?", "The capital of Karnataka is Bengaluru."),
]


def clean_answer(text: str) -> str:
    """Tidy a free-text LLM answer: drop junk tokens and labels, keep at most two sentences."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = re.sub(r"^(AuTut|LeLa|Answer|Teacher)\s*:\s*", "", text, flags=re.I).strip(" \"'")
    text = re.sub(r"\S*_\S*", "", text)            # tokens like "can_hostname"
    text = text.replace('"', "")
    text = re.sub(r"[ಀ-೿]+", "", text)            # no Kannada script in the English line
    text = re.sub(r"\s+([.,!?])", r"\1", re.sub(r"\s+", " ", text)).strip()
    sentences = re.findall(r"[^.!?]+[.!?]?", text)
    return " ".join(x.strip() for x in sentences[:2] if x.strip())[:240]


_QUESTION = re.compile(r"^\s*(what|when|where|who|whom|whose|why|how|which|is|are|am|was|were|do|does|did|can|could|will|would|shall|should|may|might|have|has|tell me|please tell)\b", re.I)


# English help requests that really are about the current step ("what does X mean?", "repeat please").
_HELP = re.compile(r"\b(say|said|mean|means|meaning|repeat|again|slow|slower|pronounce|spell|what should|answer|reply)\b", re.I)
# "How do you say X?" / "What is X in Kannada?" -> X, so AuTut can speak X in Kannada.
_HOW_SAY = [
    re.compile(r"how (?:do|would|can|should|to) (?:you |i |we )?say,?\s+['\"]?(.+?)['\"]?\s*(?:in kannada)?[?.!]*$", re.I),
    re.compile(r"what(?:'s| is) ['\"]?(.+?)['\"]? in kannada[?.!]*$", re.I),
]


# Kannada interruptions asking for help with the current step: "what does it mean?", "say it again",
# "I didn't understand", "slowly", "how do I say it?". Only used when the turn is not a match.
_KN_HELP = re.compile(r"(ಅರ್ಥ|ಗೊತ್ತಾಗಲಿಲ್ಲ|ಗೊತ್ತಾಗ್ಲಿಲ್ಲ|ಗೊತ್ತಿಲ್ಲ|ತಿಳಿಯಲಿಲ್ಲ|ಮತ್ತೊಮ್ಮೆ|ಇನ್ನೊಮ್ಮೆ ಹೇಳಿ|ನಿಧಾನ|ಹೇಗೆ ಹೇಳ)")

# End of lesson: what the student wants next. Checked in this order ("ಮತ್ತೆ ಸಿಗೋಣ" is "see you", not "again").
END_CHOICES = ("next", "repeat", "stop")
_CHOICE_RX = [
    ("stop", re.compile(r"(ಸಾಕು|ನಿಲ್ಲಿಸ|ಮತ್ತೆ ಸಿಗೋಣ|ಮತ್ತೆ ಭೇಟಿಯಾಗೋಣ|ಹೋಗಿ ಬರ|ಬೇಡ|\b(stop|end|enough|quit|later|bye|goodbye|done for (now|today)|no more|finish)\b)", re.I)),
    ("repeat", re.compile(r"(ಮತ್ತೆ|ಮತ್ತೊಮ್ಮೆ|ಇನ್ನೊಮ್ಮೆ|ಪುನಃ|ಇದೇ ಪಾಠ|\b(repeat|again|redo|once more|same lesson|start over|this lesson)\b)", re.I)),
    ("next", re.compile(r"(ಮುಂದಿನ|ಮುಂದೆ|ಹೊಸ ಪಾಠ|\b(next|continue|move on|go on|new lesson)\b)", re.I)),
]
_ASKS_TO = re.compile(r"^\s*(can|could|shall|should|may|let's|let us|please|i want|i'd like|i would like)\b", re.I)
END_ASK = ("ಮುಂದಿನ ಪಾಠ ಶುರು ಮಾಡೋಣವೇ, ಈ ಪಾಠವನ್ನು ಮತ್ತೆ ಮಾಡೋಣವೇ, ಅಥವಾ ಈಗ ಸಾಕೇ?",
           "Shall we start the next lesson, repeat this lesson, or stop for now?")
END_ASK_LAST = ("ಈ ಪಾಠವನ್ನು ಮತ್ತೆ ಮಾಡೋಣವೇ, ಅಥವಾ ಈಗ ಸಾಕೇ?",
                "That was the last lesson. Shall we repeat this lesson, or stop for now?")
# The page starts nothing by itself: AuTut points to the button, which the page highlights.
END_REPLY = {
    "next": ("ಸರಿ! ಮುಂದಿನ ಪಾಠಕ್ಕೆ ಹೋಗಲು ಕೆಳಗಿನ ಬಟನ್ ಒತ್ತಿ.",
             "Great! Press the Next lesson button below to start it."),
    "repeat": ("ಸರಿ! ಈ ಪಾಠವನ್ನು ಮತ್ತೆ ಮಾಡಲು ಕೆಳಗಿನ ಬಟನ್ ಒತ್ತಿ.",
               "Sure! Press the Repeat this lesson button below to start again."),
    "stop": ("ಸರಿ, ಮತ್ತೆ ಸಿಗೋಣ! ಕೆಳಗಿನ ಬಟನ್ ಒತ್ತಿ.",
             "Okay, see you next time! Your progress is saved. Press Stop for now to go back to the lesson list."),
}
RESUME = ("ಸರಿ, ನಾವು ನಿಲ್ಲಿಸಿದ ಕಡೆಯಿಂದ ಮುಂದುವರಿಸೋಣ.", "Let's go back to where we left off.")

# A pause in what AuTut says: app.py speaks the pieces as separate clips with a short silence between.
PAUSE = "\n"

# Practice after a question about a Kannada word ("How do you pronounce chennagiddini?"):
# AuTut says it, pauses, asks the student to say it, checks it, then goes back to the lesson.
PRACTICE_ASK = ("ಈಗ ನೀವು ಹೇಳಿ.", "Now you say it after me:")
PRACTICE_OK = ("ಚೆನ್ನಾಗಿದೆ!", "Well done!")
PRACTICE_AGAIN = ("ಇನ್ನೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸಿ.", "Almost. Try once more:")
PRACTICE_SHOWN = ("ಪರವಾಗಿಲ್ಲ, ಅಭ್ಯಾಸ ಮಾಡುತ್ತಾ ಇರಿ.", "No problem, keep practising it.")
# A question about a Kannada word or phrase (pronunciation, meaning, how to say it).
_LANG_Q = re.compile(r"\b(pronounc\w*|say|said|saying|spell\w*|mean|means|meaning|word|phrase|kannada)\b", re.I)
# English acknowledgements ("Okay", "Got it"): not an answer to grade, just a nudge to answer the step.
_ACK = re.compile(r"^\s*(ok|okay|okey|k|alright|all right|sure|fine|got it|i see|understood|hmm+|right|cool|great|nice)[\s.!]*$", re.I)
_SKIP = re.compile(r"\b(skip|later|no|not now|continue|move on|go on|next)\b", re.I)
PRACTICE_PASS = 0.75     # similarity for the practice phrase (ASR on a single new word is noisy)


def end_choice(text: str, lang: str) -> Optional[str]:
    """"next" / "repeat" / "stop" from the student's reply at the end of a lesson, or None."""
    text = (text or "").strip()
    if not text or (lang == "en" and looks_like_question(text) and not _ASKS_TO.match(text)):
        return None  # "When is the next class?" is a question, not a choice
    for choice, rx in _CHOICE_RX:
        if rx.search(text):
            return choice
    return None


def _roman_key(text: str) -> str:
    """Letters only, lower case, in English letters: 'ಚೆನ್ನಾಗಿದ್ದೀನಿ' and 'Chenna gidini' become comparable."""
    if has_kannada(text):
        text = transliterate(text, ascii_only=True)
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", text).lower())


def roman_similarity(said: str, phrase_kn: str) -> float:
    """0..1: how close the student's words (any script) are to a Kannada phrase, by sound."""
    s, p = _roman_key(said), _roman_key(phrase_kn)
    if not s or not p:
        return 0.0
    windows = [s[i : i + len(p)] for i in range(max(1, len(s) - len(p) + 1))]
    return max(difflib.SequenceMatcher(None, w, p).ratio() for w in windows)


def lesson_phrase_in(text: str, lesson: dict) -> Optional[str]:
    """The lesson's Kannada phrase or word the student is asking about, written in English letters
    ("How do you pronounce chenna gidinai?" -> "ಚೆನ್ನಾಗಿದ್ದೀನಿ"), or None. Uses only the lesson's
    own hand-checked Kannada, never the LLM's."""
    from flow import key_phrase  # local: flow imports nothing from here, but keep the module light

    cands = set()
    for st in lesson["steps"]:
        phrases = list(st.get("expected_kn") or [])
        kp = key_phrase(st)
        if kp:
            phrases.append(kp[0])
        for ph in phrases:
            ph = kannada_only(ph).strip(" .,!?")
            if ph:
                cands.add(ph)
                cands.update(w.strip(" .,!?") for w in ph.split())
    text = re.sub(r"\bkannada\b", " ", text, flags=re.I)  # "kannada" itself sounds like ಕನ್ನ...
    best, best_key = None, (0.0, 0)
    for c in cands:
        n = len(_roman_key(c))
        if n < 5:                        # too short to tell apart from English words
            continue
        key = (round(roman_similarity(text, c), 2), len(c))
        if key[0] >= (0.8 if n >= 8 else 0.9) and key > best_key:
            best, best_key = c, key
    return best


def phrase_to_say(text: str) -> Optional[str]:
    for rx in _HOW_SAY:
        m = rx.search(text or "")
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def looks_like_question(text: str) -> bool:
    return bool(text) and (text.rstrip().endswith("?") or bool(_QUESTION.match(text)))


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
    """One thing AuTut says: shown on screen and (Kannada part) spoken."""
    kn: str
    roman: str
    en: str
    state: str
    action: str = "prompt"
    passed: Optional[bool] = None
    student: Optional[str] = None
    student_lang: str = "kn"
    lesson_done: bool = False
    phase: str = ""
    choices: Optional[list] = None      # set at the end of a lesson: what the student can pick
    choice: Optional[str] = None        # the student's pick ("next" / "repeat" / "stop")
    mastery: Optional[dict] = None      # {"right": k, "total": n} once the lesson is done

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
            "phase": self.phase,
            "phase_label": PHASE_LABELS.get(self.phase, ""),
            "choices": self.choices,
            "choice": self.choice,
            "mastery": self.mastery,
        }


@dataclass
class LessonSession:
    lesson: dict
    llm: LLMFn
    step_idx: int = 0
    attempts: int = 0
    history: list[str] = field(default_factory=list)
    done: bool = False
    # False: AuTut only SPEAKS hand-checked Kannada (lesson lines, hints, praise); the LLM still
    # judges each turn and writes the English explanations. An 8B model's Kannada is often
    # ungrammatical, which IndicF5 then reads out as garble. Turn on with a stronger model.
    llm_kannada: bool = False
    turns: int = 0
    # English questions are answered in Kannada: the LLM writes the answer in English and this
    # translates it (much better Kannada than Llama 8B writes). None: use the LLM's own Kannada.
    translate: Optional[TranslateFn] = None
    # Plain-text LLM call (no JSON schema) used to answer side questions. None: use llm + JSON.
    chat: Optional[LLMFn] = None
    # When the student answers in English: "repeat" (default) = AuTut says it in Kannada and asks
    # the student to say it in Kannada before moving on; "accept" = says it in Kannada and moves on.
    english_answers: str = "repeat"
    # How each step was passed in this run, for the progress card: "first_try" (no retries),
    # "passed" (after coaching, or answered in English) or "shown" (out of attempts, AuTut modeled it).
    results: dict = field(default_factory=dict)
    # Set by the app: is there a lesson after this one? (changes the end-of-lesson question)
    has_next: bool = True
    # The student's end-of-lesson pick, once made ("next" / "repeat" / "stop").
    choice: Optional[str] = None
    # A side practice in progress: {"kn": phrase, "tries": n}. The lesson step is kept meanwhile.
    practice: Optional[dict] = None

    @property
    def step(self) -> dict:
        return self.lesson["steps"][self.step_idx]

    def _log(self, who: str, text: str) -> None:
        self.history.append(f"{who}: {text}")
        del self.history[:-HISTORY_TURNS]

    @property
    def phase(self) -> str:
        return "end" if self.done else phase_of(self.step)

    def _phase_start(self) -> Optional[tuple[str, str]]:
        """AuTut's line when the current step opens a new part of the lesson, else None."""
        if self.step_idx == 0 or "phase" not in self.step:
            return None
        prev = phase_of(self.lesson["steps"][self.step_idx - 1])
        if prev == self.phase or (self.phase == "close" and prev != "mastery"):
            return None
        return PHASE_START.get(self.phase)

    def _prompt_turn(self, prefix: Optional[TutorTurn] = None, reprompt: bool = False,
                     announce: bool = False) -> TutorTurn:
        """AuTut asks the current step's question, optionally after praise/feedback.

        reprompt: after answering a side question, use the step's shorter "reprompt_kn" if it has
        one (the goodbye step would otherwise repeat its whole closing speech), after a short
        "back to the lesson" line. announce: the step was just reached, so name a new phase."""
        st = self.step
        kn, en = st["tutor_kn"], st["tutor_en"]
        if reprompt and "reprompt_kn" in st:
            kn, en = st["reprompt_kn"], st.get("reprompt_en", "")
        if reprompt and kn:
            kn, en = f"{RESUME[0]}{PAUSE}{kn}", f"{RESUME[1]}\n{en}"
        start = self._phase_start() if announce else None
        if start:
            kn, en = f"{start[0]} {kn}", f"{start[1]}\n{en}"
        turn = TutorTurn(kn, transliterate(kn), en, st["id"], phase=self.phase)
        if prefix:
            sep = PAUSE if reprompt else " "  # after an answer: pause before going back to the lesson
            turn = TutorTurn(
                f"{prefix.kn}{sep if prefix.kn else ''}{turn.kn}".strip(),
                f"{prefix.roman}{sep if prefix.roman else ''}{turn.roman}".strip(),
                f"{prefix.en}{sep if prefix.en else ''}{turn.en}".strip(),
                st["id"], prefix.action, prefix.passed, prefix.student, prefix.student_lang,
                phase=self.phase,
            )
        self._log(TUTOR, turn.kn)
        return turn

    def opening(self) -> TutorTurn:
        """The tutor starts the class with a greeting, or welcomes the student back mid-lesson."""
        if self.step_idx == 0:
            return self._prompt_turn()
        welcome = "ಮತ್ತೆ ಸ್ವಾಗತ!"
        return self._prompt_turn(prefix=TutorTurn(
            welcome, transliterate(welcome), "Welcome back! Let's continue where you left off.", self.step["id"]),
            announce=True)

    def _record(self, result: str) -> None:
        self.results[self.step["id"]] = result

    def mastery_score(self) -> Optional[dict]:
        """{"right": steps of the mastery check passed first time, "total": its steps}, or None."""
        ids = [st["id"] for st in self.lesson["steps"] if st.get("phase") == "mastery"]
        if not ids:
            return None
        return {"right": sum(1 for i in ids if self.results.get(i) == "first_try"), "total": len(ids)}

    def _end_question(self) -> tuple[str, str]:
        return END_ASK if self.has_next else END_ASK_LAST

    def _finish(self, feedback: TutorTurn) -> TutorTurn:
        """The last step was passed: close the lesson and ask what the student wants next."""
        self.done = True
        ask_kn, ask_en = self._end_question()
        score = self.mastery_score()
        result = f" Mastery check: {score['right']} of {score['total']} right first time." if score else ""
        feedback.kn = f"{feedback.kn} {ask_kn}".strip()
        feedback.roman = transliterate(feedback.kn)
        feedback.en = f"{feedback.en}{result}\n{ask_en}".strip()
        feedback.lesson_done, feedback.state, feedback.phase = True, "END", "end"
        feedback.choices = self._choices()
        feedback.mastery = score
        self._log(TUTOR, feedback.kn)
        return feedback

    def _choices(self) -> list:
        return [c for c in END_CHOICES if self.has_next or c != "next"]

    def _end_turn(self, student_text: str, lang: str) -> Optional[TutorTurn]:
        """After the lesson: the student's next / repeat / stop pick, or None if it wasn't one."""
        choice = end_choice(student_text, lang)
        if choice == "next" and not self.has_next:
            choice = None
        if not choice:
            return None
        self.choice = choice
        kn, en = END_REPLY[choice]
        print(f"[Turn] end of lesson heard='{student_text}' [{lang}] -> {choice}")
        self._log(TUTOR, kn)
        return TutorTurn(kn, transliterate(kn), en, "END", action="choice", student=student_text,
                         student_lang=lang, lesson_done=True, phase="end", choice=choice,
                         mastery=self.mastery_score())

    def _build_messages(self, student_text: str, lang: str, score: float) -> list[dict]:
        st = self.step
        example = (st.get("expected_kn") or [""])[0]
        context = {
            "lesson": self.lesson.get("title", ""),
            "phase": PHASE_LABELS.get(phase_of(st), ""),
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

    async def _answer_call(self, student_text: str) -> Optional[dict]:
        """Separate LLM call that only answers the student's question."""
        if self.chat:
            msgs = [{"role": "system", "content": ANSWER_TEXT_PROMPT}]
            for q, a in ANSWER_EXAMPLES:
                msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": student_text})
            raw = ""
            try:
                raw = await self.chat(msgs)
            except Exception as err:
                print(f"[LLM] answer call failed: {err}")
            answer = clean_answer(raw)
            if not answer:
                print(f"[LLM] unusable answer: {raw!r}")
                return None
            return {"say_en": answer, "say_kn": ""}
        context = {
            "lesson": self.lesson.get("title", ""),
            "lesson_finished": self.done,
            "current_step_en": "" if self.done else self.step["tutor_en"],
            "recent_conversation": self.history,
            "student_said": student_text,
        }
        try:
            return parse_llm_json(await self.llm([
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, indent=1)},
            ]))
        except Exception as err:
            print(f"[LLM] answer call failed: {err}")
            return None

    async def _translate(self, en: str) -> str:
        """English -> Kannada via the translator; "" when there is none or it fails."""
        if self.translate and en:
            try:
                return kannada_only(await self.translate(en))
            except Exception as err:
                print(f"[Translate] failed: {err}")
        return ""

    async def _kannada_answer(self, say_en: str, llm_kn: str) -> str:
        """Kannada to speak for an answer: translated from the English when possible."""
        return await self._translate(say_en) or llm_kn

    def _advance(self, feedback: TutorTurn, result: str = "passed") -> TutorTurn:
        """Move to the next step (or finish) after this feedback."""
        self._record(result)
        self.attempts = 0
        feedback.action, feedback.passed = "advance", True
        if self.step_idx + 1 >= len(self.lesson["steps"]):
            return self._finish(feedback)
        self.step_idx += 1
        return self._prompt_turn(prefix=feedback, announce=True)

    async def _english_answer_turn(self, student_text: str, lang: str, accepted: bool) -> TutorTurn:
        """The student answered the step in English: AuTut says it in Kannada for them."""
        st = self.step
        said = student_text.rstrip(".! ")
        kn = await self._translate(said)
        if not kn:  # no translator: fall back to the step's hand-checked hint
            kn, roman, en = st["hint_kn"], transliterate(st["hint_kn"]), st["hint_en"]
        else:
            roman = transliterate(kn)
            en = f'In Kannada, "{said}" is: {roman}'
        en = en.rstrip(". ")
        repeat = self.english_answers != "accept"
        self.attempts += 1
        if repeat and self.attempts < st.get("max_attempts", 3):
            if kn != st["hint_kn"]:
                kn, en = f"ಕನ್ನಡದಲ್ಲಿ ಹೀಗೆ ಹೇಳಿ: {kn}", f"{en}. Now say it in Kannada."
            else:
                en += "."
            print(f"[Turn] step={st['id']} english answer heard='{student_text}' -> repeat '{kn}'")
            self._log(TUTOR, kn)
            return TutorTurn(kn, transliterate(kn), en, st["id"], action="retry", passed=False,
                             student=student_text, student_lang=lang)
        # "accept" mode, a good answer, or out of attempts: model it in Kannada and move on.
        if accepted or not repeat:
            kn, en = f"ಚೆನ್ನಾಗಿದೆ! ಕನ್ನಡದಲ್ಲಿ: {kn}", f"Well done! {en}."
        else:
            kn, en = f"ಪರವಾಗಿಲ್ಲ. {kn}", f"No problem. {en}. Let's continue."
        print(f"[Turn] step={st['id']} english answer heard='{student_text}' -> advance '{kn}'")
        return self._advance(TutorTurn(kn, transliterate(kn), en, st["id"], student=student_text, student_lang=lang),
                             "passed" if (accepted or not repeat) else "shown")

    async def _answer_turn(self, student_text: str, lang: str, decision: Optional[dict]) -> TutorTurn:
        """Answer a side question in Kannada (English on screen), then carry on with the lesson."""
        if self.chat or not decision or not decision.get("say_en"):
            decision = await self._answer_call(student_text)  # plain-text call: better answers
        if not decision or not decision.get("say_en"):
            # The LLM gave nothing usable (see the [LLM] line above in the log).
            decision = {"say_kn": "ಕ್ಷಮಿಸಿ, ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಕೇಳಿ.", "say_en": "Sorry, please ask me again."}
        say_en = decision["say_en"]
        if lang == "en" and _LANG_Q.search(student_text):
            phrase = await self._phrase_asked(student_text)
            if phrase:
                return self._start_practice(phrase, say_en, student_text, lang)
        say_kn = await self._kannada_answer(say_en, decision.get("say_kn", ""))
        print(f"[Turn] answer heard='{student_text}' [{lang}] en='{say_en}' spoken='{say_kn}'")
        state = "END" if self.done else self.step["id"]
        feedback = TutorTurn(say_kn, transliterate(say_kn), say_en, state, action="answer",
                             student=student_text, student_lang=lang, lesson_done=self.done,
                             phase=self.phase)
        if self.done:
            # Answered; now the end-of-lesson question again.
            ask_kn, ask_en = self._end_question()
            feedback.kn = f"{feedback.kn}{PAUSE}{ask_kn}".strip()
            feedback.roman, feedback.en = transliterate(feedback.kn), f"{feedback.en}\n{ask_en}"
            feedback.choices = self._choices()
            self._log(TUTOR, feedback.kn)
            return feedback
        return self._prompt_turn(prefix=feedback, reprompt=True)

    async def _phrase_asked(self, student_text: str) -> str:
        """The Kannada the student asked about: a phrase of this lesson they wrote in English
        letters, else the translation of "How do you say <English>?", else ""."""
        phrase = lesson_phrase_in(student_text, self.lesson)
        if not phrase and phrase_to_say(student_text):
            phrase = await self._translate(phrase_to_say(student_text))
        return kannada_only(phrase or "").strip(" .,!?")

    def _start_practice(self, phrase: str, say_en: str, student_text: str, lang: str) -> TutorTurn:
        """Answer about a Kannada word, then (after a pause) ask the student to say it.
        The lesson step and its attempts stay as they were."""
        self.practice = {"kn": phrase, "tries": 0}
        roman = transliterate(phrase)
        kn = f"{phrase}.{PAUSE}{PRACTICE_ASK[0]} {phrase}"
        en = f"{say_en}\n{PRACTICE_ASK[1]} {roman}"
        print(f"[Turn] step={self.step['id']} question heard='{student_text}' -> practise '{phrase}'")
        self._log(TUTOR, kn)
        return TutorTurn(kn, transliterate(kn), en, "END" if self.done else self.step["id"], action="practice",
                         student=student_text, student_lang=lang, phase=self.phase, lesson_done=self.done)

    async def _practice_reply(self, student_text: str, lang: str) -> Optional[TutorTurn]:
        """The student's try at the practice phrase. None: it was a new question, handle it normally."""
        phrase = self.practice["kn"]
        if lang == "en" and looks_like_question(student_text):
            self.practice = None
            return None
        if lang == "en" and _ACK.match(student_text):
            # "Okay": not a try yet. Ask again without counting it.
            kn = f"{PRACTICE_ASK[0]} {phrase}"
            self._log(TUTOR, kn)
            return TutorTurn(kn, transliterate(kn), f"Please say it in Kannada: {transliterate(phrase)}",
                             "END" if self.done else self.step["id"], action="practice", student=student_text,
                             student_lang=lang, phase=self.phase, lesson_done=self.done)
        if lang == "en" and _SKIP.search(student_text):
            ok = ("", "")
        else:
            score = max(match_score(student_text, [phrase], "contains") if lang == "kn" else 0.0,
                        roman_similarity(student_text, phrase))
            print(f"[Turn] practice '{phrase}' heard='{student_text}' [{lang}] score={score:.2f}")
            if score >= PRACTICE_PASS:
                ok = PRACTICE_OK
            elif self.practice["tries"] == 0:
                self.practice["tries"] = 1
                kn = f"{PRACTICE_AGAIN[0]} {phrase}"
                self._log(TUTOR, kn)
                return TutorTurn(kn, transliterate(kn), f"{PRACTICE_AGAIN[1]} {transliterate(phrase)}",
                                 self.step["id"], action="practice", student=student_text,
                                 student_lang=lang, phase=self.phase)
            else:
                ok = PRACTICE_SHOWN
        self.practice = None
        feedback = TutorTurn(ok[0], transliterate(ok[0]) if ok[0] else "", ok[1], self.step["id"],
                             action="practice", student=student_text, student_lang=lang, phase=self.phase)
        if self.done:  # practised after the lesson: back to "next lesson, repeat, or stop?"
            ask_kn, ask_en = self._end_question()
            feedback.kn = f"{feedback.kn}{PAUSE if feedback.kn else ''}{ask_kn}"
            feedback.roman, feedback.en = transliterate(feedback.kn), f"{feedback.en}\n{ask_en}".strip()
            feedback.state, feedback.lesson_done, feedback.choices = "END", True, self._choices()
            self._log(TUTOR, feedback.kn)
            return feedback
        return self._prompt_turn(prefix=feedback, reprompt=True)

    async def _ready_turn(self, student_text: str, lang: str) -> TutorTurn:
        """Concept intro: a question is answered first; anything else means "ready", move on."""
        if lang == "en" and looks_like_question(student_text):
            if _HELP.search(student_text) and not phrase_to_say(student_text):
                return self._kannada_help_turn(student_text)  # "repeat please" / "what do I say?"
            return await self._answer_turn(student_text, lang, None)
        if lang == "kn" and _KN_HELP.search(student_text):
            return self._kannada_help_turn(student_text)
        kn = "ಸರಿ, ಶುರು ಮಾಡೋಣ!"
        print(f"[Turn] step={self.step['id']} heard='{student_text}' [{lang}] -> ready")
        return self._advance(TutorTurn(kn, transliterate(kn), "Great, let's begin!", self.step["id"],
                                       student=student_text, student_lang=lang, phase=self.phase),
                             "first_try")

    def _kannada_help_turn(self, student_text: str) -> TutorTurn:
        """The student asked for help with this step (often in Kannada): AuTut gives the hint
        slowly, in Kannada with English on screen, then asks the same question again.
        No attempt is used, and the lesson stays on this step."""
        st = self.step
        kn = "ಸರಿ, ನಿಧಾನವಾಗಿ ಹೇಳುತ್ತೇನೆ." + (f" {kannada_only(st['hint_kn'])}" if st.get("match") != "any" else "")
        en = f"Sure, slowly. {st['hint_en']}"
        print(f"[Turn] step={st['id']} help request heard='{student_text}' -> hint, then same question")
        feedback = TutorTurn(kn, transliterate(kn), en, st["id"], action="explain", student=student_text,
                             student_lang="kn" if has_kannada(student_text) else "en", phase=self.phase)
        return self._prompt_turn(prefix=feedback)

    async def handle_student(self, student_text: str, lang: str = "kn") -> TutorTurn:
        student_text = (student_text or "").strip()
        if self.done:
            self._log("Student", f"{student_text} [{lang}]")
            if self.practice:
                turn = await self._practice_reply(student_text, lang)
                if turn:
                    return turn
            picked = self._end_turn(student_text, lang)
            if picked:
                return picked
            if lang == "en" and student_text and looks_like_question(student_text):
                # The lesson is over, but a question still gets a real answer.
                return await self._answer_turn(student_text, lang, None)
            kn, en = self._end_question()
            self._log(TUTOR, kn)
            return TutorTurn(kn, transliterate(kn), f"{en} You can also tap a button below.", "END",
                             action="ask_choice", student=student_text, student_lang=lang,
                             lesson_done=True, phase="end", choices=self._choices())

        self._log("Student", f"{student_text} [{lang}]")
        if self.practice:
            turn = await self._practice_reply(student_text, lang)
            if turn:
                return turn
        st = self.step
        if st.get("match") == "any":
            return await self._ready_turn(student_text, lang)
        score = match_score(student_text, st.get("expected_kn", []), st.get("match", "contains")) if lang == "kn" else 0.0
        if lang == "kn" and score < FUZZY_PASS and _KN_HELP.search(student_text):
            return self._kannada_help_turn(student_text)
        if lang == "en" and _ACK.match(student_text):
            # "Okay" after an answer or a hint is not an attempt: ask the step again, nothing counted.
            print(f"[Turn] step={st['id']} acknowledgement heard='{student_text}' -> same question")
            return self._prompt_turn(prefix=TutorTurn("ಸರಿ.", "Sari.", "Okay. Now please answer in Kannada:",
                                                      st["id"], action="ack", student=student_text,
                                                      student_lang=lang, phase=self.phase))

        decision = None
        try:
            decision = parse_llm_json(await self.llm(self._build_messages(student_text, lang, score)))
        except Exception as err:  # network / server errors: keep the lesson going
            print(f"[LLM] call failed: {err}")
        if decision is None:
            decision = self._fallback(score)

        # Guardrails: a clear match always passes; a clarification never costs an attempt.
        action = decision["action"]
        if action == "answer":
            return await self._answer_turn(student_text, lang, decision)
        if lang == "en":
            if action == "explain" and not _HELP.search(student_text):
                # Not about this step (e.g. "When is the next class?"): give a real answer.
                return await self._answer_turn(student_text, lang, None)
            if action in ("advance", "retry") and looks_like_question(student_text):
                # An English question is never an answer: answer it instead of grading it.
                return await self._answer_turn(student_text, lang, None)
            if action in ("advance", "retry") and score < FUZZY_PASS:
                return await self._english_answer_turn(student_text, lang, accepted=(action == "advance"))
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
                # "How do you say I am fine?" / "How do I pronounce chennagiddini?": AuTut says the
                # phrase, pauses, has the student say it, then goes back to the step's question.
                phrase = await self._phrase_asked(student_text) if lang == "en" else ""
                if phrase:
                    return self._start_practice(phrase, decision["say_en"], student_text, lang)
                # Otherwise: the step's hint, then the same question again.
                decision["say_kn"] = kannada_only(st["hint_kn"]) if lang == "en" else ""
        print(f"[Turn] step={st['id']} heard='{student_text}' [{lang}] score={score:.2f} action={action} "
              f"llm_kn='{llm_kn}' spoken='{decision['say_kn']}'")

        say_kn = decision["say_kn"] if (decision["say_kn"] or action == "explain") else kannada_only(st["hint_kn"])
        feedback = TutorTurn(
            say_kn, transliterate(say_kn), decision["say_en"], st["id"],
            action=action, passed=(action == "advance"), student=student_text, student_lang=lang,
            phase=self.phase,
        )

        if action == "advance":
            self._record("first_try" if self.attempts == 0 else
                         "shown" if self.attempts >= st.get("max_attempts", 3) else "passed")
            self.attempts = 0
            if self.step_idx + 1 >= len(self.lesson["steps"]):
                return self._finish(feedback)
            self.step_idx += 1
            return self._prompt_turn(prefix=feedback, announce=True)

        if action == "explain":
            # Answer the question, then ask the same question again.
            return self._prompt_turn(prefix=feedback)

        self._log(TUTOR, feedback.kn)
        return feedback
