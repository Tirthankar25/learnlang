"""LeLa lesson controller.

Pure Python (no torch / FastAPI) so it can be unit-tested without a GPU.

Split of responsibilities:
  * The controller owns the lesson plan (lessons/lessonN.json), the current step,
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
# English -> Kannada translator (e.g. Sarvam). Optional; without it the LLM's own Kannada is used.
TranslateFn = Callable[[str], Awaitable[str]]

ACTIONS = ("advance", "retry", "explain", "answer")
HISTORY_TURNS = 12          # last N lines of dialogue sent to the LLM
FUZZY_PASS = 0.80           # similarity at which code forces "advance"
# Vetted praise LeLa speaks when the LLM's own Kannada isn't trusted (see llm_kannada below).
PRAISE = [("ಚೆನ್ನಾಗಿದೆ!", "Well done!"), ("ತುಂಬಾ ಒಳ್ಳೆಯದು!", "Very good!"), ("ಸರಿಯಾಗಿದೆ!", "That's right!")]

SYSTEM_PROMPT = """You are LeLa, a warm, patient Kannada tutor in a spoken 1-on-1 lesson with a beginner whose first language is English.
The teacher leads: LeLa asks, the student answers. The lesson plan is fixed by the app; you handle ONE student turn at a time.

You receive: the current lesson step (what LeLa just asked, the goal, an example answer), how many attempts the student has made, the recent conversation, and the student's latest turn. Speech recognition can be imperfect, so judge the intent, not the spelling.

Choose exactly one action:
- "advance": the turn is an acceptable answer to the current step (small grammar or pronunciation slips are fine; praise briefly).
- "retry": the student tried to answer the step but it is wrong, incomplete, or an English answer (e.g. "hello" or "I am fine" in English). Coach them: tell them exactly what to say, in Kannada with the English meaning.
- "explain": the student asked for help with THIS step (meaning of a word, to repeat, to slow down, "what should I say?"). Answer the question briefly in English, give the Kannada phrase they need.
- "answer": the student asked any other question or made a remark that is not an attempt at the step (e.g. "When is the next class?", "Are you a real person?", "How do I say water?"). Answer it.

For "answer": say_en is LeLa's actual answer in plain, simple English, one or two short sentences, with no Kannada script, no romanization and no digits (write numbers as words), so it translates cleanly. say_kn is the same answer in Kannada. Be truthful: LeLa is an AI tutor in this app; there is no fixed class timetable, the student can start the next lesson whenever they like by picking it from the lesson list. If you do not know something, say so kindly.

Write LeLa's reply in two parallel fields (the app adds the English-letter pronunciation itself):
- "say_kn": what LeLa says aloud. Kannada script ONLY (no Latin letters, no digits), at most two short sentences.
- "say_en": English shown on screen: a translation of say_kn, plus any explanation.
Do NOT ask the next lesson question yourself; the app asks it after "advance".

Reply with JSON only:
{"action": "advance|retry|explain", "say_kn": "...", "say_en": "...", "note": "<one-line assessment for the log>"}"""

ANSWER_PROMPT = """You are LeLa, a warm Kannada tutor (an AI tutor in a lesson app) talking to a beginner whose first language is English.
The student just asked a question or made a remark in English. Answer it as LeLa.
Facts: there is no fixed class timetable; the student can start the next lesson whenever they like by picking it from the lesson list. If you do not know something, say so kindly.
- "say_en": the answer in plain, simple English, one or two short sentences, no Kannada script, no romanization, no digits (write numbers as words).
- "say_kn": the same answer in Kannada script.
Reply with JSON only:
{"action": "answer", "say_kn": "...", "say_en": "...", "note": "..."}"""

# Plain-text prompt for answering a side question. A free-text answer is far more reliable from a
# 4-bit 8B model than one squeezed into the JSON schema (that produced empty or junk answers).
ANSWER_TEXT_PROMPT = """You are LeLa, a friendly Kannada tutor (an AI tutor inside a lesson app). The student asked you something in English.
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
    text = re.sub(r"^(LeLa|Answer|Teacher)\s*:\s*", "", text, flags=re.I).strip(" \"'")
    text = re.sub(r"\S*_\S*", "", text)            # tokens like "can_hostname"
    text = text.replace('"', "")
    text = re.sub(r"[ಀ-೿]+", "", text)            # no Kannada script in the English line
    text = re.sub(r"\s+([.,!?])", r"\1", re.sub(r"\s+", " ", text)).strip()
    sentences = re.findall(r"[^.!?]+[.!?]?", text)
    return " ".join(x.strip() for x in sentences[:2] if x.strip())[:240]


_QUESTION = re.compile(r"^\s*(what|when|where|who|whom|whose|why|how|which|is|are|am|was|were|do|does|did|can|could|will|would|shall|should|may|might|have|has|tell me|please tell)\b", re.I)


# English help requests that really are about the current step ("what does X mean?", "repeat please").
_HELP = re.compile(r"\b(say|said|mean|means|meaning|repeat|again|slow|slower|pronounce|spell|what should|answer|reply)\b", re.I)
# "How do you say X?" / "What is X in Kannada?" -> X, so LeLa can speak X in Kannada.
_HOW_SAY = [
    re.compile(r"how (?:do|would|can|should|to) (?:you |i |we )?say,?\s+['\"]?(.+?)['\"]?\s*(?:in kannada)?[?.!]*$", re.I),
    re.compile(r"what(?:'s| is) ['\"]?(.+?)['\"]? in kannada[?.!]*$", re.I),
]


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
    # English questions are answered in Kannada: the LLM writes the answer in English and this
    # translates it (much better Kannada than Llama 8B writes). None: use the LLM's own Kannada.
    translate: Optional[TranslateFn] = None
    # Plain-text LLM call (no JSON schema) used to answer side questions. None: use llm + JSON.
    chat: Optional[LLMFn] = None
    # When the student answers in English: "repeat" (default) = LeLa says it in Kannada and asks
    # the student to say it in Kannada before moving on; "accept" = says it in Kannada and moves on.
    english_answers: str = "repeat"
    # How each step was passed in this run, for the progress card: "first_try" (no retries),
    # "passed" (after coaching, or answered in English) or "shown" (out of attempts, LeLa modeled it).
    results: dict = field(default_factory=dict)

    @property
    def step(self) -> dict:
        return self.lesson["steps"][self.step_idx]

    def _log(self, who: str, text: str) -> None:
        self.history.append(f"{who}: {text}")
        del self.history[:-HISTORY_TURNS]

    def _prompt_turn(self, prefix: Optional[TutorTurn] = None, reprompt: bool = False) -> TutorTurn:
        """LeLa asks the current step's question, optionally after praise/feedback.

        reprompt: after answering a side question, use the step's shorter "reprompt_kn" if it has
        one (the goodbye step would otherwise repeat its whole closing speech)."""
        st = self.step
        kn, en = st["tutor_kn"], st["tutor_en"]
        if reprompt and "reprompt_kn" in st:
            kn, en = st["reprompt_kn"], st.get("reprompt_en", "")
        turn = TutorTurn(kn, transliterate(kn), en, st["id"])
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
        """The tutor starts the class with a greeting, or welcomes the student back mid-lesson."""
        if self.step_idx == 0:
            return self._prompt_turn()
        welcome = "ಮತ್ತೆ ಸ್ವಾಗತ!"
        return self._prompt_turn(prefix=TutorTurn(
            welcome, transliterate(welcome), "Welcome back! Let's continue where you left off.", self.step["id"]))

    def _record(self, result: str) -> None:
        self.results[self.step["id"]] = result

    def _build_messages(self, student_text: str, lang: str, score: float) -> list[dict]:
        st = self.step
        example = (st.get("expected_kn") or [""])[0]
        context = {
            "lesson": self.lesson.get("title", ""),
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
            self.done = True
            feedback.lesson_done = True
            feedback.state = "END"
            self._log("LeLa", feedback.kn)
            return feedback
        self.step_idx += 1
        return self._prompt_turn(prefix=feedback)

    async def _english_answer_turn(self, student_text: str, lang: str, accepted: bool) -> TutorTurn:
        """The student answered the step in English: LeLa says it in Kannada for them."""
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
            self._log("LeLa", kn)
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
        say_kn = await self._kannada_answer(say_en, decision.get("say_kn", ""))
        print(f"[Turn] answer heard='{student_text}' [{lang}] en='{say_en}' spoken='{say_kn}'")
        state = "END" if self.done else self.step["id"]
        feedback = TutorTurn(say_kn, transliterate(say_kn), say_en, state, action="answer",
                             student=student_text, student_lang=lang, lesson_done=self.done)
        if self.done:
            self._log("LeLa", feedback.kn)
            return feedback
        return self._prompt_turn(prefix=feedback, reprompt=True)

    async def handle_student(self, student_text: str, lang: str = "kn") -> TutorTurn:
        student_text = (student_text or "").strip()
        if self.done:
            if lang == "en" and student_text:
                # The lesson is over, but a question still gets a real answer.
                self._log("Student", f"{student_text} [{lang}]")
                return await self._answer_turn(student_text, lang, None)
            return TutorTurn("ಮತ್ತೆ ಸಿಗೋಣ!", transliterate("ಮತ್ತೆ ಸಿಗೋಣ!"), "See you next time!", "END",
                             student=student_text, student_lang=lang, lesson_done=True)

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
                # English explanation on screen; LeLa speaks the Kannada phrase asked for
                # ("How do you say I am fine?"), or the step's hint, then re-asks the question.
                phrase = phrase_to_say(student_text) if lang == "en" else None
                decision["say_kn"] = (await self._translate(phrase) if phrase else "") or (
                    kannada_only(st["hint_kn"]) if lang == "en" else "")
        print(f"[Turn] step={st['id']} heard='{student_text}' [{lang}] score={score:.2f} action={action} "
              f"llm_kn='{llm_kn}' spoken='{decision['say_kn']}'")

        say_kn = decision["say_kn"] if (decision["say_kn"] or action == "explain") else kannada_only(st["hint_kn"])
        feedback = TutorTurn(
            say_kn, transliterate(say_kn), decision["say_en"], st["id"],
            action=action, passed=(action == "advance"), student=student_text, student_lang=lang,
        )

        if action == "advance":
            self._record("first_try" if self.attempts == 0 else
                         "shown" if self.attempts >= st.get("max_attempts", 3) else "passed")
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
