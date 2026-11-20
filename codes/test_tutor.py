import asyncio, json
from tutor import LessonSession, match_score, kannada_only, parse_llm_json
L = json.load(open("lessons/lesson1.json", encoding="utf-8"))
def fake(replies):
    it = iter(replies)
    async def llm(msgs):
        r = next(it)
        if isinstance(r, Exception): raise r
        return r
    return llm
J = lambda a, kn="ಸರಿ", en="ok": json.dumps({"action": a, "say_kn": kn, "say_roman": "x", "say_en": en})

async def main():
    # 1. Student ignores greeting -> coached; then greets -> advance even if LLM says retry (fuzzy guardrail)
    s = LessonSession(L, fake([J("retry", "ನಮಸ್ಕಾರ ಎಂದು ಹೇಳಿ", "Say namaskara"), J("retry")]))
    t = s.opening(); print("open:", t.kn)
    t = await s.handle_student("hello", "en"); print("1:", t.action, t.passed, s.step["id"], s.attempts)
    assert t.action == "retry" and s.step["id"] == "greet"
    t = await s.handle_student("ನಮಸ್ಕಾರ", "kn"); print("2:", t.action, t.kn)
    assert t.action == "advance" and s.step["id"] == "how_are_you" and "ಹೇಗಿದ್ದೀರಿ" in t.kn
    # 2. Clarify in English -> explain + re-ask, no attempt used
    s2 = LessonSession(L, fake([J("explain", "ಹೇಗಿದ್ದೀರಿ ಅಂದರೆ how are you", "It means how are you")])); s2.step_idx = 1
    t = await s2.handle_student("what does hegiddiri mean?", "en"); print("3:", t.action, s2.attempts, t.kn)
    assert t.action == "explain" and s2.attempts == 0 and t.kn.endswith("ನೀವು ಹೇಗಿದ್ದೀರಿ?")
    # 3. LLM down + junk -> deterministic fallback; attempts capped -> model answer & advance
    s3 = LessonSession(L, fake([RuntimeError("down"), "not json", J("retry")]))
    for i in range(3):
        t = await s3.handle_student("ಏನೋ", "kn"); print(f"4.{i}:", t.action, s3.step["id"], t.en[:50])
    assert s3.step["id"] == "how_are_you"
    # 4. Finish the lesson
    s4 = LessonSession(L, fake([J("advance")] * 10)); s4.step_idx = 4
    t = await s4.handle_student("ಧನ್ಯವಾದಗಳು", "kn"); print("5:", t.state, t.lesson_done)
    assert t.lesson_done and s4.done
    # helpers
    print("prefix:", round(match_score("ನನ್ನ ಹೆಸರು ತೀರ್ಥಂಕರ್", ["ನನ್ನ ಹೆಸರು"], "prefix"), 2))
    print("asr-noisy:", round(match_score("ನಮಸ್ಕರ", ["ನಮಸ್ಕಾರ"], "contains"), 2))
    print("sanitize:", kannada_only("ನಮಸ್ಕಾರ (Namaskara) 123!"))
    print("fenced:", parse_llm_json("```json\n" + J("advance") + "\n```")["action"])
    print("history sent:", len(s.history), s.history[-2:])
asyncio.run(main()); print("ALL OK")

# Student transcript display: Kannada -> English letters, English unchanged.
from tutor import student_display
assert student_display("ನನ್ನ ಹೆಸರು ತೀರ್ಥಂಕರ್", "kn") == "Nanna hesaru tīrthankar"
assert student_display("what does hegiddiri mean?", "en") == "what does hegiddiri mean?"
assert student_display("ನಮಸ್ಕಾರ", "en") == "Namaskāra"  # typed Kannada is detected too
print("display OK")

# Vetted speech (default): the LLM's own Kannada is never spoken.
async def vetted():
    junk = json.dumps({"action": "retry", "say_kn": "ಕಗಡತ ಪಮಲ", "say_en": "x"})
    s = LessonSession(L, fake([junk, json.dumps({"action": "advance", "say_kn": "ಕಗಡತ", "say_en": "x"})]))
    t = await s.handle_student("ಏನೋ", "kn")
    assert "ಕಗಡತ" not in t.kn and t.kn == L["steps"][0]["hint_kn"], t.kn
    t = await s.handle_student("ನಮಸ್ಕಾರ", "kn")
    assert "ಕಗಡತ" not in t.kn and t.kn.endswith(L["steps"][1]["tutor_kn"]), t.kn
    s2 = LessonSession(L, fake([junk]), llm_kannada=True)
    t = await s2.handle_student("ಏನೋ", "kn")
    assert t.kn == "ಕಗಡತ ಪಮಲ"
asyncio.run(vetted()); print("vetted OK")

# "Where are you from?" accepts a correct answer even if the LLM says retry.
async def where_from():
    s = LessonSession(L, fake([json.dumps({"action": "retry", "say_kn": "x", "say_en": "x"})]))
    s.step_idx = [st["id"] for st in L["steps"]].index("where_from")
    t = await s.handle_student("ನಾನು ಕೋಲ್ಕತ ಇಂದ ಬಂದಿದ್ದೇನೆ", "kn")
    assert t.passed and s.step["id"] == "goodbye", (t.action, s.step["id"])
asyncio.run(where_from()); print("where_from OK")

# English questions get a real answer in Kannada, and the lesson step is kept.
async def english_questions():
    ans = json.dumps({"action": "answer", "say_kn": "ಎಲ್ಎಲ್ಎಂ", "say_en": "You can start the next class any time."})
    async def tr(text):
        assert text == "You can start the next class any time."
        return "ನೀವು ಮುಂದಿನ ತರಗತಿಯನ್ನು ಯಾವಾಗ ಬೇಕಾದರೂ ಪ್ರಾರಂಭಿಸಬಹುದು."
    goodbye = [st["id"] for st in L["steps"]].index("goodbye")
    # a. At the goodbye step: answered in Kannada (translated), no repeat of the closing speech, step kept.
    s = LessonSession(L, fake([ans]), translate=tr); s.step_idx = goodbye
    t = await s.handle_student("When is the next class?", "en")
    assert t.action == "answer" and t.kn.startswith("ನೀವು ಮುಂದಿನ") and "ಮತ್ತೆ ಸಿಗೋಣ" not in t.kn, t.kn
    assert t.to_json()["student_display"] == "When is the next class?"
    assert s.step["id"] == "goodbye" and not s.done and s.attempts == 0
    # b. Mid-lesson: answer, then the current question is asked again.
    s = LessonSession(L, fake([ans]), translate=tr); s.step_idx = 1
    t = await s.handle_student("When is the next class?", "en")
    assert t.kn.endswith(L["steps"][1]["tutor_kn"]) and s.step_idx == 1, t.kn
    # c. LLM misfiles the question as "retry": a second call answers it instead of drilling.
    s = LessonSession(L, fake([J("retry"), ans]), translate=tr); s.step_idx = goodbye
    t = await s.handle_student("When is the next class?", "en")
    assert t.action == "answer" and s.attempts == 0, (t.action, s.attempts)
    # d. Lesson already over: still answers instead of repeating "ಮತ್ತೆ ಸಿಗೋಣ".
    s = LessonSession(L, fake([ans]), translate=tr); s.done = True
    t = await s.handle_student("When is the next class?", "en")
    assert t.action == "answer" and t.lesson_done and "ಮತ್ತೆ ಸಿಗೋಣ" not in t.kn, t.kn
    # e. Translator down: falls back to the LLM's own Kannada.
    async def broken(text): raise RuntimeError("offline")
    s = LessonSession(L, fake([ans]), translate=broken); s.step_idx = goodbye
    t = await s.handle_student("When is the next class?", "en")
    assert t.kn == "ಎಲ್ಎಲ್ಎಂ", t.kn
    # f. An English *answer* without a translator is coached with the step's hint.
    s = LessonSession(L, fake([J("retry")]))
    t = await s.handle_student("hello", "en")
    assert t.action == "retry" and s.attempts == 1 and t.kn == L["steps"][0]["hint_kn"]
asyncio.run(english_questions()); print("english questions OK")

# English answers to the step are said back in Kannada (replays of a real session log).
async def english_answers():
    KN = {"I am fine": "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ.", "My name is Tirthankar": "ನನ್ನ ಹೆಸರು ತೀರ್ಥಂಕರ್.",
          "I am fine?": "x", "When is the next class?": "x", "I am fine in Kannada": "x",
          "You can start the next lesson whenever you like.": "ನೀವು ಯಾವಾಗ ಬೇಕಾದರೂ ಮುಂದಿನ ಪಾಠವನ್ನು ಪ್ರಾರಂಭಿಸಬಹುದು."}
    async def tr(text): return KN.get(text, "ಅನುವಾದ")
    adv = json.dumps({"action": "advance", "say_kn": "ನಿಮ್ಮ ಹೆಸರು", "say_en": "Good"})
    # a. "I am fine." (LLM says advance): LeLa says it in Kannada, asks to repeat, step kept.
    s = LessonSession(L, fake([adv, J("advance")]), translate=tr); s.step_idx = 1
    t = await s.handle_student("I am fine.", "en")
    assert t.action == "retry" and s.step["id"] == "how_are_you", (t.action, s.step["id"])
    assert t.kn == "ಕನ್ನಡದಲ್ಲಿ ಹೀಗೆ ಹೇಳಿ: ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ." and "I am fine" in t.en, (t.kn, t.en)
    # ...then saying it in Kannada moves on.
    t = await s.handle_student("ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ", "kn")
    assert t.passed and s.step["id"] == "name"
    # b. accept mode: said back in Kannada and the lesson moves on.
    s = LessonSession(L, fake([adv]), translate=tr, english_answers="accept"); s.step_idx = 2
    t = await s.handle_student("My name is Tirthankar.", "en")
    assert t.passed and s.step["id"] == "where_from" and "ನನ್ನ ಹೆಸರು ತೀರ್ಥಂಕರ್" in t.kn, t.kn
    # c. Answering in English every time still ends after max_attempts (no endless loop).
    s = LessonSession(L, fake([adv] * 3), translate=tr); s.step_idx = 1
    for _ in range(3):
        t = await s.handle_student("I am fine.", "en")
    assert s.step["id"] == "name" and t.kn.startswith("ಚೆನ್ನಾಗಿದೆ! ಕನ್ನಡದಲ್ಲಿ: ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ"), t.kn
    # d. "How do you say I am fine?" (explain): LeLa speaks the phrase in Kannada, then re-asks.
    ex = json.dumps({"action": "explain", "say_kn": "", "say_en": "Say nanu chennagiddene."})
    s = LessonSession(L, fake([ex]), translate=tr); s.step_idx = 1
    t = await s.handle_student("How do you say I am fine?", "en")
    assert t.kn == "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ.\nಈಗ ನೀವು ಹೇಳಿ. ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ" and t.action == "practice", t.kn
    t = await s.handle_student("ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ", "kn")  # the practice try; then back to the question
    assert t.kn.startswith("ಚೆನ್ನಾಗಿದೆ!") and t.kn.endswith("ನೀವು ಹೇಗಿದ್ದೀರಿ?") and s.attempts == 0, t.kn
    # e. "When is the next class?" misfiled as explain: still a real answer in Kannada.
    ans = json.dumps({"action": "answer", "say_kn": "", "say_en": "You can start the next lesson whenever you like."})
    s = LessonSession(L, fake([ex, ans]), translate=tr); s.step_idx = 4
    t = await s.handle_student("When is the next class?", "en")
    assert t.action == "answer" and t.kn.startswith("ನೀವು ಯಾವಾಗ"), (t.action, t.kn)
asyncio.run(english_answers()); print("english answers OK")

# Side questions use the plain-text chat call (not JSON) and get a real answer.
async def plain_answers():
    seen = []
    async def chat(msgs):
        seen.append(msgs[-1]["content"])
        return "LeLa: The capital of West Bengal is Kolkata."
    async def tr(text):
        assert text == "The capital of West Bengal is Kolkata.", text
        return "ಪಶ್ಚಿಮ ಬಂಗಾಳದ ರಾಜಧಾನಿ ಕೋಲ್ಕತ್ತಾ."
    ans = json.dumps({"action": "answer", "say_kn": "", "say_en": ""})
    s = LessonSession(L, fake([ans]), translate=tr, chat=chat); s.step_idx = 3
    t = await s.handle_student("What is the state capital of West Bengal?", "en")
    assert seen == ["What is the state capital of West Bengal?"]
    assert t.kn.startswith("ಪಶ್ಚಿಮ ಬಂಗಾಳದ") and t.kn.endswith(L["steps"][3]["tutor_kn"]) and s.step_idx == 3, t.kn
    # Empty / failing chat: a "please ask again", not a false "I don't know".
    async def empty(msgs): return "  "
    s = LessonSession(L, fake([ans]), translate=tr, chat=empty); s.done = True
    t = await s.handle_student("Do we have any class tomorrow?", "en")
    assert "ಮತ್ತೊಮ್ಮೆ ಕೇಳಿ" in t.kn, t.kn
asyncio.run(plain_answers()); print("plain answers OK")

# Lessons are lessons/lesson1.json, lesson2.json, ... and every learner has a progress card per lesson.
import os, tempfile
from progress import ProgressStore, load_lessons, user_key
from translit import has_kannada
LS = load_lessons("lessons")
NUMS = ["1", "1a", "1b", "1c", "1d", "1e", "2", "2a", "2b", "2c", "2d", "2e", "3", "3a", "3b", "3c", "3d", "3e",
        "4a", "4b", "4c", "4d", "4e", "5a", "5b", "5c", "5d"]
assert [l["num"] for l in LS] == NUMS and LS[0]["id"] == L["id"], [l["num"] for l in LS]
assert LS[1]["id"] == "lesson_01a_introduction"
for l in LS:  # every lesson file has what LessonSession needs
    for st in l["steps"]:
        for k in ("id", "tutor_kn", "tutor_en", "goal_en", "hint_kn", "hint_en"):
            assert st.get(k), (l["id"], st.get("id"), k)
        assert has_kannada(st["tutor_kn"]) and kannada_only(st["tutor_kn"]) == st["tutor_kn"], st["tutor_kn"]
assert user_key("  Tirthankar B. ") == "tirthankar_b" and user_key("../..") == ""

async def progress_cards():
    with tempfile.TemporaryDirectory() as d:
        store = ProgressStore(d, LS)
        lid = LS[0]["id"]
        assert store.next_lesson("Ana") == lid and store.cards("Ana")[lid]["status"] == "not_started"
        card = store.start("Ana", lid)
        assert LS[0]["steps"][0]["id"] == "intro" and LS[0]["steps"][1]["id"] == "rep_greet"
        s = LessonSession(LS[0], fake([J("advance"), J("retry"), J("retry")]), step_idx=card["step"])
        await s.handle_student("ಸರಿ", "kn")                           # intro: ready (no LLM call)
        await s.handle_student("ನಮಸ್ಕಾರ", "kn")                      # repeat greet: first try
        await s.handle_student("ಏನೋ", "kn")                           # repeat how_are_you: retry ...
        await s.handle_student("ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ", "kn")           # ... then passed
        card = store.update("Ana", lid, s)
        assert card["status"] == "in_progress" and card["step"] == 3 and card["first_try"] == 2, card
        assert card["results"] == {"intro": "first_try", "rep_greet": "first_try", "rep_how_are_you": "passed"}, card["results"]
        # Another learner is untouched.
        assert store.cards("Bo")[lid]["status"] == "not_started"
        # Reconnect: resumes at the saved step with a welcome back.
        card = store.start("Ana", lid)
        s = LessonSession(LS[0], fake([J("advance")] * 30), step_idx=card["step"], results=dict(card["results"]))
        t = s.opening()
        assert card["step"] == 3 and t.kn.startswith("ಮತ್ತೆ ಸ್ವಾಗತ!") and t.kn.endswith(LS[0]["steps"][3]["tutor_kn"]), t.kn
        while not s.done:
            await s.handle_student(s.step["expected_kn"][0], "kn")
            card = store.update("Ana", lid, s)
        card = store.update("Ana", lid, s)  # extra turns after the end don't count twice
        assert card["status"] == "completed" and card["completions"] == 1 and len(card["results"]) == card["total"], card
        assert card["mastery"] == {"right": 3, "total": 3} and card["best_mastery"] == 3, card
        assert store.next_lesson("Ana") == LS[1]["id"]
        # Replaying a finished lesson starts at step 0 and keeps the completion count.
        card = store.start("Ana", lid)
        assert card["step"] == 0 and card["results"] == {} and card["completions"] == 1
        # Survives a restart of the server (new store, same folder).
        assert ProgressStore(d, LS).cards("ana")[lid]["completions"] == 1
        # Reset: this lesson back to zero (no completions, no bests); other lessons untouched.
        other = LS[1]["id"]
        store.start("ana", other)
        store.reset("ana", lid)
        c = ProgressStore(d, LS).cards("ana")
        assert c[lid]["status"] == "not_started" and c[lid]["completions"] == 0 and c[lid]["best_first_try"] == 0
        assert c[lid]["results"] == {} and c[other]["status"] == "in_progress"
        # A run saved before the four-part flow (no "flow" on the card) restarts at step 0.
        store.update("Ana", lid, LessonSession(LS[0], fake([]), step_idx=5))
        data = json.load(open(os.path.join(d, "ana.json"), encoding="utf-8"))
        data["lessons"][lid].pop("flow")
        json.dump(data, open(os.path.join(d, "ana.json"), "w", encoding="utf-8"))
        assert store.start("Ana", lid)["step"] == 0
asyncio.run(progress_cards()); print("progress OK")

# Each lesson can be finished by saying its expected phrases, even with a strict LLM.
async def all_lessons():
    for l in LS:
        s = LessonSession(l, fake([J("retry")] * 50))
        s.opening()
        while not s.done:
            await s.handle_student(s.step["expected_kn"][0], "kn")
        assert all(r == "first_try" for r in s.results.values()), (l["id"], s.results)
asyncio.run(all_lessons()); print("all lessons OK")


# ---------------------------------------------------------------- AuTut's four-part lesson flow
from tutor import end_choice

# Every lesson: concept intro -> guided repetition -> micro-roleplay -> mastery check -> wrap-up.
for l in LS:
    phases = [st["phase"] for st in l["steps"]]
    order = [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]
    assert order == ["concept", "repeat", "roleplay", "mastery", "close"], (l["id"], order)
    assert len({st["id"] for st in l["steps"]}) == len(phases), l["id"]
    for st in l["steps"]:
        if st["phase"] == "repeat":  # AuTut says exactly the phrase the student must repeat
            assert st["expected_kn"][0] in st["tutor_kn"], (l["id"], st["id"])
    assert 1 <= phases.count("mastery") <= 3, l["id"]
print("flow structure OK")

FL = LS[0]  # lesson 1 as a four-part flow
ids = [st["id"] for st in FL["steps"]]
checks = [st["id"] for st in FL["steps"] if st["phase"] == "mastery"]

async def four_part_flow():
    # Opening: concept intro names the key phrases; a question is answered, then "ready?" again.
    ans = json.dumps({"action": "answer", "say_kn": "ಹೌದು", "say_en": "Yes."})
    s = LessonSession(FL, fake([ans]))
    t = s.opening()
    assert t.phase == "concept" and "ನಮಸ್ಕಾರ" in t.kn and "Key phrases" in t.en, t.en
    t = await s.handle_student("Is this lesson about greetings?", "en")
    assert t.action == "answer" and s.step["id"] == "intro" and t.kn.endswith("ಸಿದ್ಧವೇ?"), t.kn
    # Any other reply means ready: on to guided repetition, announced.
    t = await s.handle_student("ok", "en")
    assert s.step["id"] == "rep_greet" and t.phase == "repeat" and "guided repetition" in t.en, t.en
    assert "ಮೊದಲು ಅಭ್ಯಾಸ ಮಾಡೋಣ." in t.kn and t.kn.endswith("ನನ್ನ ನಂತರ ಹೇಳಿ. ನಮಸ್ಕಾರ"), t.kn

    # Interrupt mid-repetition in English: answered, "back to the lesson", same step and attempts.
    s = LessonSession(FL, fake([J("retry"), ans]), step_idx=ids.index("rep_how_are_you"))
    await s.handle_student("ಏನೋ", "kn")
    assert s.attempts == 1
    t = await s.handle_student("Why do we repeat phrases?", "en")
    assert t.action == "answer" and s.step["id"] == "rep_how_are_you" and s.attempts == 1, (t.action, s.step["id"])
    assert "ನಾವು ನಿಲ್ಲಿಸಿದ ಕಡೆಯಿಂದ ಮುಂದುವರಿಸೋಣ." in t.kn and t.kn.endswith("ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ"), t.kn
    assert "where we left off" in t.en

    # Interrupt in Kannada ("I didn't understand, say it again"): hint slowly, same question,
    # no attempt used and no LLM call (the fake LLM has no replies).
    s = LessonSession(FL, fake([]), step_idx=ids.index("name"))
    t = await s.handle_student("ಅರ್ಥ ಆಗಲಿಲ್ಲ, ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ", "kn")
    assert t.action == "explain" and s.step["id"] == "name" and s.attempts == 0, (t.action, s.step["id"])
    assert t.kn.startswith("ಸರಿ, ನಿಧಾನವಾಗಿ ಹೇಳುತ್ತೇನೆ.") and t.kn.endswith(FL["steps"][ids.index("name")]["tutor_kn"]), t.kn

    # Entering the roleplay and the mastery check is announced.
    s = LessonSession(FL, fake([J("advance")] * 5), step_idx=ids.index([i for i in ids if i.startswith("rep_")][-1]))
    t = await s.handle_student(s.step["expected_kn"][0], "kn")
    assert s.step["id"] == "greet" and t.phase == "roleplay" and "micro-roleplay" in t.en, t.en
    s = LessonSession(FL, fake([J("advance")] * 5), step_idx=ids.index("where_from"))
    t = await s.handle_student("ನಾನು ಕೋಲ್ಕತ ಇಂದ ಬಂದಿದ್ದೇನೆ", "kn")
    assert s.step["id"] == "check_greet" and t.phase == "mastery" and "ಪರೀಕ್ಷೆ" in t.kn, t.kn

    # Mastery check gives fewer attempts: two misses, then AuTut models it and moves on.
    s = LessonSession(FL, fake([J("retry")] * 5), step_idx=ids.index("check_greet"))
    await s.handle_student("ಏನೋ", "kn")
    await s.handle_student("ಏನೋ", "kn")
    assert s.step["id"] == checks[1] and s.results["check_greet"] == "shown", s.step["id"]

    # End of lesson: mastery score, then "next lesson, repeat, or stop?".
    s = LessonSession(FL, fake([J("advance")] * 5), step_idx=ids.index("goodbye"),
                      results=dict(zip(checks, ["first_try", "passed", "first_try"])))
    t = await s.handle_student("ಧನ್ಯವಾದಗಳು", "kn")
    assert t.lesson_done and t.choices == ["next", "repeat", "stop"] and t.mastery == {"right": 2, "total": 3}
    assert t.kn.endswith("ಅಥವಾ ಈಗ ಸಾಕೇ?") and "Mastery check: 2 of 3" in t.en, (t.kn, t.en)
    j = t.to_json()
    assert j["choices"] == ["next", "repeat", "stop"] and j["phase"] == "end"
    # A question is still answered, then the choice is asked again.
    s.llm = fake([ans])
    t = await s.handle_student("When is the next class?", "en")
    assert t.action == "answer" and t.choices and s.choice is None and t.kn.endswith("ಸಾಕೇ?"), t.kn
    # Something unclear: asked again, nothing picked.
    t = await s.handle_student("ಹ್ಮ್", "kn")
    assert t.action == "ask_choice" and s.choice is None
    for said, lang, want in [("ಮುಂದಿನ ಪಾಠ", "kn", "next"), ("Let's do it again", "en", "repeat"),
                             ("ಈಗ ಸಾಕು", "kn", "stop"), ("Can we do the next lesson?", "en", "next"),
                             ("stop for now", "en", "stop"), ("ಮತ್ತೊಮ್ಮೆ ಮಾಡೋಣ", "kn", "repeat")]:
        s.choice = None
        t = await s.handle_student(said, lang)
        assert t.action == "choice" and t.choice == want and s.choice == want, (said, t.action, t.choice)
    # Last lesson: no "next" choice.
    s = LessonSession(FL, fake([J("advance")]), step_idx=ids.index("goodbye"), has_next=False)
    t = await s.handle_student("ಧನ್ಯವಾದಗಳು", "kn")
    assert t.choices == ["repeat", "stop"] and "last lesson" in t.en
    t = await s.handle_student("next", "en")
    assert t.action == "ask_choice" and s.choice is None
asyncio.run(four_part_flow()); print("four-part flow OK")

assert end_choice("When is the next class?", "en") is None
assert end_choice("ಮತ್ತೆ ಸಿಗೋಣ", "kn") == "stop"  # "see you again" is stop, not repeat
print("end choice OK")


# A question about a Kannada word mid-lesson: answer, pause, "now you say it", check, then
# "let's go back to where we left off", pause, and the step's question again.
from tutor import lesson_phrase_in, PAUSE
assert lesson_phrase_in("How do you pronounce Chenna Gidinai?", L) == "ಚೆನ್ನಾಗಿದ್ದೀನಿ" or \
       lesson_phrase_in("How do you pronounce Chenna Gidinai?", L).startswith("ಚೆನ್ನಾಗಿದ್ದ")
assert lesson_phrase_in("How do you say thank you in Kannada?", L) is None
assert lesson_phrase_in("When is the next class?", L) is None

async def side_practice():
    name = [st["id"] for st in L["steps"]].index("name")
    async def chat(msgs): return "Chennagiddini is pronounced as Chen-nah Gid-dee-nee."
    ans = json.dumps({"action": "answer", "say_kn": "", "say_en": ""})
    s = LessonSession(L, fake([ans]), chat=chat, step_idx=name)
    t = await s.handle_student("How do you pronounce Chenna Gidinai?", "en")
    phrase = s.practice["kn"]
    assert t.action == "practice" and s.step["id"] == "name", (t.action, t.kn)
    assert t.kn == f"{phrase}.{PAUSE}ಈಗ ನೀವು ಹೇಳಿ. {phrase}" and "pronounced" in t.en and "say it after me" in t.en, (t.kn, t.en)
    # A miss: one more try, still on the practice.
    t = await s.handle_student("ನಮಸ್ಕಾರ", "kn")
    assert t.action == "practice" and s.practice and t.kn.startswith("ಇನ್ನೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸಿ."), t.kn
    # Said right: praise, pause, "back to where we left off", pause, the step's question. No LLM call.
    t = await s.handle_student(phrase, "kn")
    q = L["steps"][name]["tutor_kn"]
    assert t.kn == f"ಚೆನ್ನಾಗಿದೆ!{PAUSE}ಸರಿ, ನಾವು ನಿಲ್ಲಿಸಿದ ಕಡೆಯಿಂದ ಮುಂದುವರಿಸೋಣ.{PAUSE}{q}", t.kn
    assert "where we left off" in t.en and s.practice is None and s.step["id"] == "name" and s.attempts == 0
    # Typed in English letters counts too; two misses end the practice kindly.
    s = LessonSession(L, fake([ans]), chat=chat, step_idx=name)
    await s.handle_student("How do you pronounce chennagiddini?", "en")
    t = await s.handle_student("chennagiddini", "en")
    assert t.kn.startswith("ಚೆನ್ನಾಗಿದೆ!"), t.kn
    s = LessonSession(L, fake([ans]), chat=chat, step_idx=name)
    await s.handle_student("How do you pronounce chennagiddini?", "en")
    await s.handle_student("ಏನೋ", "kn")
    t = await s.handle_student("ಏನೋ", "kn")
    assert t.kn.startswith("ಪರವಾಗಿಲ್ಲ") and t.kn.endswith(q) and s.practice is None, t.kn
    # "skip" goes straight back to the lesson.
    s = LessonSession(L, fake([ans]), chat=chat, step_idx=name)
    await s.handle_student("How do you pronounce chennagiddini?", "en")
    t = await s.handle_student("skip", "en")
    assert t.kn == f"ಸರಿ, ನಾವು ನಿಲ್ಲಿಸಿದ ಕಡೆಯಿಂದ ಮುಂದುವರಿಸೋಣ.{PAUSE}{q}", t.kn
    # A question that isn't about a Kannada word: answer, pause, back to where we left off, pause, question.
    async def chat2(msgs): return "There is no fixed timetable."
    async def tr(text): return "ಯಾವುದೇ ನಿಗದಿತ ವೇಳಾಪಟ್ಟಿ ಇಲ್ಲ."
    s = LessonSession(L, fake([ans]), chat=chat2, step_idx=name, translate=tr)
    t = await s.handle_student("When is the next class?", "en")
    assert t.kn.count(PAUSE) == 2 and t.kn.endswith(f"ಮುಂದುವರಿಸೋಣ.{PAUSE}{q}") and s.practice is None, t.kn
asyncio.run(side_practice()); print("side practice OK")

# "Okay" is not an answer, the goodbye step says what to do after a question, practice also works
# after the lesson, and the end-of-lesson reply points to the buttons.
async def acks_and_end():
    q_name = FL["steps"][ids.index("name")]["tutor_kn"]
    s = LessonSession(FL, fake([]), step_idx=ids.index("name"))
    t = await s.handle_student("Okay", "en")      # no LLM call, no attempt
    assert t.action == "ack" and s.attempts == 0 and s.step["id"] == "name" and t.kn.endswith(q_name), t.kn
    async def tr(text): return {"see you next time": "ಮತ್ತೆ ಸಿಗೋಣ"}.get(text.strip("'\" ?"), "ಅನುವಾದ")
    async def chat(msgs): return "You say matte sigona."
    ans = json.dumps({"action": "answer", "say_kn": "", "say_en": ""})
    # At the goodbye step: practise, "Okay" doesn't count as a try, then the goodbye reprompt (not silence).
    s = LessonSession(FL, fake([ans]), step_idx=ids.index("goodbye"), translate=tr, chat=chat)
    t = await s.handle_student("How to say 'see you next time'", "en")
    assert t.action == "practice" and s.practice["kn"] == "ಮತ್ತೆ ಸಿಗೋಣ", t.kn
    t = await s.handle_student("Okay", "en")
    assert t.action == "practice" and s.practice["tries"] == 0, t.kn
    t = await s.handle_student("ಮತ್ತೆ ಸಿಗೋಣ", "kn")
    assert t.kn.endswith("ಪಾಠ ಮುಗಿಸಲು ಧನ್ಯವಾದಗಳು ಎಂದು ಹೇಳಿ.") and not s.done, t.kn
    # After the lesson: practise, then "next lesson, repeat, or stop?" again.
    s = LessonSession(FL, fake([ans]), step_idx=ids.index("goodbye"), translate=tr, chat=chat); s.done = True
    t = await s.handle_student("How to say 'see you next time'", "en")
    assert t.action == "practice" and t.lesson_done
    t = await s.handle_student("ಮತ್ತೆ ಸಿಗೋಣ", "kn")
    assert t.kn.endswith("ಅಥವಾ ಈಗ ಸಾಕೇ?") and t.choices and s.choice is None, t.kn
    # "Let's go to the next lesson": AuTut points to the button.
    t = await s.handle_student("lets go to the next lesson", "en")
    assert t.choice == "next" and "Next lesson button" in t.en and "ಬಟನ್" in t.kn, t.en
asyncio.run(acks_and_end()); print("acks and end OK")


# The new screen: each turn says what the phrase card shows, where the learner is, and the score.
async def screen_fields():
    from progress import load_lessons
    L1 = load_lessons("lessons")[0]
    s = LessonSession(L1, fake([J("advance")] * 20))
    j = s.opening().to_json()
    assert j["focus"]["kind"] == "phrases" and j["focus"]["items"][0]["roman"], j["focus"]
    assert j["position"] == {"phase": "concept", "i": 1, "n": 1}
    j = (await s.handle_student("ಸರಿ", "kn")).to_json()
    first = j["focus"]
    assert first["kind"] == "repeat" and first["kn"] and first["roman"], first
    assert j["position"]["phase"] == "repeat" and j["position"]["i"] == 1
    j = (await s.handle_student(first["kn"], "kn")).to_json()
    assert j["score"] == 1.0 and j["passed"] and j["position"]["i"] == 2, j
    while s.phase != "roleplay":
        await s.handle_student(s.step["expected_kn"][0], "kn")
    f = s.focus()
    assert f["kind"] == "question" and "answer" in f, f
    while s.phase != "mastery":
        await s.handle_student(s.step["expected_kn"][0], "kn")
    assert s.focus()["kind"] == "question" and "answer" not in s.focus()  # no model answer in the check
asyncio.run(screen_fields()); print("screen fields OK")


# Lesson start: AuTut's opening is also spoken in English (english_tts.speakable cleans it up).
def spoken_english():
    from english_tts import speakable
    from progress import load_lessons
    L1 = load_lessons("lessons")[0]
    t = LessonSession(L1, fake([])).opening()
    assert t.speak_en == t.en and t.to_json()["focus"]
    line = speakable(t.speak_en)
    assert line and not any("ಀ" <= c <= "೿" for c in line) and "Namaskara, meaning hello" in line, line
    assert "Say sari when you are ready" in line and "\n" not in line, line
    back = LessonSession(L1, fake([]), step_idx=4).opening()
    assert back.speak_en.startswith("Welcome back!")
    s = LessonSession(L1, fake([J("advance")]))
    s.opening()
    assert asyncio.run(s.handle_student("ಸರಿ", "kn")).speak_en is None  # only the opening
spoken_english(); print("spoken english OK")
