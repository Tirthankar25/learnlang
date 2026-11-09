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
    assert t.kn.startswith("ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ.") and t.kn.endswith("ನೀವು ಹೇಗಿದ್ದೀರಿ?") and s.attempts == 0, t.kn
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
import tempfile
from progress import ProgressStore, load_lessons, user_key
from translit import has_kannada
LS = load_lessons("lessons")
assert [l["num"] for l in LS][:3] == ["1", "1a", "2"] and LS[0]["id"] == L["id"], [l["num"] for l in LS]
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
        s = LessonSession(LS[0], fake([J("advance"), J("retry"), J("retry")]), step_idx=card["step"])
        await s.handle_student("ನಮಸ್ಕಾರ", "kn")                      # first try
        await s.handle_student("ಏನೋ", "kn")                           # retry ...
        await s.handle_student("ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ", "kn")           # ... then passed
        card = store.update("Ana", lid, s)
        assert card["status"] == "in_progress" and card["step"] == 2 and card["first_try"] == 1, card
        assert card["results"] == {"greet": "first_try", "how_are_you": "passed"}, card["results"]
        # Another learner is untouched.
        assert store.cards("Bo")[lid]["status"] == "not_started"
        # Reconnect: resumes at the saved step with a welcome back.
        card = store.start("Ana", lid)
        s = LessonSession(LS[0], fake([J("advance")] * 20), step_idx=card["step"], results=dict(card["results"]))
        t = s.opening()
        assert card["step"] == 2 and t.kn.startswith("ಮತ್ತೆ ಸ್ವಾಗತ!") and t.kn.endswith(LS[0]["steps"][2]["tutor_kn"]), t.kn
        while not s.done:
            await s.handle_student(s.step["expected_kn"][0], "kn")
            card = store.update("Ana", lid, s)
        card = store.update("Ana", lid, s)  # extra turns after the end don't count twice
        assert card["status"] == "completed" and card["completions"] == 1 and len(card["results"]) == card["total"], card
        assert store.next_lesson("Ana") == LS[1]["id"]
        # Replaying a finished lesson starts at step 0 and keeps the completion count.
        card = store.start("Ana", lid)
        assert card["step"] == 0 and card["results"] == {} and card["completions"] == 1
        # Survives a restart of the server (new store, same folder).
        assert ProgressStore(d, LS).cards("ana")[lid]["completions"] == 1
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
