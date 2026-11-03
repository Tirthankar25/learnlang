"""Kannada script -> English-letter (Latin) transliteration for learners.

Self-contained (no dependencies) so every romanized line in the app comes from one
place and is spelled the same way: what LeLa says, the hints, and what the student said.

Style: long vowels get a macron (ā ī ū ē ō) because length changes meaning in Kannada;
retroflex consonants keep a dot (ṭ ḍ ṇ ḷ); ಚ/ಶ/ಷ are written ch/sh/sh so they read
naturally in English. transliterate(text, ascii_only=True) drops the marks entirely.
"""

import re
import unicodedata

VOWELS = {
    "ಅ": "a", "ಆ": "ā", "ಇ": "i", "ಈ": "ī", "ಉ": "u", "ಊ": "ū", "ಋ": "ru", "ೠ": "rū",
    "ಌ": "lu", "ಎ": "e", "ಏ": "ē", "ಐ": "ai", "ಒ": "o", "ಓ": "ō", "ಔ": "au",
}
SIGNS = {
    "ಾ": "ā", "ಿ": "i", "ೀ": "ī", "ು": "u", "ೂ": "ū", "ೃ": "ru", "ೄ": "rū",
    "ೆ": "e", "ೇ": "ē", "ೈ": "ai", "ೊ": "o", "ೋ": "ō", "ೌ": "au",
}
CONSONANTS = {
    "ಕ": "k", "ಖ": "kh", "ಗ": "g", "ಘ": "gh", "ಙ": "ng",
    "ಚ": "ch", "ಛ": "chh", "ಜ": "j", "ಝ": "jh", "ಞ": "ny",
    "ಟ": "ṭ", "ಠ": "ṭh", "ಡ": "ḍ", "ಢ": "ḍh", "ಣ": "ṇ",
    "ತ": "t", "ಥ": "th", "ದ": "d", "ಧ": "dh", "ನ": "n",
    "ಪ": "p", "ಫ": "ph", "ಬ": "b", "ಭ": "bh", "ಮ": "m",
    "ಯ": "y", "ರ": "r", "ಱ": "r", "ಲ": "l", "ವ": "v",
    "ಶ": "sh", "ಷ": "sh", "ಸ": "s", "ಹ": "h", "ಳ": "ḷ", "ೞ": "ḻ",
}
NUKTA_CONSONANTS = {"ಫ": "f", "ಜ": "z"}  # ಫ಼ ಜ಼ in loanwords
VIRAMA, ANUSVARA, VISARGA, NUKTA, AVAGRAHA = "್", "ಂ", "ಃ", "಼", "ಽ"
DIGITS = {chr(0x0CE6 + i): str(i) for i in range(10)}

# Anusvara (ಂ) is pronounced as the nasal of the following consonant's class.
_NASAL_BEFORE = {}
for group, nasal in (("ಕಖಗಘ", "n"), ("ಚಛಜಝ", "n"), ("ಟಠಡಢ", "ṇ"), ("ತಥದಧ", "n"), ("ಪಫಬಭ", "m")):
    for ch in group:
        _NASAL_BEFORE[ch] = nasal


def has_kannada(text: str) -> bool:
    return any("ಀ" <= ch <= "೿" for ch in text or "")


def transliterate(text: str, ascii_only: bool = False) -> str:
    text = unicodedata.normalize("NFC", text or "")
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch in CONSONANTS:
            base = CONSONANTS[ch]
            if nxt == NUKTA:
                base = NUKTA_CONSONANTS.get(ch, base)
                i += 1
                nxt = text[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                out.append(base)
                i += 2
                continue
            if nxt in SIGNS:
                out.append(base + SIGNS[nxt])
                i += 2
                continue
            out.append(base + "a")  # inherent vowel; Kannada keeps it at word end too
        elif ch in VOWELS:
            out.append(VOWELS[ch])
        elif ch == ANUSVARA:
            out.append(_NASAL_BEFORE.get(nxt, "m"))
        elif ch == VISARGA:
            out.append("h")
        elif ch in DIGITS:
            out.append(DIGITS[ch])
        elif ch in (NUKTA, VIRAMA, AVAGRAHA, "‌", "‍"):
            pass  # stray marks / zero-width joiners
        else:
            out.append(ch)  # spaces, punctuation, Latin text pass through
        i += 1

    result = "".join(out)
    # Capitalize the start of each sentence so lines read like English.
    result = re.sub(r"(^|[.!?]\s+)([a-zāīūēōṭḍṇḷ])", lambda m: m.group(1) + m.group(2).upper(), result)
    if ascii_only:
        result = "".join(c for c in unicodedata.normalize("NFD", result) if not unicodedata.combining(c))
    return result
