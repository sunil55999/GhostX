import re
from typing import List
from flashtext import KeywordProcessor

# Invisible characters to be removed
INVISIBLE_CHARS = [
    '\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '\u202a', '\u202b',
    '\u202c', '\u202d', '\u202e', '\u2060', '\u2061', '\u2062', '\u2063',
    '\u2064', '\ufeff'
]
INVISIBLE_CHARS_PATTERN = re.compile(f"[{''.join(INVISIBLE_CHARS)}]")

# Symbol-based trap patterns
SYMBOL_TRAP_PATTERNS = re.compile(r"""
    ^[\s/\\*\-–—•·.]{1,3}$          # Lines with only 1-3 symbols like /, *, -
    | ^[0-9#*]+\.\s*$              # Numbered list items like "1.", "2."
    | ^[\U0001F600-\U0001F64F\s]+$  # Lines with only emojis and whitespace
""", re.VERBOSE | re.MULTILINE)

# Homoglyph detection (simple version)
HOMOGLYPHS = {
    'a': 'а', 'e': 'е', 'o': 'о', 'c': 'с', 'p': 'р', 'x': 'х', 'y': 'у',
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М', 'H': 'Н', 'O': 'О',
    'P': 'Р', 'C': 'С', 'T': 'Т', 'X': 'Х'
}
HOMOGLYPH_PATTERN = re.compile(f"[{''.join(HOMOGLYPHS.values())}]")

def normalize_text(text: str) -> str:
    """
    Removes invisible characters and normalizes whitespace.
    """
    if not text:
        return ""
    text = INVISIBLE_CHARS_PATTERN.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _build_keyword_processor(phrases: List[str]) -> KeywordProcessor:
    processor = KeywordProcessor(case_sensitive=False)
    for phrase in phrases:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase:
            processor.add_keyword(normalized_phrase)
    return processor

def is_trap_candidate(text: str, trap_phrases: List[str], honeypot_phrases: List[str]) -> tuple[bool, str]:
    """
    Comprehensive trap detection function.
    Returns (is_trap, reason).
    """
    if not text or not text.strip():
        return True, "Message is empty or contains only whitespace"

    normalized_text = normalize_text(text)
    if not normalized_text:
        return True, "Message is empty after normalization (invisible chars)"

    # 1. Check for very short messages
    if len(normalized_text) < 5:
        return True, f"Message is too short ({len(normalized_text)} chars)"

    # 2. Check for symbol-based traps
    if SYMBOL_TRAP_PATTERNS.search(text):
        return True, "Message matches symbol trap patterns (e.g., '1.', '*', '/')"

    # 3. Check for homoglyphs
    if HOMOGLYPH_PATTERN.search(text):
        return True, "Message contains homoglyphs (visually similar characters)"

    # 4. Check for honeypot phrases (silent skip)
    if honeypot_phrases:
        honeypot_processor = _build_keyword_processor(honeypot_phrases)
        found = honeypot_processor.extract_keywords(normalized_text.lower())
        if found:
            return True, "Honeypot phrase detected (silent skip)"

    # 5. Check for explicit trap phrases (block and notify)
    if trap_phrases:
        trap_processor = _build_keyword_processor(trap_phrases)
        found = trap_processor.extract_keywords(normalized_text.lower())
        if found:
            return True, f"Explicit trap phrase detected: '{found[0]}'"

    return False, ""