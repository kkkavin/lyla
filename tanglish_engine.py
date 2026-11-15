"""
tanglish_engine.py - Advanced Tanglish & Regional Tamil Nadu Dialect Engine
Engineered for Lyla: Room-scale full-duplex conversational voice assistant.

Features:
1. Four Regional Dialect Profiles:
   - Chennai Gethu / Madras Baashai
   - Coimbatore / Kongu Tamil
   - Madurai / Theni Dialect
   - Modern Urban Tanglish (Default)
   - Global English
2. Tanglish-to-IPA Phonetic Bridge for Kokoro Neural TTS (authentic South Indian pronunciation).
3. Conversational Prosody & Banter Formatter (natural spoken flow, fillers, micro-pauses).
4. Far-field STT Lexical Normalizer & Phonetic Repair Dictionary.
5. In-conversation Dialect Switching Intent Detector.
"""

import re
from typing import Dict, Optional, Tuple, List

# ================= 1. REGIONAL DIALECT CONFIGURATIONS =================

DIALECTS: Dict[str, dict] = {
    "tanglish": {
        "id": "tanglish",
        "name": "Modern Tanglish",
        "tag": "Tanglish",
        "speed": 1.02,
        "pitch_shift": 1.0,
        "greeting": "Hey macha! Naan semmaya iruken. Innaiku enna plan?",
        "switch_confirm": "Done deal macha! Modern Tanglish vibe-la pesuren.",
        "prompt": (
            "You are Lyla, a lively, clever AI friend speaking natural Modern Tanglish (Tamil + English code-mixed). "
            "STYLE RULES:\n"
            "- Speak English blended effortlessly with popular colloquial Tamil words written in English alphabet.\n"
            "- Use words like: macha, super bro, chill pannu, actually paatha, no scene, semma vibe, kandippa, aama, seri, pakka.\n"
            "- Tone: warm, witty, energetic, like a great friend chatting in a studio room.\n"
            "- Length: 1 to 2 short spoken sentences. Never bullet points or formal essays.\n"
            "- Never write in Tamil script, only Latin English alphabet."
        )
    },
    "chennai": {
        "id": "chennai",
        "name": "Chennai Gethu",
        "tag": "Chennai Slang",
        "speed": 1.06,
        "pitch_shift": 1.02,
        "greeting": "Hey machi! Vada thala, naan getha iruken da. Enna matter-u?",
        "switch_confirm": "Vera level machi! Ipo paaru, namma Chennai Madras Baashai-la therikka vidrom!",
        "prompt": (
            "You are Lyla, an ultra-sharp, humorous AI friend speaking authentic Chennai / Madras Baashai.\n"
            "STYLE RULES:\n"
            "- Use iconic Chennai slang: machi, mapla, enna thala, semma, galaata, kalaikadha, scene podadha, "
            "gethu, besh besh, billa madhiri, thakkali, vitta paaru, therikka vidrom, vaapla.\n"
            "- Tone: fast-paced, teasing, confident, full of street-smart humor.\n"
            "- Length: 1 to 2 punchy spoken lines. Speak like a close college buddy from North/Central Chennai.\n"
            "- Never write in Tamil script, only Latin English alphabet."
        )
    },
    "kongu": {
        "id": "kongu",
        "name": "Kongu / Kovai",
        "tag": "Kongu Tamil",
        "speed": 0.98,
        "pitch_shift": 0.98,
        "greeting": "Vanakkamunga! Naan nalla irukene, neenga aptheengala kannu?",
        "switch_confirm": "Sari paapomunga! Coimbatore Kongu Tamil-la paasama pesuren, kettu paarunga.",
        "prompt": (
            "You are Lyla, speaking warm, respectful, affectionate Coimbatore / Kongu Tamil.\n"
            "STYLE RULES:\n"
            "- Always use trademark Kongu '-unga' honorifics: ennaunga, sollunga, aptheengala, poiteengala, paathukonga.\n"
            "- Use gentle words: kannu, romba porumai, gavanama irunga, nalla irukeengala, neenga sonna correct-unga, sari paapom.\n"
            "- Tone: deeply polite, affectionate, warm hospitality, soothing and caring.\n"
            "- Length: 1 to 2 short spoken lines.\n"
            "- Never write in Tamil script, only Latin English alphabet."
        )
    },
    "madurai": {
        "id": "madurai",
        "name": "Madurai Veeram",
        "tag": "Madurai Style",
        "speed": 1.04,
        "pitch_shift": 1.01,
        "greeting": "Yov kettukko! Namma irukara appo enna bayam? Enna matter-nu sollu pa!",
        "switch_confirm": "Aaha! Madurai stylenu sonna odane gethu koodiruchu pa. Direct-ah therika viduvom!",
        "prompt": (
            "You are Lyla, speaking bold, energetic, direct Madurai / Theni dialect.\n"
            "STYLE RULES:\n"
            "- Use authentic Madurai phrases: yov, enna pa, kettukko, vanthutenla, vetti pechu vendaam, anney, "
            "semma gethu, paathukalam, nermaiya pesu, sollitten la.\n"
            "- Tone: bold, loud, fearless, intensely loyal, brotherly/sisterly, straight to the point.\n"
            "- Length: 1 to 2 punchy spoken lines.\n"
            "- Never write in Tamil script, only Latin English alphabet."
        )
    },
    "english": {
        "id": "english",
        "name": "Global English",
        "tag": "English",
        "speed": 1.0,
        "pitch_shift": 1.0,
        "greeting": "Hey Kavin! I'm doing great and ready to chat. What's on your mind?",
        "switch_confirm": "Switched to English! Ready when you are.",
        "prompt": (
            "You are Lyla, a bright, natural, conversational AI companion.\n"
            "STYLE RULES:\n"
            "- Speak clean, natural, human-like spoken English with conversational flow.\n"
            "- Use contractions, natural pauses, and lively tone.\n"
            "- Length: 1 to 2 concise spoken sentences."
        )
    }
}

# ================= 2. IN-CONVERSATION DIALECT SWITCH DETECTOR =================

DIALECT_TRIGGERS: List[Tuple[str, re.Pattern]] = [
    ("chennai", re.compile(r"\b(chennai|madras|vada\s?chennai|gethu)\s*(slang|style|baashai|la|mode|dialect)?\b", re.I)),
    ("kongu", re.compile(r"\b(kongu|coimbatore|kovai|unga\s?style)\s*(slang|style|la|mode|dialect)?\b", re.I)),
    ("madurai", re.compile(r"\b(madurai|theni|yov\s?style)\s*(slang|style|la|mode|dialect)?\b", re.I)),
    ("tanglish", re.compile(r"\b(tanglish|tamil\s?english|modern\s?tanglish|macha\s?style)\s*(slang|style|la|mode|dialect)?\b", re.I)),
    ("english", re.compile(r"\b(english|only\s?english|angilam)\s*(slang|style|la|mode|dialect)?\b", re.I))
]

def detect_dialect_switch_intent(text: str) -> Optional[str]:
    """Detects if the user is asking Lyla to switch her spoken slang/dialect."""
    low = text.lower()
    # Check for direct change/switch intent or direct dialect mention with 'slang' / 'style'
    has_switch_verb = any(v in low for v in [
        "switch", "change", "speak in", "talk in", "pesu", "maathu", "maathikiriya", 
        "slang-la", "style-la", "baashai-la", "mode"
    ])
    for dialect_key, pattern in DIALECT_TRIGGERS:
        if pattern.search(low):
            if has_switch_verb or "slang" in low or "style" in low or "baashai" in low:
                return dialect_key
    return None

# ================= 3. STT LEXICON BIAS & PHONETIC NORMALIZER =================

# Initial prompt injected into Faster-Whisper to strongly bias the acoustic beam toward Tanglish
STT_INITIAL_PROMPT = (
    "Vocabulary: Lyla, Kavin, Machi, Mapla, Semma, Epdi, Iruka, Iruken, Pannu, Panra, Aama, "
    "Seri, Romba, Kandippa, Sollu, Thala, Galaata, Kalaikadha, Ennaunga, Aptheengala, "
    "Poiteengala, Kannu, Porumai, Yov, Kettukko, Vanthutenla, Vada, Poda, Illai, Nalla, "
    "Paatha, Vibe, Gethu, Ollama, Whisper, Kokoro, Python."
)

PHONETIC_STT_FIXES: List[Tuple[re.Pattern, str]] = [
    # Wake word & system names
    (re.compile(r"\b(lylah|layla|laylah|lila|lilah|leila|leilah|laila|leyla|lyler|liler|lie\s?la|la\s?la|lala|isla)\b", re.I), "Lyla"),
    (re.compile(r"\bo\s?llama\b", re.I), "Ollama"),

    # Common far-field misrecognitions for Tanglish words
    (re.compile(r"\b(solo\s?match|solo\s?machi|solo\s?matchy)\b", re.I), "sollu machi"),
    (re.compile(r"\b(solo|sollow)\b", re.I), "sollu"),
    (re.compile(r"\b(in\s?a\s?panda|in\s?a\s?panra|in\s?a\s?pan\s?raw)\b", re.I), "enna panra"),
    (re.compile(r"\b(in\s?a\s?matter|in\s?matter)\b", re.I), "enna matter"),
    (re.compile(r"\b(every\s?rookie|every\s?rooka|epdy\s?ruka|apdi\s?iruka|abdy\s?iruka)\b", re.I), "epdi iruka"),
    (re.compile(r"\b(van\s?a\s?come|vanna\s?come|vanak\s?kam|vanakam)\b", re.I), "vanakkam"),
    (re.compile(r"\b(summer\s?hot|summer\s?vibe|summer-ah)\b", re.I), "semma hot"),
    (re.compile(r"\b(armour|ah\s?mah|a\s?mah)\b", re.I), "aama"),
    (re.compile(r"\b(say\s?ree|cherry\s?cherry)\b", re.I), "seri"),
    (re.compile(r"\b(can\s?deep\s?ah|candipa|kandeepa)\b", re.I), "kandippa"),
    (re.compile(r"\b(app\s?thing\s?ala|opting\s?ala|abtheengala)\b", re.I), "aptheengala"),
    (re.compile(r"\b(in\s?a\s?unga|en\s?a\s?unga)\b", re.I), "ennaunga"),
    (re.compile(r"\b(yo\s?v|y\s?of|yow)\b", re.I), "yov"),
    (re.compile(r"\b(one\s?ten\s?la|vandhutenla)\b", re.I), "vanthutenla"),
    (re.compile(r"\b(get\s?to|get\s?through)\b", re.I), "gethu"),
    (re.compile(r"\b(collect\s?other|kalaikadha)\b", re.I), "kalaikadha"),
    (re.compile(r"\b(chill\s?panel|chill\s?pan)\b", re.I), "chill pannu")
]

def enhance_tanglish_stt(transcript: str) -> str:
    """Cleans up raw ASR output using domain-specific phonetic corrections."""
    text = transcript.strip()
    for pattern, replacement in PHONETIC_STT_FIXES:
        text = pattern.sub(replacement, text)
    return text

# ================= 4. TANGLISH-TO-IPA PHONETIC BRIDGE =================
# Maps colloquial Tamil & Tanglish words to exact South Indian IPA phonemes.
# Kokoro's neural model uses this directly to speak with natural Tamil phonetics
# instead of mispronouncing them with an American English accent.

TANGLISH_IPA: Dict[str, str] = {
    # Greetings & Common Address
    "vanakkam": "vʌnɐkːʌm",
    "vanakkamunga": "vʌnɐkːʌmʊŋɡə",
    "machi": "mɑːtʃi",
    "macha": "mɑːtʃɑː",
    "mapla": "mɑːplə",
    "mappillai": "mɑːppɪɭɭaɪ",
    "thala": "t̪ʌlɑː",
    "anney": "ʌnneːj",
    "thambi": "t̪ʌmbi",
    "kannu": "kʌnnʊ",
    "pa": "pɑː",
    "da": "d̪ɑː",
    "di": "d̪i",

    # Pronouns & Questions
    "naan": "nɑːn",
    "neenga": "niːŋɡə",
    "ne": "niː",
    "namma": "nʌmmɑː",
    "enna": "ɛnnɑː",
    "ennaunga": "ɛnnɑːʊŋɡə",
    "epdi": "ɛpdi",
    "enga": "ɛŋɡə",
    "yaaru": "jɑːrʊ",
    "edhuku": "eːd̪ʊkːʊ",

    # Status & Emotions
    "semma": "sɛmmɑː",
    "super": "sˈupəɹ",
    "gethu": "ɡeːt̪ʊ",
    "nalla": "nʌllɑː",
    "nallaa": "nʌllɑː",
    "galaata": "ɡʌlɑːʈʈɑː",
    "kalaikadha": "kʌlaɪkːɑːd̪ə",
    "scene": "sˈin",
    "bayam": "bʌjʌm",
    "porumai": "pɔrʊmaɪ",
    "porumaiya": "pɔrʊmaɪjɑː",
    "gavanama": "ɡʌvʌnʌmɑː",

    # Verbs & Conjugations
    "iruka": "ɪrʊkːə",
    "iruken": "ɪrʊkːeːn",
    "irukene": "ɪrʊkːeːneː",
    "iruku": "ɪrʊkːʊ",
    "irukeengala": "ɪrʊkːiːŋɡəɭɑː",
    "irukara": "ɪrʊkːərə",
    "sollu": "sɔllʊ",
    "sollunga": "sɔllʊŋɡə",
    "sollitten": "sɔllɪt̪ːeːn",
    "solliteanla": "sɔllɪt̪ːeːnlɑː",
    "panra": "pʌnrɑː",
    "pannu": "pʌnnʊ",
    "paapom": "pɑːppoːm",
    "paapomunga": "pɑːppoːmʊŋɡə",
    "paatha": "pɑːt̪ːɑː",
    "paathukalam": "pɑːt̪ːʊkːʌlɑːm",
    "paathukonga": "pɑːt̪ːʊkːɔŋɡə",
    "aptheengala": "ɑːpt̪iːŋɡəɭɑː",
    "poiteengala": "poːjtiːŋɡəɭɑː",
    "vanthutenla": "vʌnd̪ʊteːnlɑː",
    "therikka": "t̪eːrɪkːə",
    "vidrom": "vɪɖroːm",
    "viduvom": "vɪɖʊvoːm",
    "kettukko": "keːʈʈʊkːoː",
    "pesu": "peːsʊ",
    "pesuren": "peːsʊreːn",
    "kudunga": "kʊɖʊŋɡə",
    "vada": "vɑːd̪ɑː",
    "poda": "poːd̪ɑː",

    # Markers & Affirmatives
    "aama": "ɑːmɑː",
    "seri": "sɛri",
    "sari": "sʌri",
    "kandippa": "kʌndɪppɑː",
    "illai": "ɪllaɪ",
    "illa": "ɪllɑː",
    "romba": "rɔmbə",
    "ennaikku": "ɪnnaɪkːʊ",
    "innaiku": "ɪnnaɪkːʊ",
    "matter": "mˈætəɹ",
    "vibe": "vˈaɪb",
    "chill": "ʧˈɪl",
    "pakka": "pʌkːɑː",
    "yov": "joːv",
    "vaapla": "vɑːplə",
    "besh": "beːʃ",
    "thakkali": "t̪ʌkːɑːɭi",
    "vetti": "veːʈʈi",
    "pechu": "peːʧʊ",
    "vendaam": "veːndɑːm",
    "nermai": "neːrmaɪ",
    "nermaiya": "neːrmaɪjɑː",
    "koodiruchu": "kuːɖɪrʊtʃʊ"
}

def format_conversational_prosody(text: str) -> str:
    """
    Formats the text with human-like pauses and expressive punctuation
    to produce rich, natural spoken cadence in Kokoro.
    """
    # Replace long pauses / ellipses with an em-dash which Kokoro renders with a natural 120ms breath pause
    t = re.sub(r"\.{2,}", " — ", text)
    # Ensure natural comma spacing
    t = re.sub(r"\s*,\s*", ", ", t)
    # Ensure exclamation spacing
    t = re.sub(r"!+", "! ", t)
    # Clean up double dashes or spaces
    t = re.sub(r"\s+—\s+", " — ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

def text_to_tanglish_phonemes(text: str, g2p_engine) -> str:
    """
    Hybrid G2P Transcriber:
    - Splits text into tokens.
    - If a token is in TANGLISH_IPA: injects South Indian IPA phonemes directly.
    - If a token has Tamil suffix markers (-unga, -la, -da, -pa, -ah): converts stem and joins.
    - Otherwise: passes through standard English G2P so English words maintain clear fluent pronunciation.
    """
    # Split words while preserving punctuation
    tokens = re.findall(r"[\w'-]+|[.,!?;:—]", text)
    result_ps = []

    for tok in tokens:
        if tok in ".,!?;:—":
            result_ps.append(tok)
            continue

        low = tok.lower().strip("'-")
        # Direct match in Tanglish IPA dictionary
        if low in TANGLISH_IPA:
            result_ps.append(TANGLISH_IPA[low])
            continue

        # Check for attached suffix markers: e.g. "hot-ah", "matter-u", "slang-la"
        suffix_match = re.match(r"^([a-z]+)[-_]?(ah|la|da|pa|unga|nga|u)$", low)
        if suffix_match:
            stem, suffix = suffix_match.groups()
            suffix_ipa_map = {
                "ah": "ɑː", "la": "lɑː", "da": "d̪ɑː", "pa": "pɑː",
                "unga": "ʊŋɡə", "nga": "ŋɡə", "u": "ʊ"
            }
            stem_ipa = TANGLISH_IPA.get(stem)
            if stem_ipa:
                result_ps.append(stem_ipa + suffix_ipa_map.get(suffix, ""))
                continue

        # Fallback to English G2P engine for English words
        try:
            _, en_tokens = g2p_engine(tok)
            word_ps = "".join(t.phonemes for t in en_tokens if hasattr(t, "phonemes") and t.phonemes)
            if word_ps:
                result_ps.append(word_ps)
            else:
                result_ps.append(tok)
        except Exception:
            result_ps.append(tok)

    # Join phonemes with natural spacing
    out = " ".join(result_ps)
    # Clean up punctuation spacing
    out = re.sub(r"\s+([.,!?;:—])", r"\1", out)
    return out
