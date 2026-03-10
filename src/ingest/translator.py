from deep_translator import GoogleTranslator


def is_greek(text):
    """Detect if text is primarily Greek."""
    greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03ff' or '\u1f00' <= c <= '\u1fff')
    return greek_chars > len(text) * 0.3


def translate_to_english(text):
    """Translate Greek text to English using Google Translate."""
    try:
        # Google Translate has a 5000 char limit per request
        if len(text) <= 4500:
            return GoogleTranslator(source='el', target='en').translate(text)
        
        # Split long text into chunks and translate each
        parts = []
        words = text.split()
        current = ""
        for word in words:
            if len(current) + len(word) + 1 < 4500:
                current += " " + word
            else:
                parts.append(GoogleTranslator(source='el', target='en').translate(current.strip()))
                current = word
        if current:
            parts.append(GoogleTranslator(source='el', target='en').translate(current.strip()))
        
        return " ".join(parts)

    except Exception as e:
        print(f"  Translation error: {e}")
        return None