import re
from collections import Counter
from itertools import groupby

def is_gibberish(text):
    """Check if text appears to be gibberish"""
    if not text or not isinstance(text, str):
        return True
    
    words = re.findall(r"\w+", text.lower())
    if len(words) == 0:
        return True
    
    # Check for excessive repetition
    word_counts = Counter(words)
    if word_counts:
        most_common, count = word_counts.most_common(1)[0]
        if count > 12 or (count / len(words)) > 0.4:
            return True
    
    # Check minimum word count
    if len(words) < 2:
        return True
    
    return False

def build_description_from_codes(codes):
    """Build description from Gardiner codes"""
    from config import Config
    config = Config()
    
    labels = [config.CODE_TO_LABEL.get(code, code) for code in codes]
    compressed = []
    
    for key, group in groupby(labels):
        count = len(list(group))
        name = "unknown" if (key == "?" or key is None) else key
        compressed.append(f"{name} (x{count})" if count > 1 else name)
    
    return ", ".join(compressed)

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_words(text, min_length=2):
    """Extract words from text with minimum length"""
    if not text:
        return []
    
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    return [word for word in words if len(word) >= min_length]

def calculate_text_stats(text):
    """Calculate basic text statistics"""
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "unique_chars": 0,
            "avg_word_length": 0
        }
    
    words = extract_words(text)
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "unique_chars": len(set(text)),
        "avg_word_length": sum(len(word) for word in words) / max(1, len(words))
    }
