import os
from pathlib import Path
import torch

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent

    TESSERACT_EXE = os.getenv("TESSERACT_EXE", "tesseract")
    TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")

    REFERENCES_PATH = BASE_DIR / "references.json"
    ANCIENT_GREEK_TESSDATA = BASE_DIR / "tessdata" / "ancient-greek"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
    # Model Settings
    HF_TRANSLATOR_MODEL = "AnushS/Hieroglyph-Translator-Using-Gardiner-Codes"
    CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    DEVICE = 0 if torch.cuda.is_available() else -1
    
    # Groq Settings
    GROQ_MODEL = "openai/gpt-oss-120b"
    GROQ_TEMPERATURE = 1.0
    GROQ_STORY_MAX_TOKENS = 1024
    GROQ_CONTEXT_MAX_TOKENS = 2048
    
    # File Upload Settings
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Gardiner Code Mapping
    GARDINER_MAP = {
        "man_seated": "A1", "woman_seated": "B1", "god_figure": "C1",
        "eye": "D4", "hippopotamus": "E25", "leg": "F28", "owl": "G17",
        "feather": "H2", "lizard": "I1", "fish": "K1", "insect": "L1",
        "reed": "M17", "sun": "N5", "crown": "S39", "bow": "T14",
        "hoe": "U25", "rope": "V1", "jar": "W1", "bread": "X3", "scribe_tools": "Y5"
    }
    TESSERACT_CONFIGS = {
        'ancient_greek': "--psm 6 --oem 1 -c preserve_interword_spaces=1",
        'standard_greek': "--psm 6 --oem 1",
        'fallback': "--psm 3 --oem 1"
    }
    
    @property
    def CODE_TO_LABEL(self):
        return {v: k for k, v in self.GARDINER_MAP.items()}
