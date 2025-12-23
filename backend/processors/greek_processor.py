import pytesseract
import re
import os
import cv2
import numpy as np
from PIL import Image
from .base_processor import BaseScriptProcessor
from utils.text_utils import is_gibberish

class GreekProcessor(BaseScriptProcessor):
    def __init__(self, groq_client, references, clip_classifier):
        super().__init__(groq_client, references, clip_classifier)
        self.clip_classifier = clip_classifier
        self.setup_ancient_greek_ocr()
    
    def setup_ancient_greek_ocr(self):
        """Setup Ancient Greek OCR with specialized tessdata"""
        # Path to Ancient Greek tessdata (download from ancientgreekocr.org)
        self.ancient_greek_tessdata = os.path.join(
            os.path.dirname(__file__), 
            "..", "tessdata", "ancient-greek"
        )
        
        # Verify tessdata exists
        if os.path.exists(self.ancient_greek_tessdata):
            print(f"[INFO] Ancient Greek tessdata found: {self.ancient_greek_tessdata}")
        else:
            print(f"[WARN] Ancient Greek tessdata not found at: {self.ancient_greek_tessdata}")
            print("[INFO] Download from: https://ancientgreekocr.org")
    def detect_script(self, image_path):
        """Simplified detection - Groq Vision handles main classification"""
        try:
            # Check if Ancient Greek OCR is available
            grc_file = os.path.join(self.ancient_greek_tessdata, "grc.traineddata")
            if not os.path.exists(grc_file):
                print("[INFO] Ancient Greek OCR not available")
                return False, 0.5
            
            # If called by Groq Vision classification, accept with high confidence
            print("[INFO] Greek processor activated by Groq Vision (Llama-4-Scout)")
            return True, 0.95
            
        except Exception as e:
            print(f"[ERROR] Greek detection failed: {e}")
            return False, 0.0

    
    def _quick_greek_ocr_test(self, image_path):
        """Quick OCR test to validate Greek content"""
        try:
            # Quick test with small image crop
            image = Image.open(image_path)
            # Take center crop for testing
            w, h = image.size
            crop_box = (w//4, h//4, 3*w//4, 3*h//4)
            test_crop = image.crop(crop_box)
            
            # Test with standard Greek OCR
            test_text = pytesseract.image_to_string(test_crop, lang="ell")
            greek_char_count = self._count_greek_chars(test_text or "")
            
            # If we find Greek characters, it's likely Greek
            return greek_char_count >= 3
            
        except Exception:
            return False
    
    def extract_text(self, image_path):
        """Enhanced Greek text extraction with Ancient Greek OCR"""
        try:
            image = Image.open(image_path)
            
            # Quick pre-check: if CLIP strongly suggests Latin, skip Greek OCR
            try:
                label, score = self.clip_classifier.classify_script_type(image)
                if label and "latin" in label.lower() and score > 0.7:
                    print("[INFO] CLIP suggests Latin script, skipping Greek OCR")
                    return ""
            except Exception:
                pass
            
            # Method 1: Ancient Greek OCR (if available and safe)
            grc_file = os.path.join(self.ancient_greek_tessdata, "grc.traineddata")
            if os.path.exists(grc_file):
                ancient_greek_text = self._extract_with_ancient_greek_ocr(image)
                if ancient_greek_text and self._validate_greek_text(ancient_greek_text):
                    print("[INFO] Using Ancient Greek OCR result")
                    return ancient_greek_text
            
            # Method 2: Standard Greek OCR
            standard_greek_text = self._extract_with_standard_greek_ocr(image)
            if standard_greek_text and self._validate_greek_text(standard_greek_text):
                print("[INFO] Using standard Greek OCR result")
                return standard_greek_text
            
            # Method 3: Final validation - if no good Greek text found, return empty
            print("[INFO] No valid Greek text detected")
            return ""
        
        except Exception as e:
            print(f"[ERROR] Greek text extraction failed: {e}")
            return ""

    
    def _extract_with_ancient_greek_ocr(self, image):
        """Extract using specialized Ancient Greek OCR"""
        try:
            # Save original tessdata path
            original_tessdata = os.environ.get("TESSDATA_PREFIX", "")
            
            # Set tessdata path properly (fix the path format)
            if os.path.exists(self.ancient_greek_tessdata):
                # Ensure proper path format without trailing quotes
                clean_path = str(self.ancient_greek_tessdata).replace('"', '')
                os.environ["TESSDATA_PREFIX"] = clean_path
                print(f"[INFO] Set TESSDATA_PREFIX to: {clean_path}")
            else:
                print(f"[WARN] Ancient Greek tessdata not found at: {self.ancient_greek_tessdata}")
                return ""
            
            # Use ancient Greek language code 'grc' with optimized settings
            config = "--psm 6 --oem 1 -c preserve_interword_spaces=1"
            
            # Try ancient Greek language pack
            text = pytesseract.image_to_string(
                image, 
                lang="grc",  # Ancient Greek language code
                config=config
            )
            
            # Restore original tessdata path
            if original_tessdata:
                os.environ["TESSDATA_PREFIX"] = original_tessdata
            else:
                # Remove the environment variable if it wasn't set before
                if "TESSDATA_PREFIX" in os.environ:
                    del os.environ["TESSDATA_PREFIX"]
            
            return text.strip()
            
        except Exception as e:
            print(f"[WARN] Ancient Greek OCR failed: {e}")
            # Make sure to restore tessdata path even on error
            if 'original_tessdata' in locals() and original_tessdata:
                os.environ["TESSDATA_PREFIX"] = original_tessdata
            return ""

    
    def _extract_with_standard_greek_ocr(self, image):
        """Extract using standard Greek OCR with optimized settings"""
        try:
            # Multiple OCR attempts with different settings
            configs = [
                "--psm 6 --oem 1",  # Uniform text block
                "--psm 4 --oem 1",  # Single column text
                "--psm 3 --oem 1",  # Default, automatic page segmentation
                "--psm 8 --oem 1"   # Single word
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(
                        image,
                        lang="ell",  # Modern Greek
                        config=config
                    )
                    
                    if text and self._validate_greek_text(text):
                        return text.strip()
                        
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            print(f"[WARN] Standard Greek OCR failed: {e}")
            return ""
    
    def _extract_with_preprocessing(self, image):
        """Fallback extraction with image preprocessing"""
        try:
            # Convert PIL to CV2
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Image preprocessing for better OCR
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Try different preprocessing approaches
            preprocessed_images = [
                gray,  # Original grayscale
                cv2.GaussianBlur(gray, (1, 1), 0),  # Slight blur
                cv2.medianBlur(gray, 3),  # Noise reduction
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Adaptive threshold
            ]
            
            for processed_img in preprocessed_images:
                try:
                    pil_img = Image.fromarray(processed_img)
                    text = pytesseract.image_to_string(
                        pil_img,
                        lang="ell",
                        config="--psm 6 --oem 1"
                    )
                    
                    if self._validate_greek_text(text):
                        return text.strip()
                        
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            print(f"[WARN] Fallback Greek OCR failed: {e}")
            return ""
    
    def _count_greek_chars(self, text):
        """Count Greek Unicode characters including polytonic marks"""
        if not text:
            return 0
            
        def is_greek_char(ch):
            o = ord(ch)
            # Greek and Coptic (0x0370-0x03FF)
            # Greek Extended (0x1F00-0x1FFF) - includes polytonic marks
            return (0x0370 <= o <= 0x03FF) or (0x1F00 <= o <= 0x1FFF)
        
        return sum(is_greek_char(ch) for ch in text)
    
    def _validate_greek_text(self, text):
        """Validate if text contains meaningful Greek content"""
        if not text or len(text.strip()) < 3:
            return False
        
        # Count Greek characters
        greek_char_count = self._count_greek_chars(text)
        total_chars = len(re.sub(r'\s+', '', text))
        
        if total_chars == 0:
            return False
        
        # Check for Latin characters (should reject if too many)
        latin_chars = sum(c.isalpha() and c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in text)
        latin_ratio = latin_chars / total_chars if total_chars > 0 else 0
        
        # If text is mostly Latin characters, reject it
        if latin_ratio > 0.8 and greek_char_count < 3:
            print(f"[INFO] Rejecting text as Greek - too many Latin chars: {latin_ratio:.2f}")
            return False
        
        # At least 20% should be Greek characters, or minimum 5 Greek chars
        greek_ratio = greek_char_count / total_chars
        
        return greek_char_count >= 5 or greek_ratio >= 0.20

    
    def _extract_distinct_terms(self, text):
        """Extract distinct Greek terms from text"""
        if not text:
            return []
        
        # Find Greek words (including those with diacritical marks)
        tokens = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
        
        def is_greek_word(word):
            return any((0x0370 <= ord(ch) <= 0x03FF) or (0x1F00 <= ord(ch) <= 0x1FFF) 
                      for ch in word)
        
        distinct_terms = []
        seen = set()
        
        for token in tokens:
            if len(token) < 2:  # Skip single characters
                continue
                
            if is_greek_word(token):
                normalized = token.lower()
                if normalized not in seen:
                    distinct_terms.append(token)
                    seen.add(normalized)
        
        return distinct_terms[:20]  # Limit to 20 terms
    
    def process_text(self, greek_text):
        """Process extracted Greek text"""
        if not greek_text:
            return {"text": "", "terms": [], "char_analysis": {}, "validation": {}}
        
        # Extract distinct terms
        terms = self._extract_distinct_terms(greek_text)
        
        # Character analysis
        char_analysis = {
            "total_chars": len(greek_text),
            "greek_chars": self._count_greek_chars(greek_text),
            "unique_chars": len(set(greek_text)),
            "words": len(greek_text.split())
        }
        
        # Validation metrics
        validation = {
            "has_polytonic": self._has_polytonic_marks(greek_text),
            "greek_ratio": char_analysis["greek_chars"] / max(1, char_analysis["total_chars"]),
            "quality_score": self._calculate_quality_score(greek_text)
        }
        
        return {
            "text": greek_text,
            "terms": terms,
            "char_analysis": char_analysis,
            "validation": validation
        }
    
    def _has_polytonic_marks(self, text):
        """Check if text contains polytonic Greek marks"""
        # Greek Extended block contains polytonic diacritical marks
        return any(0x1F00 <= ord(ch) <= 0x1FFF for ch in text)
    
    def _calculate_quality_score(self, text):
        """Calculate a quality score for the extracted text"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Base score from Greek character ratio
        greek_ratio = self._count_greek_chars(text) / max(1, len(text))
        score += greek_ratio * 0.4
        
        # Bonus for polytonic marks (indicates authentic ancient Greek)
        if self._has_polytonic_marks(text):
            score += 0.3
        
        # Penalty for too many non-alphabetic characters
        alpha_chars = sum(ch.isalpha() for ch in text)
        alpha_ratio = alpha_chars / max(1, len(text))
        score += alpha_ratio * 0.3
        
        return min(1.0, score)
    
    def generate_historical_context(self, processed_result):
        """Generate historical context for Greek text"""
        greek_text = processed_result.get("text", "")
        terms = processed_result.get("terms", [])
        
        # Generate Groq context
        groq_detail = self._generate_groq_context(greek_text)
        
        return {
            "uses_box": {
                "title": "Each symbol's possible use by the Greek people",
                "items": self._build_uses_list(greek_text)
            },
            "meaning_box": self._build_meaning_box(terms, groq_detail)
        }
    
    def _generate_groq_context(self, greek_text):
        """Generate contextual information using Groq"""
        if not self.groq_client.is_available():
            return "(Groq unavailable) Context generation requires GROQ_API_KEY and groq package."
        
        prompt = (
            f"This ancient Greek text was found: {greek_text}\n\n"
            "Write a concise, scholarly paragraph (6-10 sentences) giving cultural and historical context: textual tradition, "
            "possible meanings, links to Greek culture/myth/philosophy, manuscript practices (accents, breathings, ligatures, nomina sacra), "
            "and paleographic cues. Avoid repeating the prompt."
        )
        
        system_prompt = "You are an expert philologist of Ancient Greece. Provide concise, accurate scholarly context."
        
        return self.groq_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt
        ) or "(context unavailable due to Groq error)"
    
    def _build_uses_list(self, greek_text):
        """Build list of character uses"""
        notes = self.references.get("greek_symbol_notes", {}) or {}
        default_hint = self.references.get("greek_hint", 
            "If no specific character note is found, treat as lexical marker considering diacriticals (breathing marks, accents, vowel quantity) which affect pronunciation, meaning, and grammatical function in ancient Greek texts.")
        
        seen = set()
        items = []
        
        for ch in greek_text:
            if ch in seen or not ch.strip():
                continue
            seen.add(ch)
            
            if ch in notes:
                note = notes[ch]
            else:
                note = default_hint
                
            items.append(f"- {ch}: {note}")
        
        if not items:
            items.append("- —: " + default_hint)
        
        return items

    
    def _build_meaning_box(self, terms, groq_detail):
        """Build meaning interpretation box"""
        intro_lines = [
            "The lexical concentration suggests a connected passage with recurring words or themes, consistent with Greek manuscript traditions.",
            "Scribal features such as accents/breathings, abbreviations, and marginal cues guide reading and assist with dating and genre identification."
        ]
        
        points = [
            "• Presence of nomina sacra, lection signs, or ekphonetic marks indicates liturgical usage; scholia imply classroom or commentary context.",
            "• Orthographic variation (e.g., iotacism) and common ligatures inform palaeographic placement and regional practice.",
        ]
        
        if groq_detail and isinstance(groq_detail, str) and groq_detail.strip():
            points.append(groq_detail.strip())
        
        return {
            "title": "Possible meaning:",
            "intro_lines": intro_lines,
            "frequent_label": "Key terms noted",
            "frequent": terms[:10],
            "points": points
        }
    
    def generate_story(self, processed_result):
        """Generate creative story for Greek text"""
        greek_text = processed_result.get("text", "")
        
        if not self.groq_client.is_available():
            return "Groq client unavailable, cannot generate story."
        
        styles = [
            "as an epic poem told by a travelling rhapsode",
            "as a prophecy inscribed on the Oracle at Delphi",
            "as a philosophical dialogue in the Academy",
            "as a myth recounted by ancient storytellers",
            "as a recovered scroll from the Library of Alexandria",
            "as a hymn sung in honor of the gods"
        ]
        
        import random
        chosen_style = random.choice(styles)
        seed = random.randint(1000, 9999)
        
        prompt = (
            f"The following ancient Greek text was found: {greek_text}\n\n"
            f"Create a long, vivid, imaginative story from ancient Greek times "
            f"based on this Greek text. Write it as one rich paragraph with "
            f"much detail, mystery, and cultural atmosphere. At least 200 words.\n\n"
            f"Creative seed: {seed}\n"
            f"Write a detailed, imaginative myth-like story {chosen_style}. "
            "Include multiple characters, rich imagery, and scenes. "
            "Avoid repetition and keep it unpredictable."
        )
        
        system_prompt = "You are a learned ancient Greek storyteller and scholar of Hellenic culture."
        
        story = self.groq_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        if not story or is_gibberish(story):
            return "Failed to create quality story; the ancient texts remain silent."
        
        return story
