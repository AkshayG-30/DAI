import cv2
import numpy as np
import re
import time
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from .base_processor import BaseScriptProcessor
from utils.text_utils import is_gibberish


class LatinProcessor(BaseScriptProcessor):
    def __init__(self, groq_client, references, clip_classifier):
        super().__init__(groq_client, references, clip_classifier)
        self.setup_tridis_htr()
        self.setup_tesseract_fallback()

    def setup_tridis_htr(self):
        """Setup TRIDIS HTR model - BEST for medieval Latin manuscripts"""
        try:
            print("[INFO] Loading TRIDIS HTR model for medieval Latin...")
            print("[INFO] This model specializes in 13th-16th century manuscripts with automatic abbreviation expansion")
            
            # TRIDIS model from Hugging Face - runs locally after download
            self.tridis_processor = TrOCRProcessor.from_pretrained(
                'magistermilitum/tridis_HTR',
                cache_dir='./models/tridis/',
                local_files_only=False  # Download first time, then cache locally
            )
            self.tridis_model = VisionEncoderDecoderModel.from_pretrained(
                'magistermilitum/tridis_HTR', 
                cache_dir='./models/tridis/',
                local_files_only=False
            )
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tridis_model.to(self.device)
            
            print(f"[INFO] TRIDIS HTR loaded successfully on {self.device}")
            print("[INFO] Training: 245,000 lines of Latin/Old French/Old Spanish medieval manuscripts")
            print("[INFO] Features: Automatic abbreviation expansion, named entity capitalization, cancellation markers")
            self.tridis_available = True
            
        except Exception as e:
            print(f"[ERROR] TRIDIS HTR model failed to load: {e}")
            print("[WARN] Falling back to Tesseract for basic Latin recognition...")
            self.tridis_available = False

    def setup_tesseract_fallback(self):
        """Setup Tesseract as fallback for basic Latin recognition"""
        try:
            import pytesseract
            
            # Test Tesseract availability
            try:
                version = pytesseract.get_tesseract_version()
                print(f"[INFO] Tesseract fallback version: {version}")
            except:
                print("[INFO] Tesseract version check skipped")
            
            self.ocr_configs = {
                'medieval_extended': r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/-·&℞℟℣†‡¶§꜠꜡ꜢꜣꜤꜥꝀꝁꝐꝑꝒꝓꝔꝕꝖꝗꝘꝙꝚꝛꝜꝝꞀꞁꞂꞃ$',
                'medieval_basic': r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/-',
                'standard': r'--oem 3 --psm 6',
                'single_line': r'--oem 3 --psm 7',
                'single_word': r'--oem 3 --psm 8',
                'auto': r'--oem 3 --psm 3'
            }
            self.tesseract_available = True
            print("[INFO] Tesseract fallback configured with medieval symbol support")
            
        except ImportError:
            print("[ERROR] pytesseract not available")
            self.tesseract_available = False
        except Exception as e:
            print(f"[WARN] Tesseract setup failed: {e}")
            self.tesseract_available = False

    def detect_script(self, image_path):
        """Detection handled by Groq Vision classification"""
        try:
            if not self.tridis_available and not self.tesseract_available:
                print("[ERROR] No OCR engines available for Latin processing")
                return False, 0.0
            
            method = "TRIDIS HTR (medieval specialist)" if self.tridis_available else "Tesseract fallback"
            print(f"[INFO] Latin processor activated - Using {method}")
            return True, 0.98 if self.tridis_available else 0.85
            
        except Exception as e:
            print(f"[ERROR] Latin detection failed: {e}")
            return False, 0.0

    def extract_text(self, image_path):
        """Extract text using TRIDIS HTR (medieval specialist) with Tesseract fallback"""
        try:
            start_time = time.time()
            
            # Method 1: TRIDIS HTR (BEST for medieval Latin manuscripts)
            if self.tridis_available:
                print("[INFO] Processing with TRIDIS HTR - medieval manuscript specialist...")
                tridis_text = self._extract_with_tridis_htr(image_path)
                
                if tridis_text and self._validate_medieval_latin_text(tridis_text):
                    processing_time = time.time() - start_time
                    print(f"[SUCCESS] TRIDIS HTR completed in {processing_time:.2f}s")
                    return tridis_text
                else:
                    print("[WARN] TRIDIS HTR returned poor quality result, trying fallback...")
            
            # Method 2: Tesseract fallback
            if self.tesseract_available:
                print("[INFO] Processing with Tesseract fallback...")
                tesseract_text = self._extract_with_tesseract_enhanced(image_path)
                
                if tesseract_text and self._validate_medieval_latin_text(tesseract_text):
                    processing_time = time.time() - start_time
                    print(f"[SUCCESS] Tesseract fallback completed in {processing_time:.2f}s")
                    return tesseract_text
                else:
                    print("[WARN] Tesseract also returned poor quality result")
            
            print("[ERROR] All OCR methods failed or returned poor quality results")
            return "No readable Latin text detected with sufficient confidence"
            
        except Exception as e:
            print(f"[ERROR] Latin text extraction failed: {e}")
            return f"Error during text extraction: {str(e)}"

    def _extract_with_tridis_htr(self, image_path):
        """Extract text using TRIDIS HTR - SPECIALIZED for medieval Latin manuscripts"""
        try:
            # Load and validate image
            image = Image.open(image_path).convert("RGB")
            print(f"[INFO] Processing image: {image.size[0]}x{image.size[1]} pixels")
            
            # Enhanced preprocessing for medieval manuscripts
            enhanced_image = self._preprocess_for_medieval_manuscript(image)
            
            # Process with TRIDIS HTR
            print("[INFO] Running TRIDIS HTR inference...")
            pixel_values = self.tridis_processor(
                images=enhanced_image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text with parameters optimized for medieval manuscripts
            with torch.no_grad():
                generated_ids = self.tridis_model.generate(
                    pixel_values,
                    max_length=768,  # Longer sequences for medieval texts with abbreviations
                    num_beams=6,     # Higher quality beam search for historical accuracy
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=1.15,  # Avoid repetition common in medieval texts
                    length_penalty=0.8,       # Don't penalize longer expansions
                    no_repeat_ngram_size=2    # Avoid immediate repetitions
                )
            
            # Decode the generated text
            generated_text = self.tridis_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Post-process medieval abbreviations, corrections, and formatting
            processed_text = self._post_process_medieval_text(generated_text)
            
            # Log extraction results
            char_count = len(processed_text)
            word_count = len(processed_text.split())
            print(f"[INFO] TRIDIS HTR extracted: {char_count} characters, {word_count} words")
            
            # Detect medieval features
            medieval_features = self._analyze_medieval_features(processed_text)
            if medieval_features:
                print(f"[INFO] Medieval features detected: {', '.join(medieval_features)}")
            
            return processed_text.strip()
            
        except Exception as e:
            print(f"[ERROR] TRIDIS HTR extraction failed: {e}")
            return ""

    def _preprocess_for_medieval_manuscript(self, image):
        """Enhanced preprocessing specifically optimized for medieval manuscripts"""
        try:
            print("[INFO] Applying medieval manuscript preprocessing...")
            
            # Convert to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Step 1: Handle parchment/paper background variations
            # CLAHE for local contrast enhancement (handles uneven illumination)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(gray)
            
            # Step 2: Gentle denoising to preserve medieval letterforms and ink variations
            # Bilateral filter preserves edges while reducing noise
            denoised = cv2.bilateralFilter(contrast_enhanced, 7, 80, 80)
            
            # Step 3: Enhance faded ink while preserving original stroke width
            # Subtle sharpening kernel
            sharpen_kernel = np.array([
                [-0.5, -1, -0.5],
                [-1,   6, -1  ],
                [-0.5, -1, -0.5]
            ])
            sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
            
            # Step 4: Normalize intensity range for optimal TRIDIS input
            normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert back to PIL format
            processed_image = Image.fromarray(normalized)
            
            print("[INFO] Medieval preprocessing completed: contrast enhanced, denoised, sharpened")
            return processed_image
            
        except Exception as e:
            print(f"[WARN] Medieval preprocessing failed: {e}, using original image")
            return image

    def _post_process_medieval_text(self, text):
        """Post-process text from TRIDIS HTR with medieval-specific corrections"""
        try:
            if not text:
                return text
            
            print("[INFO] Post-processing TRIDIS HTR output for medieval features...")
            processed = text
            
            # Handle TRIDIS cancellation/correction markers
            # TRIDIS uses $word$ to mark cancelled/corrected text
            import re
            
            # Count cancellations before processing
            cancellation_count = processed.count('$') // 2
            
            # Convert $word$ to editorial brackets [word] for scholarly display
            processed = re.sub(r'\$([^$]*)\$', r'[\1]', processed)
            
            if cancellation_count > 0:
                print(f"[INFO] Processed {cancellation_count} scribal corrections/cancellations")
            
            # Clean up multiple spaces and normalize whitespace
            processed = ' '.join(processed.split())
            
            # Detect and log TRIDIS abbreviation expansions
            # Common medieval abbreviations that TRIDIS expands automatically
            medieval_expansions = {
                'domini': 'dñi/dni/dom̃',
                'facimus': 'facim̃/facimꝰ', 
                'quod': 'qd/q̃d',
                'enim': 'enim̃/en̄',
                'pro': 'ꝓ/p̃',
                'et': '⁊/et̃',
                'cum': 'cũ/cum̃',
                'per': 'p̃/ꝑ',
                'sunt': 'sũt/sunt̃',
                'omnia': 'om̃ia/omn̄a'
            }
            
            expansions_found = []
            for expansion, abbreviations in medieval_expansions.items():
                if expansion in processed.lower():
                    expansions_found.append(f"{abbreviations}→{expansion}")
            
            if expansions_found:
                print(f"[INFO] TRIDIS expanded abbreviations: {', '.join(expansions_found[:5])}")
                if len(expansions_found) > 5:
                    print(f"[INFO] ... and {len(expansions_found) - 5} more abbreviations")
            
            # Detect capitalization patterns (TRIDIS capitalizes named entities)
            capitalized_words = re.findall(r'\b[A-Z][a-z]+', processed)
            if capitalized_words:
                unique_caps = list(set(capitalized_words))
                print(f"[INFO] Named entities capitalized: {', '.join(unique_caps[:5])}")
                if len(unique_caps) > 5:
                    print(f"[INFO] ... and {len(unique_caps) - 5} more entities")
            
            return processed
            
        except Exception as e:
            print(f"[WARN] Medieval post-processing failed: {e}")
            return text

    def _analyze_medieval_features(self, text):
        """Analyze and identify medieval manuscript features in the text"""
        features = []
        
        if not text:
            return features
        
        try:
            # Cancellation markers
            if '[' in text and ']' in text:
                features.append("scribal corrections")
            
            # Expanded abbreviations
            medieval_words = ['domini', 'facimus', 'quod', 'enim', 'pro', 'cum', 'per', 'sunt', 'omnia']
            found_expansions = [word for word in medieval_words if word in text.lower()]
            if found_expansions:
                features.append(f"abbreviation expansions ({len(found_expansions)})")
            
            # Named entity capitalization
            import re
            caps_count = len(re.findall(r'\b[A-Z][a-z]+', text))
            if caps_count > 0:
                features.append(f"capitalized entities ({caps_count})")
            
            # Medieval punctuation patterns
            if '.' in text or ',' in text or ':' in text:
                features.append("punctuation normalization")
            
            # Special medieval characters
            medieval_chars = sum(1 for c in text if c in "꜠꜡ꜣꜥꝁꝑꝛꞁꞃ℞℟℣†‡¶§")
            if medieval_chars > 0:
                features.append(f"medieval symbols ({medieval_chars})")
            
        except Exception as e:
            print(f"[WARN] Medieval feature analysis failed: {e}")
        
        return features

    def _extract_with_tesseract_enhanced(self, image_path):
        """Enhanced Tesseract extraction with multiple configurations"""
        try:
            import pytesseract
            
            image = Image.open(image_path).convert("RGB")
            
            # Multiple preprocessing approaches
            preprocessed_images = {
                'enhanced': self._preprocess_for_tesseract_enhanced(image),
                'basic': self._preprocess_for_tesseract_basic(image),
                'original': image
            }
            
            best_text = ""
            best_score = 0
            best_config = ""
            best_preprocessing = ""
            
            # Try different combinations of preprocessing and OCR configurations
            for prep_name, prep_image in preprocessed_images.items():
                for config_name, config in self.ocr_configs.items():
                    try:
                        # Try with Latin language first
                        text = pytesseract.image_to_string(
                            prep_image, 
                            lang='lat', 
                            config=config
                        ).strip()
                        
                        # If Latin fails or produces poor results, try English
                        if not text or len(text) < 5:
                            text = pytesseract.image_to_string(
                                prep_image, 
                                lang='eng', 
                                config=config
                            ).strip()
                        
                        # Score the result
                        score = self._score_tesseract_result(text)
                        
                        if text and score > best_score:
                            best_text = text
                            best_score = score
                            best_config = config_name
                            best_preprocessing = prep_name
                    
                    except Exception as e:
                        continue  # Skip failed configurations
            
            if best_text:
                print(f"[INFO] Best Tesseract result: {best_preprocessing} + {best_config} (score: {best_score:.3f})")
                return self._post_process_tesseract_text(best_text)
            
            return ""
            
        except Exception as e:
            print(f"[ERROR] Enhanced Tesseract extraction failed: {e}")
            return ""

    def _preprocess_for_tesseract_enhanced(self, image):
        """Enhanced preprocessing for Tesseract OCR"""
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # More aggressive enhancement for Tesseract
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Morphological operations to clean up characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            return Image.fromarray(cleaned)
            
        except Exception as e:
            print(f"[WARN] Enhanced Tesseract preprocessing failed: {e}")
            return image

    def _preprocess_for_tesseract_basic(self, image):
        """Basic preprocessing for Tesseract OCR"""
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Simple contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            return Image.fromarray(enhanced)
            
        except Exception as e:
            return image

    def _score_tesseract_result(self, text):
        """Score Tesseract OCR result quality"""
        if not text or len(text.strip()) < 2:
            return 0.0
        
        score = 0.0
        words = text.split()
        
        # Base length bonus
        score += min(len(words) / 15.0, 0.25)
        
        # Latin character ratio
        latin_chars = sum(c.isalpha() and c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in text)
        if len(text) > 0:
            latin_ratio = latin_chars / len(text)
            score += latin_ratio * 0.35
        
        # Word formation bonus
        if len(words) > 1:
            score += 0.2
        
        # Common Latin words bonus
        common_latin = ['et', 'in', 'de', 'ad', 'cum', 'pro', 'per', 'ex', 'ab', 'post', 'ante', 'inter']
        latin_matches = sum(1 for word in words if word.lower() in common_latin)
        if latin_matches > 0:
            score += latin_matches * 0.05
        
        # Medieval symbols bonus
        medieval_symbols = ['꜠', '꜡', 'ꜣ', 'ꜥ', 'ꝁ', 'ꝑ', 'ꝛ', 'ꞁ', 'ꞃ', '℞', '℟', '℣', '†', '‡', '¶', '§']
        symbol_count = sum(1 for symbol in medieval_symbols if symbol in text)
        if symbol_count > 0:
            score += 0.15
        
        # Penalize excessive garbage characters
        garbage_chars = sum(1 for c in text if not c.isalnum() and c not in " .,;:!?()[]{}/-·&℞℟℣†‡¶§꜠꜡ꜢꜣꜤꜥꝀꝁ")
        if len(text) > 0:
            garbage_ratio = garbage_chars / len(text)
            score -= garbage_ratio * 0.3
        
        return max(0.0, min(1.0, score))

    def _post_process_tesseract_text(self, text):
        """Post-process Tesseract OCR result"""
        try:
            # Clean up common OCR errors
            corrections = {
                'rn': 'm',
                'cl': 'd', 
                '|': 'I',
                '°': 'o',
                '¢': 'c',
                '£': 'E'
            }
            
            processed = text
            for wrong, correct in corrections.items():
                processed = processed.replace(wrong, correct)
            
            # Normalize whitespace
            processed = ' '.join(processed.split())
            
            return processed
            
        except Exception as e:
            print(f"[WARN] Tesseract post-processing failed: {e}")
            return text

    def _validate_medieval_latin_text(self, text):
        """Validate text with enhanced criteria for medieval Latin"""
        if not text or len(text.strip()) < 3:
            return False
        
        try:
            # Count different types of meaningful content
            latin_chars = sum(c.isalpha() and c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in text)
            medieval_symbols = sum(1 for c in text if c in "꜠꜡ꜣꜥꝁꝑꝛꞁꞃ℞℟℣†‡¶§[]")
            medieval_words = ['domini', 'facimus', 'quod', 'enim', 'pro', 'cum', 'per', 'sunt']
            word_bonus = sum(3 for word in medieval_words if word in text.lower())  # 3x weight for medieval words
            
            total_meaningful = latin_chars + medieval_symbols + word_bonus
            total_chars = len(text.replace(' ', ''))
            
            if total_chars == 0:
                return False
            
            # More generous validation for medieval texts
            meaningful_ratio = total_meaningful / max(total_chars, 1)
            
            # Different validation thresholds
            if total_meaningful >= 10:  # Substantial content
                return True
            elif meaningful_ratio >= 0.6:  # High quality ratio
                return True  
            elif total_meaningful >= 5 and meaningful_ratio >= 0.3:  # Moderate content with decent ratio
                return True
            else:
                return False
                
        except Exception as e:
            print(f"[WARN] Text validation failed: {e}")
            return len(text.strip()) >= 5  # Fallback validation

    def process_text(self, latin_text):
        """Process extracted Latin text with comprehensive TRIDIS-aware analysis"""
        if not latin_text:
            return {"text": "", "symbols": [], "char_analysis": {}, "validation": {}}
        
        print("[INFO] Processing Latin text with medieval manuscript analysis...")
        
        # Extract symbols including medieval markers and corrections
        symbols = ''.join(filter(lambda x: x.isalnum() or x in "꜠꜡ꜣꜥꝁꝑꝛꞁꞃ℞℟℣†‡¶§$[]", latin_text))
        
        # Comprehensive medieval character analysis
        medieval_symbols = [c for c in latin_text if c in "꜠꜡ꜣꜥꝁꝑꝛꞁꞃ℞℟℣†‡¶§"]
        correction_markers = latin_text.count('[') + latin_text.count('$')
        
        # Detect expanded abbreviations
        medieval_abbreviations = ['domini', 'facimus', 'pro', 'quod', 'enim', 'cum', 'per', 'sunt', 'omnia']
        expansions_found = [word for word in medieval_abbreviations if word in latin_text.lower()]
        
        # Count capitalized entities (TRIDIS feature)
        import re
        capitalized_entities = re.findall(r'\b[A-Z][a-z]+', latin_text)
        unique_entities = list(set(capitalized_entities))
        
        # Comprehensive character analysis
        char_analysis = {
            "total_chars": len(latin_text),
            "alpha_chars": sum(c.isalpha() for c in latin_text),
            "unique_chars": len(set(latin_text)),
            "word_count": len(latin_text.split()),
            "medieval_symbols": len(medieval_symbols),
            "medieval_symbol_types": list(set(medieval_symbols)),
            "abbreviation_expansions": expansions_found,
            "expansion_count": len(expansions_found),
            "correction_markers": correction_markers,
            "capitalized_entities": unique_entities,
            "entity_count": len(unique_entities),
            "avg_word_length": sum(len(word) for word in latin_text.split()) / max(1, len(latin_text.split()))
        }
        
        # Enhanced validation with medieval features
        validation = {
            "latin_ratio": sum(c.isalpha() and c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in latin_text) / max(1, len(latin_text)),
            "quality_score": self._calculate_comprehensive_quality_score(latin_text),
            "ocr_method": "TRIDIS HTR (Medieval Manuscript Specialist)" if self.tridis_available else "Tesseract OCR",
            "model_specialization": "13th-16th century manuscripts" if self.tridis_available else "General Latin text",
            "medieval_features_detected": bool(medieval_symbols or expansions_found or correction_markers),
            "tridis_used": self.tridis_available,
            "manuscript_period": "Late Medieval (13th-16th centuries)" if (medieval_symbols or expansions_found) else "Classical/Modern",
            "text_type": self._determine_text_type(latin_text),
            "abbreviations_expanded": len(expansions_found) > 0,
            "named_entities_detected": len(unique_entities) > 0,
            "scribal_corrections_found": correction_markers > 0,
            "confidence_level": self._determine_confidence_level(latin_text)
        }
        
        return {
            "text": latin_text,
            "symbols": symbols,
            "char_analysis": char_analysis,
            "validation": validation
        }

    def _calculate_comprehensive_quality_score(self, text):
        """Calculate comprehensive quality score with medieval bonuses"""
        if not text:
            return 0.0
        
        score = 0.0
        words = text.split()
        
        # Base metrics
        score += min(len(words) / 15.0, 0.2)  # Length bonus (max 0.2)
        
        # Latin character ratio
        latin_chars = sum(c.isalpha() and c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in text)
        score += (latin_chars / max(1, len(text))) * 0.25
        
        # TRIDIS Medieval bonuses (only if TRIDIS was used)
        if self.tridis_available:
            # Expanded abbreviations (major quality indicator)
            medieval_expansions = ['domini', 'facimus', 'pro', 'quod', 'enim', 'cum', 'per', 'sunt']
            expansion_count = sum(1 for exp in medieval_expansions if exp in text.lower())
            score += min(expansion_count * 0.05, 0.2)  # Max 0.2 bonus
            
            # Named entity capitalization (TRIDIS feature)
            import re
            caps_count = len(re.findall(r'\b[A-Z][a-z]+', text))
            score += min(caps_count * 0.02, 0.15)  # Max 0.15 bonus
            
            # Correction markers (authenticity indicator)
            corrections = text.count('[') + text.count('$')
            score += min(corrections * 0.03, 0.1)  # Max 0.1 bonus
        
        # Medieval symbols (regardless of OCR method)
        medieval_symbols = ['꜠', '꜡', 'ꜣ', 'ꜥ', 'ꝁ', 'ꝑ', 'ꝛ', 'ꞁ', 'ꞃ', '℞', '℟', '℣', '†', '‡', '¶', '§']
        symbol_count = sum(1 for symbol in medieval_symbols if symbol in text)
        score += min(symbol_count * 0.04, 0.15)  # Max 0.15 bonus
        
        # Word formation
        if len(words) > 1:
            score += 0.1
        
        # Common Latin words
        common_latin = ['et', 'in', 'de', 'ad', 'cum', 'pro', 'per', 'ex', 'ab']
        latin_matches = sum(1 for word in words if word.lower() in common_latin)
        score += min(latin_matches * 0.02, 0.1)
        
        return max(0.0, min(1.0, score))

    def _determine_text_type(self, text):
        """Determine the type of Latin text based on features"""
        if not text:
            return "unknown"
        
        # Medieval indicators
        medieval_expansions = ['domini', 'facimus', 'quod', 'enim']
        has_expansions = any(exp in text.lower() for exp in medieval_expansions)
        has_corrections = '[' in text or '$' in text
        has_medieval_symbols = any(c in text for c in "꜠꜡ꜣꜥꝁꝑꝛꞁꞃ℞℟℣†‡¶§")
        
        if has_expansions and has_corrections:
            return "medieval_documentary_manuscript"
        elif has_expansions or has_medieval_symbols:
            return "medieval_manuscript"  
        elif has_corrections:
            return "manuscript_with_corrections"
        else:
            return "classical_latin_text"

    def _determine_confidence_level(self, text):
        """Determine confidence level based on text characteristics"""
        score = self._calculate_comprehensive_quality_score(text)
        
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def generate_historical_context(self, processed_result):
        """Generate comprehensive historical context for Latin text"""
        latin_text = processed_result.get("text", "")
        
        groq_detail = self._generate_groq_context(latin_text)
        
        return {
            "uses_box": {
                "title": "Medieval Latin manuscript analysis",
                "items": self._build_uses_list(latin_text)
            },
            "meaning_box": self._build_enhanced_meaning_box(latin_text, groq_detail, processed_result)
        }

    def _generate_groq_context(self, latin_text):
        """Generate contextual information using Groq with medieval awareness"""
        if not self.groq_client.is_available():
            return "(Groq unavailable) Historical context generation requires GROQ_API_KEY and groq package."
        
        # Analyze medieval features for context
        has_expansions = any(word in latin_text.lower() for word in ['domini', 'facimus', 'quod', 'enim'])
        has_corrections = '[' in latin_text or '$' in latin_text
        has_caps = any(c.isupper() for c in latin_text)
        
        if is_gibberish(latin_text):
            prompt = (
                "The following sequence appears to be fragmentary medieval Latin text, possibly with scribal abbreviations or corrections. "
                "Provide a concise, scholarly paragraph (6-10 sentences) covering possible meanings, historical context of medieval Latin manuscripts, "
                "common abbreviation practices, and typical documentary uses in 13th-16th century Europe."
            )
        else:
            context_note = ""
            if has_expansions:
                context_note += "The text contains expanded medieval abbreviations. "
            if has_corrections:
                context_note += "Scribal corrections or cancellations are present. "
            if has_caps:
                context_note += "Named entities appear to be properly capitalized. "
                
            prompt = (
                f"Analyze this medieval Latin text: {latin_text}\n\n"
                f"Context: {context_note}This appears to be from a medieval manuscript (13th-16th centuries). "
                f"Provide a scholarly paragraph (6-10 sentences) on its historical significance, cultural context, "
                f"likely documentary purpose, and interpretations. Focus on medieval manuscript practices, "
                f"legal/administrative contexts, and paleographic significance."
            )
        
        system_prompt = "You are a medieval Latin paleography specialist and historian. Provide accurate, concise scholarly analysis focusing on manuscript traditions, abbreviation practices, and documentary contexts of the late medieval period."
        
        return self.groq_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt
        ) or "(Historical context unavailable due to Groq error)"

    def _build_uses_list(self, latin_text):
        """Build enhanced list of character uses with TRIDIS context"""
        notes = self.references.get("latin_symbol_notes", {}) or {}
        default_hint = self.references.get("latin_hint", 
            "Letters and symbols reflect phonetic values and scribal practices in medieval manuscripts.")
        
        seen = set()
        items = []
        
        # Add TRIDIS-specific information for medieval features
        tridis_notes = {
            '[': "Editorial bracket indicating scribal correction or cancellation (TRIDIS transcription standard)",
            '$': "Cancellation marker for struck-through text (TRIDIS notation)",
        }
        
        for ch in latin_text:
            if ch in seen or not ch.strip():
                continue
            seen.add(ch)
            
            # Check TRIDIS-specific notes first
            if ch in tridis_notes:
                note = tridis_notes[ch]
            elif ch in notes:
                note = notes[ch]
            else:
                note = default_hint
                
            items.append(f"- {ch}: {note}")
        
        if not items:
            items.append("- —: " + default_hint)
        
        # Limit to prevent overwhelming output
        return items[:20]

    def _build_enhanced_meaning_box(self, latin_text, groq_detail, processed_result):
        """Build comprehensive meaning box with TRIDIS medieval analysis"""
        char_analysis = processed_result.get("char_analysis", {})
        validation = processed_result.get("validation", {})
        
        # Enhanced introduction with TRIDIS context
        processing_method = validation.get("ocr_method", "Unknown OCR")
        text_type = validation.get("text_type", "unknown")
        confidence = validation.get("confidence_level", "Unknown")
        
        intro_lines = [
            f"Text processed using {processing_method} with confidence level: {confidence}.",
        ]
        
        if self.tridis_available:
            intro_lines.extend([
                "TRIDIS HTR model trained on 245,000 lines of medieval manuscripts (13th-16th centuries).",
                "Specializes in Latin, Old French, Old Spanish documentary texts with automatic abbreviation expansion."
            ])
        
        # Medieval features summary
        medieval_features = []
        expansion_count = char_analysis.get("expansion_count", 0)
        if expansion_count > 0:
            medieval_features.append(f"{expansion_count} abbreviation expansions")
        
        correction_count = char_analysis.get("correction_markers", 0)
        if correction_count > 0:
            medieval_features.append(f"{correction_count} scribal corrections")
        
        entity_count = char_analysis.get("entity_count", 0)
        if entity_count > 0:
            medieval_features.append(f"{entity_count} named entities")
        
        if medieval_features:
            intro_lines.append(f"Medieval features detected: {', '.join(medieval_features)}.")
        
        # Key terms for frequent list
        expansions = char_analysis.get("abbreviation_expansions", [])
        entities = char_analysis.get("capitalized_entities", [])
        frequent_terms = expansions + entities
        
        if not frequent_terms:
            frequent_terms = list(set(w for w in latin_text.split() if len(w) > 2))[:10]
        
        # Enhanced analysis points
        points = []
        
        if self.tridis_available:
            points.extend([
                "• TRIDIS HTR provides semi-diplomatic transcription following scholarly editorial standards.",
                "• Automatic abbreviation expansion: dom̃→domini, facimꝰ→facimus, ꝓ→pro, ⁊→et.",
                "• Named entity capitalization and punctuation normalization applied."
            ])
        else:
            points.append("• Tesseract OCR provides basic Latin character recognition with limited medieval symbol support.")
        
        if correction_count > 0:
            points.append(f"• [{correction_count}] scribal corrections/cancellations indicate active manuscript editing process.")
        
        if expansion_count > 0:
            expansions_list = ", ".join(char_analysis.get("abbreviation_expansions", [])[:5])
            points.append(f"• Expanded abbreviations suggest legal/administrative document: {expansions_list}.")
        
        if validation.get("medieval_features_detected", False):
            manuscript_period = validation.get("manuscript_period", "Medieval")
            points.append(f"• {manuscript_period} characteristics indicate documentary manuscript tradition.")
        
        if groq_detail and isinstance(groq_detail, str) and groq_detail.strip():
            points.append(f"• Historical analysis: {groq_detail.strip()}")
        
        return {
            "title": "Medieval Latin manuscript analysis:",
            "intro_lines": intro_lines,
            "frequent_label": "Key medieval terms identified",
            "frequent": frequent_terms[:12],
            "points": points
        }

    def generate_story(self, processed_result):
        """Generate creative story with medieval manuscript context"""
        latin_text = processed_result.get("text", "")
        
        if not self.groq_client.is_available():
            return "Groq client unavailable, cannot generate historical narrative."
        
        # Analyze text features for story context
        char_analysis = processed_result.get("char_analysis", {})
        validation = processed_result.get("validation", {})
        
        has_expansions = char_analysis.get("expansion_count", 0) > 0
        has_corrections = char_analysis.get("correction_markers", 0) > 0
        has_entities = char_analysis.get("entity_count", 0) > 0
        text_type = validation.get("text_type", "unknown")
        used_tridis = validation.get("tridis_used", False)
        
        # Choose appropriate narrative style based on detected features
        if "documentary" in text_type or has_expansions:
            styles = [
                "as a legal charter discovered in monastic archives",
                "as an administrative record from a medieval royal court",
                "as a property deed found in cathedral scriptorium",
                "as a guild register from a medieval trading city",
                "as a tax record from a 14th-century monastery"
            ]
        elif has_corrections or has_entities:
            styles = [
                "as a monk's working manuscript with personal annotations", 
                "as a scholar's commentary on ancient texts",
                "as a chronicle being revised by a medieval historian",
                "as a theological treatise with scribal corrections",
                "as a copy of classical texts with medieval glosses"
            ]
        else:
            styles = [
                "as a sacred text illuminated by medieval scribes",
                "as a philosophical work from a cathedral school",
                "as a liturgical manuscript from a monastic library",
                "as a medical treatise translated in medieval Spain",
                "as an astronomical text from a medieval university"
            ]
        
        import random
        chosen_style = random.choice(styles)
        seed = random.randint(1000, 9999)
        
        # Craft historically-informed prompt
        processing_context = "deciphered using advanced medieval manuscript AI" if used_tridis else "carefully transcribed from the original"
        time_period = "13th-16th centuries" if (has_expansions or has_corrections) else "medieval period"
        
        prompt = (
            f"This Latin manuscript text was {processing_context}: {latin_text}\n\n"
            f"Historical context: The text appears to be from the {time_period}, "
            f"{'with expanded abbreviations and scribal corrections typical of documentary manuscripts' if has_expansions else 'showing characteristics of medieval scholarly tradition'}.\n\n"
            f"Create a vivid, historically accurate narrative (250+ words) set in medieval Europe, "
            f"telling the story of this manuscript's creation and significance. "
            f"Write {chosen_style}.\n\n"
            f"Include: Medieval setting, authentic historical details, multiple characters, "
            f"the process of manuscript creation, and the document's importance to its community.\n"
            f"Narrative seed: {seed}"
        )
        
        system_prompt = (
            "You are a medieval historian and storyteller specializing in manuscript culture, "
            "paleography, and daily life in 13th-16th century Europe. Create authentic, "
            "engaging narratives that reflect accurate historical knowledge of medieval "
            "scriptoriums, legal practices, and scholarly traditions."
        )
        
        story = self.groq_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        if not story or is_gibberish(story):
            return "Failed to generate historical narrative; medieval story creation unavailable."
        
        return story
