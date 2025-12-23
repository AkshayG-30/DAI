import base64
import json
import os
from io import BytesIO
from PIL import Image
from groq import Groq


class GroqVisionScriptClassifier:
    def __init__(self, groq_api_key):
        self.groq_client = Groq(api_key=groq_api_key)
        # FIXED: Use the correct stable model name
        self.vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        print(f"[INFO] Groq Vision Classifier initialized with {self.vision_model}")
    
    def classify_script(self, image_path):
        """Enhanced script classification including cuneiform using Groq's Llama Vision model"""
        try:
            # Convert image to base64
            base64_image = self._image_to_base64(image_path)
            if not base64_image:
                return "unknown"
            
            # Query Groq Vision API
            response = self._query_groq_vision(base64_image)
            
            # Parse the response
            script_type = self._parse_classification_response(response)
            
            print(f"[INFO] Llama Vision classified script as: {script_type}")
            return script_type.lower()
            
        except Exception as e:
            print(f"[ERROR] Groq Vision script classification failed: {e}")
            return "unknown"
    
    def _image_to_base64(self, image_path):
        """Convert image to base64 for Groq Vision API (4MB limit)"""
        try:
            image = Image.open(image_path)
            
            # Resize if too large (keep under 4MB base64 limit)
            if max(image.size) > 1200:
                image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            
            # Convert to base64 JPEG (smaller than PNG)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Check size (base64 should be < 4MB)
            if len(image_b64) > 4 * 1024 * 1024:  # 4MB limit
                # Reduce quality and try again
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=70)
                image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_b64
            
        except Exception as e:
            print(f"[ERROR] Image to base64 conversion failed: {e}")
            return None
    
    def _query_groq_vision(self, base64_image):
        """Enhanced query for Groq Llama Vision API including cuneiform"""
        try:
            # FIXED: Simplified prompt to avoid token limit issues
            prompt = """Analyze this image of ancient text/script as an expert paleographer.

Classify it as ONE of these ancient script types:

- EGYPTIAN: Hieroglyphic symbols (birds, eyes, human figures, cartouches)
- GREEK: Ancient/medieval Greek alphabet (α,β,γ,δ,ε,ζ,η,θ) with diacritics
- LATIN: Latin alphabet letters, Roman inscriptions, medieval manuscripts
- CUNEIFORM: Wedge-shaped impressions on clay tablets (triangular marks)

IMPORTANT: Cuneiform has geometric wedge patterns, NOT pictures like hieroglyphs.

Respond ONLY with JSON:
{"classification": "EGYPTIAN" or "GREEK" or "LATIN" or "CUNEIFORM", "confidence": 0.0-1.0}"""

            completion = self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_completion_tokens=100,  # FIXED: Reduced to avoid token errors
                top_p=0.9,
                stream=False,
                response_format={"type": "json_object"}
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"[ERROR] Groq Vision API call failed: {e}")
            return None
    
    def _parse_classification_response(self, response):
        """Enhanced parsing for JSON response including cuneiform"""
        if not response:
            return "unknown"
        
        try:
            # Parse JSON response
            data = json.loads(response)
            classification = data.get('classification', '').upper()
            confidence = data.get('confidence', 0.0)
            
            print(f"[INFO] Vision model confidence: {confidence:.3f}")
            
            # Enhanced classification mapping including cuneiform
            if classification == "EGYPTIAN":
                return "egyptian"
            elif classification == "GREEK":
                return "greek"
            elif classification == "LATIN":
                return "latin"
            elif classification == "CUNEIFORM":
                return "cuneiform"
            else:
                print(f"[WARN] Unknown classification: {classification}")
                return "unknown"
                
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse JSON response, trying text parsing: {response}")
            # Enhanced fallback to text parsing
            response_upper = response.strip().upper()
            
            # Priority order: cuneiform keywords first (most specific)
            cuneiform_keywords = ["CUNEIFORM", "WEDGE", "CLAY", "MESOPOTAMIAN", "AKKADIAN", "SUMERIAN", "BABYLONIAN"]
            if any(keyword in response_upper for keyword in cuneiform_keywords):
                return "cuneiform"
            elif "EGYPTIAN" in response_upper or "HIEROGLYPH" in response_upper:
                return "egyptian"
            elif "GREEK" in response_upper:
                return "greek"
            elif "LATIN" in response_upper or "ROMAN" in response_upper:
                return "latin"
        
        except Exception as e:
            print(f"[ERROR] Response parsing failed: {e}")
        
        return "unknown"
    
    def classify_with_fallback(self, image_path, max_retries=2):
        """Enhanced classification with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                result = self.classify_script(image_path)
                
                if result != "unknown":
                    return result
                elif attempt < max_retries:
                    print(f"[INFO] Classification attempt {attempt + 1} returned unknown, retrying...")
                    continue
                else:
                    print(f"[WARN] All classification attempts returned unknown")
                    return "unknown"
                    
            except Exception as e:
                if attempt < max_retries:
                    print(f"[WARN] Classification attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                else:
                    print(f"[ERROR] All classification attempts failed: {e}")
                    return "unknown"
        
        return "unknown"
    
    def get_supported_scripts(self):
        """Get list of supported script types"""
        return ["egyptian", "greek", "latin", "cuneiform"]
    
    def validate_classification(self, script_type, confidence_threshold=0.7):
        """Validate classification result"""
        supported_scripts = self.get_supported_scripts()
        
        if script_type not in supported_scripts:
            print(f"[WARN] Unsupported script type: {script_type}")
            return False
        
        # All classifications from Llama Vision are considered valid
        return True
    
    def get_model_info(self):
        """Get information about the vision model being used"""
        return {
            "model": self.vision_model,
            "provider": "Groq",
            "supported_scripts": self.get_supported_scripts(),
            "features": [
                "Ancient script classification",
                "Multi-script support", 
                "Cuneiform wedge detection",
                "Clay tablet recognition",
                "High-resolution image processing"
            ]
        }

    def debug_classification(self, image_path, save_debug_info=False):
        """Debug classification with detailed information"""
        try:
            print(f"[DEBUG] Starting classification for: {image_path}")
            
            # Check image properties
            image = Image.open(image_path)
            print(f"[DEBUG] Image size: {image.size}")
            print(f"[DEBUG] Image mode: {image.mode}")
            
            # Get base64 size
            base64_image = self._image_to_base64(image_path)
            if base64_image:
                print(f"[DEBUG] Base64 size: {len(base64_image)} characters")
            
            # Get raw response
            response = self._query_groq_vision(base64_image)
            print(f"[DEBUG] Raw API response: {response}")
            
            # Parse and return
            result = self._parse_classification_response(response)
            print(f"[DEBUG] Final classification: {result}")
            
            if save_debug_info:
                debug_info = {
                    "image_path": image_path,
                    "image_size": image.size,
                    "base64_length": len(base64_image) if base64_image else 0,
                    "raw_response": response,
                    "classification": result
                }
                
                debug_file = f"debug_classification_{result}_{hash(image_path) % 10000}.json"
                with open(debug_file, 'w') as f:
                    json.dump(debug_info, f, indent=2)
                print(f"[DEBUG] Debug info saved to: {debug_file}")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Debug classification failed: {e}")
            return "unknown"
