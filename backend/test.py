import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import modular components
from config import Config
from models.groq_client import GroqClient
from models.clip_classifier import CLIPClassifier
from models.tesseract_ocr import TesseractOCR
from models.huggingface_models import HuggingFaceModels
from services.groq_vision_classifier import GroqVisionScriptClassifier
from services.script_detector import ScriptDetectionService
from utils.image_utils import validate_image
from utils.text_utils import clean_text

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global components
config = Config()
groq_client = None
clip_classifier = None
hf_models = None
script_detector = None
references = {}

def load_references():
    """Load references from JSON file"""
    global references
    try:
        import json
        with open(config.REFERENCES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        references = {
    "egypt_symbol_notes": data.get("egypt_symbol_notes", {}),
    "greek_symbol_notes": data.get("greek_symbol_notes", {}),  # ADD THIS LINE
    "greek_hint": data.get("greek_hint", "If no specific character note is found, treat as lexical marker considering diacriticals (breathing marks, accents, vowel quantity) which affect pronunciation, meaning, and grammatical function in ancient Greek texts."),
    "latin_symbol_notes": data.get("latin_symbol_notes", {}),
    "latin_hint": data.get("latin_hint", "If no specific character note is found, consider standard Latin letters or medieval scribal abbreviations.")
}

        print(f"[INFO] Loaded references from {config.REFERENCES_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load references: {e}")
        references = {
            "egypt_symbol_notes": {},
            "greek_term_hint": "Possible lexical marker.",
            "greek_term_hints": {},
            "latin_symbol_notes": {},
            "latin_term_hint": "Latin scribal practice."
        }

def initialize_models():
    """Initialize all AI models and services"""
    global groq_client, clip_classifier, hf_models, script_detector
    
    print("[INFO] Initializing models...")
    
    # Initialize clients
    groq_client = GroqClient()
    clip_classifier = CLIPClassifier()
    hf_models = HuggingFaceModels()
    
    # Load references
    load_references()
    
    # Initialize script detection service
    script_detector = ScriptDetectionService(
        groq_client=groq_client,
        references=references,
        clip_classifier=clip_classifier,
        translator_pipe=hf_models.get_translator()
    )
    
    print("[INFO] Models initialized successfully")

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint with Groq Vision classification"""
    tmp_path = None
    
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        img_file = request.files['image']
        if img_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Validate image file
        try:
            validate_image(img_file)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp_path = tmp.name
            img_file.save(tmp_path)
        
        # Process image with Groq Vision classification
        result = script_detector.detect_and_process(tmp_path)
        
        if not result:
            return jsonify({"error": "Could not process image"}), 500
        
        # Get Vision classification info
        vision_classification = result.get('vision_classification', 'unknown')
        classification_method = result.get('classification_method', 'unknown')
        classification_confidence = result.get('classification_confidence', 0.0)
        script_type = result.get('script_type', 'egyptian')
        
        # Base response with Vision classification info
        base_response = {
            "script_type": script_type,
            "vision_classification": vision_classification,
            "classification_method": classification_method,
            "classification_confidence": classification_confidence,
            "confidence": result.get('confidence', 0.0),
            "historical_context": result.get('historical_context', {}),
            "creative_story": result.get('creative_story', ''),
            "model_used": "meta-llama/llama-4-scout-17b-16e-instruct"
        }
        
        if script_type in ['greek', 'latin']:
            processed_result = result.get('processed_result', {})
            validation = processed_result.get('validation', {})
            
            response_data = {
                **base_response,
                "labels": [],
                "gardiner_codes": [],
                "translation": processed_result.get('text', ''),
                "translation_ok": True,
            }
            
            # Add enhanced validation info for Greek
            if script_type == 'greek':
                response_data["validation"] = {
                    "quality_score": validation.get('quality_score', 0.0),
                    "greek_ratio": validation.get('greek_ratio', 0.0),
                    "has_polytonic": validation.get('has_polytonic', False),
                    "char_analysis": processed_result.get('char_analysis', {}),
                    "ocr_method": "ancient_greek_ocr" if validation.get('quality_score', 0) > 0.7 else "standard_greek_ocr"
                }
            
            # Add enhanced validation info for Latin
            elif script_type == 'latin':
                response_data["validation"] = {
                    "quality_score": validation.get('quality_score', 0.0),
                    "latin_ratio": validation.get('latin_ratio', 0.0),
                    "trocr_used": validation.get('trocr_used', False),
                    "char_analysis": processed_result.get('char_analysis', {}),
                    "ocr_method": "medieval_trocr" if validation.get('trocr_used') else "standard_latin_ocr"
                }
            
            return jsonify(response_data)
        
        else:  # Egyptian
            processed = result['processed_result']
            return jsonify({
                **base_response,
                "labels": processed['labels'],
                "gardiner_codes": processed['codes'],
                "translation": processed['translation'],
                "translation_ok": processed['translation_ok']
            })
    
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return jsonify({"error": "Processing failed"}), 500
    
    finally:
        # Cleanup temporary file
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass




@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "groq": groq_client.is_available() if groq_client else False,
            "clip": clip_classifier.pipeline is not None if clip_classifier else False,
            "translator": hf_models.get_translator() is not None if hf_models else False
        }
    })

@app.route('/info', methods=['GET'])
def info():
    """Information endpoint"""
    return jsonify({
        "app": "Ancient Script Recognition System",
        "version": "2.0.0",
        "supported_scripts": ["Egyptian Hieroglyphs", "Ancient Greek", "Latin"],
        "features": [
            "Multi-script detection",
            "OCR text extraction", 
            "Historical context generation",
            "Creative story generation"
        ]
    })

if __name__ == "__main__":
    print("[INIT] Starting Ancient Script Recognition System...")
    
    # Initialize all models
    initialize_models()
    
    # Start Flask app
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"[INFO] Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
