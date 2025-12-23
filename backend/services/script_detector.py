from processors.egyptian_processor import EgyptianProcessor
from processors.greek_processor import GreekProcessor
from processors.latin_processor import LatinProcessor
from processors.cuneiform_processor import CuneiformProcessor
from .groq_vision_classifier import GroqVisionScriptClassifier


class ScriptDetectionService:
    def __init__(self, groq_client, references, clip_classifier, translator_pipe):
        # Initialize processors including cuneiform
        self.egyptian_processor = EgyptianProcessor(groq_client, references, clip_classifier, translator_pipe)
        self.greek_processor = GreekProcessor(groq_client, references, clip_classifier)
        self.latin_processor = LatinProcessor(groq_client, references, clip_classifier)
        
        # Initialize cuneiform processor
        try:
            print("[INFO] Initializing cuneiform processor in detection service...")
            self.cuneiform_processor = CuneiformProcessor(groq_client, references, clip_classifier)
            print("[INFO] Cuneiform processor initialized successfully")
        except Exception as e:
            print(f"[WARN] Failed to initialize cuneiform processor: {e}")
            self.cuneiform_processor = None
        
        # FIXED: Get API key from groq_client with multiple fallback options
        api_key = None
        if hasattr(groq_client, 'api_key'):
            api_key = groq_client.api_key
        elif hasattr(groq_client, 'client') and hasattr(groq_client.client, 'api_key'):
            api_key = groq_client.client.api_key
        else:
            # Fallback: get from config or environment
            try:
                from config import Config
                config = Config()
                api_key = config.GROQ_API_KEY
            except:
                import os
                api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found! Check your configuration.")
        
        # Initialize Groq Vision script classifier
        self.vision_classifier = GroqVisionScriptClassifier(api_key)
        
        # Enhanced processor mapping with cuneiform
        self.processors = {
            'egyptian': self.egyptian_processor,
            'greek': self.greek_processor,
            'latin': self.latin_processor,
            'cuneiform': self.cuneiform_processor
        }
        
        print("[INFO] Groq Vision Script Detection Service initialized")
        if self.cuneiform_processor:
            print("[INFO] Cuneiform support: ENABLED (praeclarum/cuneiform model)")
        else:
            print("[WARN] Cuneiform support: DISABLED (processor initialization failed)")
    
    def detect_and_process(self, image_path):
        """Enhanced detection with cuneiform support - uses Groq Vision"""
        try:
            # Step 1: Get script classification from Groq Vision
            script_type = self.vision_classifier.classify_script(image_path)
            
            print(f"[INFO] Groq Vision final classification: {script_type}")
            
            # Step 2: Route to appropriate processor including cuneiform
            if script_type == "egyptian":
                print("[INFO] Routing to Egyptian processor...")
                result = self.egyptian_processor.process_image(image_path)
                
            elif script_type == "greek":
                print("[INFO] Routing to Greek processor...")
                result = self.greek_processor.process_image(image_path)
                
            elif script_type == "latin":
                print("[INFO] Routing to Latin processor...")
                result = self.latin_processor.process_image(image_path)
                
            elif script_type == "cuneiform":
                print("[INFO] Routing to Cuneiform processor...")
                if self.cuneiform_processor and self.cuneiform_processor.cuneiform_available:
                    result = self.cuneiform_processor.process_image(image_path)
                else:
                    print("[ERROR] Cuneiform processor not available!")
                    # Create error result
                    result = {
                        'script_type': 'cuneiform',
                        'confidence': 0.0,
                        'processed_result': {
                            'text': 'Cuneiform processor unavailable',
                            'validation': {'quality_score': 0.0, 'error': 'Model not loaded'}
                        },
                        'historical_context': {},
                        'creative_story': 'Cuneiform processing failed - model not available'
                    }
                
            else:  # unknown
                print(f"[INFO] Unknown classification '{script_type}', defaulting to Egyptian...")
                result = self.egyptian_processor.process_image(image_path)
            
            # Step 3: Return result with classification metadata
            if result:
                result['vision_classification'] = script_type
                result['classification_method'] = 'groq_vision'
                result['classification_confidence'] = 0.95
                print(f"[INFO] {script_type.title()} processing completed successfully")
                return result
            else:
                print(f"[ERROR] {script_type.title()} processor returned None")
                return None
            
        except Exception as e:
            print(f"[ERROR] Groq Vision processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_processor_by_type(self, script_type):
        """Get processor by script type - now includes cuneiform"""
        processor = self.processors.get(script_type.lower())
        
        if script_type.lower() == 'cuneiform' and processor and not processor.cuneiform_available:
            print(f"[WARN] Cuneiform processor exists but model not available")
            return None
            
        return processor
    
    def get_supported_scripts(self):
        """Get list of supported script types"""
        scripts = ['egyptian', 'greek', 'latin']
        
        if self.cuneiform_processor and self.cuneiform_processor.cuneiform_available:
            scripts.append('cuneiform')
            
        return scripts
    
    def get_processor_status(self):
        """Get status of all processors"""
        status = {
            'egyptian': self.egyptian_processor is not None,
            'greek': self.greek_processor is not None,
            'latin': self.latin_processor is not None,
            'cuneiform': self.cuneiform_processor is not None and getattr(self.cuneiform_processor, 'cuneiform_available', False)
        }
        
        return status

    def validate_script_detection(self, script_type, processed_result):
        """Validate script detection results - enhanced for cuneiform"""
        try:
            validation = processed_result.get('validation', {})
            quality_score = validation.get('quality_score', 0.0)
            
            # Script-specific validation thresholds
            thresholds = {
                'egyptian': 0.3,
                'greek': 0.4, 
                'latin': 0.4,
                'cuneiform': 0.2  # Lower threshold due to OCR challenges
            }
            
            threshold = thresholds.get(script_type, 0.3)
            
            # Additional cuneiform validation
            if script_type == 'cuneiform':
                cuneiform_ratio = validation.get('cuneiform_ratio', 0.0)
                atf_ratio = validation.get('atf_ratio', 0.0)
                
                # Accept if either Unicode cuneiform or ATF format detected
                if cuneiform_ratio > 0.1 or atf_ratio > 0.3:
                    print(f"[INFO] Cuneiform validation passed: cuneiform_ratio={cuneiform_ratio:.3f}, atf_ratio={atf_ratio:.3f}")
                    return True
                    
            # Standard quality validation
            is_valid = quality_score >= threshold
            
            if is_valid:
                print(f"[INFO] {script_type.title()} validation passed: quality={quality_score:.3f} >= {threshold}")
            else:
                print(f"[WARN] {script_type.title()} validation failed: quality={quality_score:.3f} < {threshold}")
                
            return is_valid
            
        except Exception as e:
            print(f"[ERROR] Validation failed: {e}")
            return False
