from abc import ABC, abstractmethod
from PIL import Image

class BaseScriptProcessor(ABC):
    def __init__(self, groq_client, references, clip_classifier=None):  # Add clip_classifier parameter
        self.groq_client = groq_client
        self.references = references
        self.clip_classifier = clip_classifier  # Store clip_classifier

    
    @abstractmethod
    def detect_script(self, image_path):
        """Detect if image contains this script type"""
        pass
    
    @abstractmethod
    def extract_text(self, image_path):
        """Extract text/symbols from image"""
        pass
    
    @abstractmethod
    def process_text(self, extracted_text):
        """Process extracted text into meaningful output"""
        pass
    
    @abstractmethod
    def generate_historical_context(self, processed_text):
        """Generate historical context for the text"""
        pass
    
    @abstractmethod
    def generate_story(self, processed_text):
        """Generate creative story based on the text"""
        pass
    
    def process_image(self, image_path):
        """Main processing pipeline"""
        try:
            # Step 1: Detect script
            is_detected, confidence = self.detect_script(image_path)
            if not is_detected:
                return None
            
            # Step 2: Extract text
            extracted_text = self.extract_text(image_path)
            if not extracted_text:
                return None
            
            # Step 3: Process text
            processed_result = self.process_text(extracted_text)
            
            # Step 4: Generate context and story
            historical_context = self.generate_historical_context(processed_result)
            creative_story = self.generate_story(processed_result)
            
            return {
                "script_type": self.__class__.__name__.replace("Processor", "").lower(),
                "confidence": confidence,
                "extracted_text": extracted_text,
                "processed_result": processed_result,
                "historical_context": historical_context,
                "creative_story": creative_story
            }
        
        except Exception as e:
            print(f"[ERROR] Processing failed in {self.__class__.__name__}: {e}")
            return None
