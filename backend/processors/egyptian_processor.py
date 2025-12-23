import cv2
import numpy as np
from PIL import Image
from itertools import groupby
from collections import Counter
from .base_processor import BaseScriptProcessor
from utils.image_utils import segment_hieroglyphs
from utils.text_utils import is_gibberish, build_description_from_codes
from config import Config

class EgyptianProcessor(BaseScriptProcessor):
    def __init__(self, groq_client, references, clip_classifier, translator_pipe):
        super().__init__(groq_client, references)
        self.clip_classifier = clip_classifier
        self.translator_pipe = translator_pipe
        self.config = Config()
    
    def detect_script(self, image_path):
        """Simplified detection - Groq Vision handles main classification"""
        try:
            # If called by Groq Vision classification, accept
            print("[INFO] Egyptian processor activated by Groq Vision (Llama-4-Scout)")
            return True, 0.95
            
        except Exception as e:
            print(f"[ERROR] Egyptian detection failed: {e}")
            return False, 0.0

    
    def extract_text(self, image_path):
        """Extract hieroglyphs using image segmentation"""
        try:
            print("[INFO] Starting Egyptian hieroglyph extraction...")
            
            # Import required modules
            from utils.image_utils import segment_hieroglyphs
            
            # Segment hieroglyphs
            crops = segment_hieroglyphs(image_path)
            print(f"[INFO] Segmented {len(crops)} hieroglyph regions")
            
            if not crops:
                print("[WARN] No hieroglyph regions found")
                return []
            
            # Classify symbols using CLIP
            candidate_labels = list(self.config.GARDINER_MAP.keys())
            labels = self.clip_classifier.classify_symbols(crops, candidate_labels)
            
            print(f"[INFO] Classified {len(labels)} symbols: {labels}")
            return labels
            
        except Exception as e:
            print(f"[ERROR] Egyptian text extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    def process_text(self, labels):
        """Process hieroglyph labels into translation"""
        if not labels:
            return {"labels": [], "codes": [], "translation": "", "translation_ok": False}
        
        # Convert labels to Gardiner codes
        codes = [self.config.GARDINER_MAP.get((lbl or "").lower(), "?") for lbl in labels]
        
        # Attempt translation
        translation, translation_ok = self._translate_sequence(labels, codes)
        
        return {
            "labels": labels,
            "codes": codes, 
            "translation": translation,
            "translation_ok": translation_ok
        }
    
    def _translate_sequence(self, labels, codes):
        """Translate Gardiner sequence using HuggingFace model"""
        seq = " ".join(codes)
        prompt = f"Translate hieroglyph unicode sequence to English: {seq}"
        
        if self.translator_pipe:
            try:
                output = self.translator_pipe(prompt, max_new_tokens=128, do_sample=False, num_beams=4)
                text = output[0].get('generated_text') or output[0].get('translation_text') or str(output[0])
                
                if not is_gibberish(text):
                    return text, True
                
                # Try alternative approach
                alt_output = self.translator_pipe(seq, max_new_tokens=128, do_sample=False, num_beams=4)
                alt_text = alt_output[0].get('generated_text') or alt_output[0].get('translation_text') or str(alt_output[0])
                
                if not is_gibberish(alt_text):
                    return alt_text, True
                
            except Exception as e:
                print(f"[WARN] Translation failed: {e}")
        
        # Fallback to description
        description = build_description_from_codes(codes)
        return f"(Symbols described as: {description})", False
    
    def generate_historical_context(self, processed_result):
        """Generate historical context for Egyptian text"""
        translation = processed_result.get("translation", "")
        codes = processed_result.get("codes", [])
        labels = processed_result.get("labels", [])
        
        # Generate Groq context
        groq_detail = self._generate_groq_context(translation, codes)
        
        # Build structured context
        return {
            "uses_box": {
                "title": "Each symbol's possible use by the egyptian people",
                "items": self._build_uses_list(labels)
            },
            "meaning_box": self._build_meaning_box(labels, groq_detail)
        }
    
    def _generate_groq_context(self, translation_text, codes):
        """Generate contextual information using Groq"""
        if not self.groq_client.is_available():
            return "(Groq unavailable) Context generation requires GROQ_API_KEY and groq package."
        
        if is_gibberish(translation_text):
            prompt_body = build_description_from_codes(codes)
            prompt = (
                f"The following sequence of ancient Egyptian symbols is described as: {prompt_body}.\n\n"
                "Provide a concise, scholarly paragraph (6-10 sentences) covering cultural context, symbolic meanings, "
                "typical usage, probable time period, and relevant archaeological comparisons. Avoid repeating the prompt."
            )
        else:
            prompt = (
                f"Provide a concise, scholarly paragraph (6-10 sentences) on the historical significance, cultural context, "
                f"symbolism, and possible interpretations of this ancient Egyptian text: {translation_text}. Avoid repeating the prompt."
            )
        
        system_prompt = "You are a careful Egyptologist and historian. Provide accurate, concise scholarly context."
        
        return self.groq_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=self.config.GROQ_CONTEXT_MAX_TOKENS
        ) or "(context unavailable due to Groq error)"
    
    def _build_uses_list(self, labels):
        """Build list of symbol uses"""
        groups = []
        for key, g in groupby(labels):
            if not key:
                continue
            groups.append((key, len(list(g))))
        
        notes = self.references.get("egypt_symbol_notes", {}) or {}
        seen = set()
        items = []
        
        for name, count in groups:
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            
            count_str = f" (x{count})" if count > 1 else ""
            note = notes.get(name.lower(), "Common sign whose meaning varies by phonetic/ideogram/determinative roles.")
            items.append(f"- {name}{count_str}: {note}")
        
        if not items:
            items.append("- unknown: No stable mapping; likely decorative or damaged glyphs.")
        
        return items
    
    def _build_meaning_box(self, labels, groq_detail):
        """Build meaning interpretation box"""
        freq = Counter([l for l in labels if l])
        frequent = [f"{name} (x{cnt})" for name, cnt in freq.most_common(6)]
        
        intro_lines = [
            "The dense recurrence of signs suggests a formulaic or protective sequence, where phonograms articulate a core utterance and determinatives or iconic signs reinforce ritual intent.",
            "Comparable sequences appear on funerary equipment from the Middle Kingdom onward."
        ]
        
        points = [
            "• Offering and action signs (bread, jar, hoe, bow) commonly structure invocations or provisioning lists for the afterlife.",
            "• Repetition often encodes names or epithets; determinatives (eye, feather, god_figure) frame a protective or ritual context.",
            "• Repertoire and layout align with New Kingdom funerary practice focused on protection, sustenance, and legitimation."
        ]
        
        if groq_detail and isinstance(groq_detail, str) and groq_detail.strip():
            points.append(groq_detail.strip())
        
        return {
            "title": "Possible meaning:",
            "intro_lines": intro_lines,
            "frequent_label": "Frequently observed signs",
            "frequent": frequent,
            "points": points
        }
    
    def generate_story(self, processed_result):
        """Generate creative story for Egyptian text"""
        labels = processed_result.get("labels", [])
        description = ", ".join([lbl for lbl in labels if lbl])
        
        if not self.groq_client.is_available():
            return self._simple_templated_story(description)
        
        style = [
            "as an epic poem from a wandering bard",
            "as a prophecy carved in stone", 
            "as a fireside tale with vivid emotions",
            "as a dialogue between two ancient gods",
            "as a lost papyrus narrative recovered from the sands",
            "as a myth told by a court poet"
        ]
        
        import random
        chosen_style = random.choice(style)
        seed = random.randint(1000, 9999)
        
        prompt = (
            f"The following sequence of ancient Egyptian symbols is described as: {description}\n\n"
            f"Can you create a long, vivid, imaginative story from ancient times "
            f"based on this sequence of Egyptian symbols: [your sequence]. "
            f"Write it as one rich paragraph with a lot of detail, mystery, and historical atmosphere. "
            f"At least 200 words.\n\n"
            f"Creative seed: {seed}\n"
            f"Write a richly detailed, imaginative myth-like story {chosen_style}. "
            "Include multiple characters, vivid imagery, and at least 3 short scenes. "
            "Do NOT repeat the same sentence or phrase verbatim. "
            "Keep it evocative and unpredictable."
        )
        
        system_prompt = "You are a creative ancient historian and myth-maker. Invent rich, imaginative tales."
        
        story = self.groq_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=self.config.GROQ_STORY_MAX_TOKENS
        )
        
        if not story or is_gibberish(story):
            return self._simple_templated_story(description)
        
        return story
    
    def _simple_templated_story(self, description):
        """Fallback story generation"""
        import re
        parts = [p.strip() for p in re.split(r',\s*', description) if p.strip()]
        keywords = []
        
        for p in parts:
            m = re.match(r'([a-zA-Z0-9_-]+)', p)
            if m:
                kw = m.group(1)
                if kw not in keywords:
                    keywords.append(kw)
            if len(keywords) >= 8:
                break
        
        flavor = {
            "bow": "strength and vigilance",
            "hoe": "the work of the fields", 
            "reed": "the scribe's craft",
            "owl": "hidden wisdom of the night",
            "eye": "divine sight",
            "bread": "offerings to the ka",
            "unknown": "mysterious signs"
        }
        
        lead = []
        if keywords:
            lead.append(f"In an age of river and stone, a tale was told of {flavor.get(keywords[0], keywords[0])}.")
        if len(keywords) > 1:
            second = flavor.get(keywords[1], keywords[1])
            third = flavor.get(keywords[2], keywords[2]) if len(keywords) > 2 else "omens"
            lead.append(f"It spoke of {second} and {third} guiding a soul beyond the horizon.")
        lead.append("Under the stars, elders whispered a vow that the names would endure.")
        
        return " ".join(lead)
