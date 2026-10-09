import os
import sys
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

class LayoutPreservingTranslator:
    def __init__(self, target_lang="hi", device=None):
        """
        Initializes the OCR and Translation models.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_lang = target_lang
        
        print(f"Loading models on {self.device}...")
        
        # 1. Load OCR Model (bodhan-ai/indic-ocr) using custom pipeline
        self.ocr_model_id = "bodhan-ai/indic-ocr"
        try:
            print("Downloading/Loading indic-ocr repository...")
            repo = snapshot_download(self.ocr_model_id)
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from indic_ocr import IndicOCR
            self.ocr_parser = IndicOCR.from_pretrained(repo)
            print("OCR Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load OCR model from huggingface: {e}")
            self.ocr_parser = None
            
        # 2. Load Translation Model (bodhan-ai/indic-translate)
        self.translate_model_id = "bodhan-ai/indic-translate"
        try:
            self.translator_tokenizer = AutoTokenizer.from_pretrained(self.translate_model_id, trust_remote_code=True)
            self.translator_model = AutoModelForCausalLM.from_pretrained(
                self.translate_model_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(self.device)
            print("Translation Model loaded.")
        except Exception as e:
            print(f"Failed to load Translation model: {e}")
            self.translator_model = None

    def extract_text_and_layout(self, image: Image.Image):
        """
        Extracts text and bounding boxes from the image using indic-ocr.
        Returns a list of dicts: [{'text': str, 'bbox': [xmin, ymin, xmax, ymax]}]
        """
        print("Extracting layout and text...")
        if self.ocr_parser is None:
            print("Warning: Using mock OCR data because indic-ocr failed to load.")
            return [
                {"text": "Photosynthesis", "bbox": [100, 50, 300, 80]},
                {"text": "Sunlight", "bbox": [50, 150, 150, 180]}
            ]

        # Save PIL image to a temporary file since parse() takes a file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            
        # Save image then pass to parse
        image.save(tmp_path)
            
        try:
            page_data = self.ocr_parser.parse(tmp_path)
        finally:
            os.remove(tmp_path)

        results = []
        for block in page_data.get("blocks", []):
            text = block.get("text", "")
            # Skip empty texts (figures, charts, etc. that were not transcribed)
            if not text.strip():
                continue
                
            bbox = block.get("bbox_xyxy", [0, 0, 0, 0])
            results.append({
                "text": text,
                "bbox": [int(b) for b in bbox]
            })
            
        return results

    def translate_texts(self, text_elements):
        """
        Translates the extracted texts into the target language.
        """
        print(f"Translating {len(text_elements)} texts to {self.target_lang}...")
        translated_elements = []
        
        if self.translator_model is None:
            # Mock translation
            for el in text_elements:
                translated_elements.append({
                    "original_text": el["text"],
                    "translated_text": f"अनुवाद: {el['text']}",
                    "bbox": el["bbox"]
                })
            return translated_elements

        # Translating using Gemma 4 architecture
        for element in text_elements:
            original_text = element["text"]
            
            # Simple prompt assuming standard instruction format
            prompt = f"Translate from English to {self.target_lang}:\n{original_text}\nTranslation:"
            inputs = self.translator_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.translator_model.generate(**inputs, max_new_tokens=256)
                
            # Decode only the newly generated tokens
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            translated_text = self.translator_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            translated_elements.append({
                "original_text": original_text,
                "translated_text": translated_text,
                "bbox": element["bbox"]
            })
            
        return translated_elements

    def inpaint_image(self, image: Image.Image, text_elements) -> Image.Image:
        """
        Erases the original text from the image using the bounding boxes as a mask.
        """
        print("Inpainting original image to remove English text...")
        img_cv = np.array(image.convert("RGB"))
        img_cv = img_cv[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV
        
        mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        
        # Create mask based on bounding boxes
        for element in text_elements:
            xmin, ymin, xmax, ymax = element["bbox"]
            # Add some padding to the bbox to ensure complete removal
            pad = 5
            cv2.rectangle(mask, (max(0, xmin-pad), max(0, ymin-pad)), 
                          (min(img_cv.shape[1], xmax+pad), min(img_cv.shape[0], ymax+pad)), 
                          255, -1)
            
        # Use Navier-Stokes based inpainting
        inpainted_img = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_NS)
        
        # Convert back to PIL Image
        inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(inpainted_img)

    def render_translated_text(self, inpainted_image: Image.Image, translated_elements) -> Image.Image:
        """
        Draws the translated text back onto the inpainted image within original bounding boxes.
        """
        print("Rendering translated text onto the image...")
        result_image = inpainted_image.copy()
        draw = ImageDraw.Draw(result_image)
        
        try:
            font = ImageFont.truetype("NotoSansDevanagari.ttf", 20) 
        except IOError:
            font = ImageFont.load_default()

        for element in translated_elements:
            translated_text = element["translated_text"]
            xmin, ymin, xmax, ymax = element["bbox"]
            
            draw.text((xmin, ymin), translated_text, fill="black", font=font)
            
        return result_image

    def process(self, image_path: str, output_path: str):
        image = Image.open(image_path)
        text_elements = self.extract_text_and_layout(image)
        translated_elements = self.translate_texts(text_elements)
        inpainted_image = self.inpaint_image(image, text_elements)
        final_image = self.render_translated_text(inpainted_image, translated_elements)
        final_image.save(output_path)
        print(f"Saved translated image to {output_path}")

if __name__ == "__main__":
    translator = LayoutPreservingTranslator(target_lang="hi")
