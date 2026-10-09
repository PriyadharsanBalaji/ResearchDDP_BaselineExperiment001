import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

class LayoutPreservingTranslator:
    def __init__(self, target_lang="hi", device=None):
        """
        Initializes the OCR and Translation models.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_lang = target_lang
        
        print(f"Loading models on {self.device}...")
        
        # 1. Load OCR Model (bodhan-ai/indic-ocr)
        self.ocr_model_id = "bodhan-ai/indic-ocr"
        try:
            # We must use trust_remote_code=True for brand new custom models
            self.ocr_processor = AutoProcessor.from_pretrained(self.ocr_model_id, trust_remote_code=True)
            self.ocr_model = AutoModelForCausalLM.from_pretrained(
                self.ocr_model_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(self.device)
            print("OCR Model loaded.")
        except Exception as e:
            print(f"Failed to load OCR model from huggingface: {e}")
            
        # 2. Load Translation Model (bodhan-ai/indic-translate)
        self.translate_model_id = "bodhan-ai/indic-translate"
        try:
            self.translator_tokenizer = AutoTokenizer.from_pretrained(self.translate_model_id, trust_remote_code=True)
            # It uses Gemma4 which is a CausalLM, not a Seq2SeqLM!
            self.translator_model = AutoModelForCausalLM.from_pretrained(
                self.translate_model_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(self.device)
            print("Translation Model loaded.")
        except Exception as e:
            print(f"Failed to load Translation model: {e}")

    def extract_text_and_layout(self, image: Image.Image):
        """
        Extracts text and bounding boxes from the image using indic-ocr.
        Returns a list of dicts: [{'text': str, 'bbox': [xmin, ymin, xmax, ymax]}]
        """
        print("Extracting layout and text...")
        # Placeholder for actual model inference based on Bodhan's model card
        # Example pseudo-code:
        # inputs = self.ocr_processor(images=image, return_tensors="pt").to(self.device)
        # outputs = self.ocr_model.generate(**inputs)
        # parsed_results = self.ocr_processor.post_process(outputs)
        
        # Mocking output for the baseline structure
        # (You will replace this block with the actual inference code once you check the model card)
        mock_results = [
            {"text": "Photosynthesis", "bbox": [100, 50, 300, 80]},
            {"text": "Sunlight", "bbox": [50, 150, 150, 180]}
        ]
        return mock_results

    def translate_texts(self, text_elements):
        """
        Translates the extracted texts into the target language.
        """
        print(f"Translating texts to {self.target_lang}...")
        translated_elements = []
        for element in text_elements:
            original_text = element["text"]
            
            # Example inference
            # inputs = self.translator_tokenizer(original_text, return_tensors="pt").to(self.device)
            # outputs = self.translator_model.generate(**inputs)
            # translated_text = self.translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Mocking output
            translated_text = f"अनुवाद: {original_text}" # Mock Hindi translation
            
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
        
        # NOTE: You will need a font file that supports Indic scripts (like NotoSansDevanagari)
        # Download from Google Fonts and upload to Kaggle.
        try:
            # Try to load a generic font, or fallback to default
            font = ImageFont.truetype("arial.ttf", 20) 
        except IOError:
            font = ImageFont.load_default()

        for element in translated_elements:
            translated_text = element["translated_text"]
            xmin, ymin, xmax, ymax = element["bbox"]
            
            # Basic rendering. For robust layout preservation, you would calculate font size 
            # based on bbox dimensions.
            draw.text((xmin, ymin), translated_text, fill="black", font=font)
            
        return result_image

    def process(self, image_path: str, output_path: str):
        """
        Runs the full end-to-end pipeline.
        """
        image = Image.open(image_path)
        
        # 1. Extract layout
        text_elements = self.extract_text_and_layout(image)
        
        # 2. Translate text
        translated_elements = self.translate_texts(text_elements)
        
        # 3. Inpaint
        inpainted_image = self.inpaint_image(image, text_elements)
        
        # 4. Render
        final_image = self.render_translated_text(inpainted_image, translated_elements)
        
        # Save and display
        final_image.save(output_path)
        print(f"Saved translated image to {output_path}")
        
        # Display in notebook
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(final_image)
        axes[1].set_title(f"Translated ({self.target_lang})")
        axes[1].axis("off")
        plt.show()

if __name__ == "__main__":
    # Example usage for testing
    translator = LayoutPreservingTranslator(target_lang="hi")
    # translator.process("test_diagram.jpg", "output_diagram.jpg")
