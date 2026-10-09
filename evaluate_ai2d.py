import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from pipeline import LayoutPreservingTranslator

def evaluate_on_ai2d(num_samples=3, target_lang="hi"):
    print("Loading AI2D dataset from Hugging Face...")
    try:
        # Load the popular version of AI2D from Hugging Face
        dataset = load_dataset("lmms-lab/ai2d", split="test", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    translator = LayoutPreservingTranslator(target_lang=target_lang)
    
    os.makedirs("ai2d_results", exist_ok=True)

    print(f"\nProcessing {num_samples} samples from AI2D...")
    count = 0
    for item in dataset:
        if count >= num_samples:
            break
            
        # Extract the PIL image from the dataset
        image = item["image"]
        image_name = f"ai2d_sample_{count}.jpg"
        output_path = os.path.join("ai2d_results", image_name)
        
        print(f"\n--- Processing Sample {count + 1} ---")
        
        # 1. Extract layout
        text_elements = translator.extract_text_and_layout(image)
        
        # 2. Translate text
        translated_elements = translator.translate_texts(text_elements)
        
        # 3. Inpaint
        inpainted_image = translator.inpaint_image(image, text_elements)
        
        # 4. Render
        final_image = translator.render_translated_text(inpainted_image, translated_elements)
        
        # Save output
        final_image.save(output_path)
        print(f"Saved translated sample to {output_path}")
        
        count += 1
        
    print(f"\nCompleted! Check the 'ai2d_results' folder for the translated diagrams.")

if __name__ == "__main__":
    # Feel free to increase num_samples when running on Kaggle!
    evaluate_on_ai2d(num_samples=5)
