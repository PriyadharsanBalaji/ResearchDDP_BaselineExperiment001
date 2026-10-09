import os
from datasets import load_dataset
from pipeline import LayoutPreservingTranslator

def test_dataset(translator, dataset_path, dataset_name, split="train", num_samples=2, image_col="image"):
    print(f"\n{'='*50}")
    print(f"Testing on {dataset_name} ({dataset_path})")
    print(f"{'='*50}")
    
    try:
        dataset = load_dataset(dataset_path, split=split, streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    output_dir = f"{dataset_name}_results"
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for item in dataset:
        if count >= num_samples:
            break
            
        try:
            image = item[image_col]
            # Ensure image is in RGB mode for OpenCV processing later
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            image_name = f"{dataset_name}_sample_{count}.jpg"
            output_path = os.path.join(output_dir, image_name)
            
            print(f"\n--- {dataset_name} Sample {count + 1} ---")
            
            text_elements = translator.extract_text_and_layout(image)
            translated_elements = translator.translate_texts(text_elements)
            inpainted_image = translator.inpaint_image(image, text_elements)
            final_image = translator.render_translated_text(inpainted_image, translated_elements)
            
            final_image.save(output_path)
            print(f"Saved to {output_path}")
            
            count += 1
        except Exception as e:
            print(f"Skipping sample {count} due to error: {e}")

if __name__ == "__main__":
    print("Initializing Models (this takes a moment)...")
    translator = LayoutPreservingTranslator(target_lang="hi")
    
    # Dataset 1: FUNSD (Forms and standard scanned documents)
    test_dataset(translator, "nielsr/funsd-layoutlmv3", "FUNSD", num_samples=2)
    
    # Dataset 2: DocVQA (Receipts, invoices, typed reports)
    test_dataset(translator, "nielsr/docvqa_1200_examples", "DocVQA", num_samples=2)
    
    # Dataset 3: RVL-CDIP (Letters, memos, emails)
    test_dataset(translator, "aharley/rvl_cdip", "RVL_CDIP", num_samples=2)
    
    print("\nFinished testing all alternative datasets! Check the result folders.")
