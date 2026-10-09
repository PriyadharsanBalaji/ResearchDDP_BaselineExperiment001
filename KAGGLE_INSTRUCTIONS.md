# Kaggle Setup Instructions for BaselineExperiment001

Since we are using heavy models like `bodhan-ai/indic-ocr` (0.8B to 1.2B params depending on the version) and `bodhan-ai/indic-translate` (4B params), it's highly recommended to run this on **Kaggle** with a GPU accelerator.

## 1. Setting up the Kaggle Environment

1. Log into your Kaggle account and click **Create -> New Notebook**.
2. Under **Settings** on the right panel:
   - Change the **Accelerator** to **GPU P100** or **GPU T4x2**.
   - Ensure **Internet** is toggled **ON** (needed to download HuggingFace models).

## 2. Clone the Repository into Kaggle

In the very first cell of your Kaggle notebook, run this to clone your code:

```bash
!git clone https://github.com/PriyadharsanBalaji/ResearchDDP_BaselineExperiment001.git
%cd ResearchDDP_BaselineExperiment001
!pip install -r requirements.txt
```

## 3. Prepare an Indic Font (Required for Rendering)

Standard environments don't have Indian language fonts installed.
Download a font like Noto Sans Devanagari (or Tamil/Telugu, etc.) from Google Fonts. 

In a new cell, run:
```bash
!wget "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf" -O NotoSansDevanagari.ttf
```

Make sure you update line 112 in `pipeline.py` to point to `NotoSansDevanagari.ttf` instead of `arial.ttf`.

## 4. Authenticate with Hugging Face (CRITICAL)

Because Bodhan AI's models are "gated" (you have to accept their terms on HuggingFace), you must authenticate before running the script.

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens) and create an Access Token.
2. Ensure you have visited the Bodhan AI model pages and clicked "Agree and access repository".
3. In Kaggle, run this cell and paste your token:

```python
from huggingface_hub import login
login()
```

## 5. Run the Pipeline

Upload a test diagram to your Kaggle workspace (e.g., `test_diagram.jpg`).

Create a new cell and run the pipeline:

```python
from pipeline import LayoutPreservingTranslator

# Initialize the pipeline (downloads the models to Kaggle)
# It will use GPU automatically if available.
translator = LayoutPreservingTranslator(target_lang="hi")

# Run the full process (OCR -> Translate -> Inpaint -> Render)
translator.process("../input/your-dataset/test_diagram.jpg", "translated_diagram.jpg")
```

## 5. Evaluate on AI2D Dataset (Large Scale Test)

To test the pipeline on a sample of the AI2D dataset (educational science diagrams), create a new cell and run:

```python
!python evaluate_ai2d.py
```

This will automatically stream diagrams from HuggingFace, process them through the layout-preserving pipeline, and save the translated versions into an `ai2d_results` folder that you can view in Kaggle.

## 6. Next Steps / Modifications

- The `pipeline.py` script has placeholder model inference code. Once you check the exact input/output formats of the newly released `bodhan-ai/indic-ocr` and `indic-translate` models on HuggingFace, you can edit `extract_text_and_layout()` and `translate_texts()` accordingly directly in Kaggle, and commit the changes back to your GitHub repository.
