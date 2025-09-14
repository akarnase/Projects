# Medical Text Generation by Fine-Tuning GPT-2

## Project Overview
This project demonstrates my ability to fine-tune large language models (LLMs) for domain-specific tasks. I took a pre-trained GPT-2 model and specialized it to generate coherent medical reports using the Hugging Face `transformers` library and PyTorch.

**Key Skills Demonstrated:**
- Natural Language Processing (NLP)
- Transfer Learning & Model Fine-Tuning
- Hugging Face Ecosystem (Transformers, Datasets, PEFT)
- PyTorch
- Parameter-Efficient Fine-Tuning (LoRA)
- Working with GPU accelerators (e.g., Google Colab)

## Dataset
I used a synthetic dataset of medical reports to train the model. The data includes descriptions of symptoms, diagnoses, and treatments to teach the model clinical language patterns.

*(Note: In a real-world scenario, this would be done with a large, real dataset of de-identified medical notes.)*

## Methodology
1.  **Model Selection:** Started with the pre-trained `gpt2` model.
2.  **Data Preprocessing:** Tokenized the medical text using the GPT-2 tokenizer.
3.  **Efficient Training:** Employed **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning, allowing me to adapt the large model on a single GPU without catastrophic forgetting.
4.  **Training:** Fine-tuned the model for several epochs, monitoring the loss.
5.  **Inference:** Tested the model by providing prompts like "Patient presents with..." to generate medically-plausible text continuations.

## Results
The fine-tuned model successfully learned to generate text in a clinical style.

**Example Prompt:**
`"Patient presents with fever and a persistent cough"`

**Model Output:**
`"Patient presents with fever and a persistent cough for three days. Chest X-ray shows bilateral infiltrates. COVID-19 test is ordered. Patient is placed on supportive care and oxygen therapy."`

## How to Run the Code
1.  Clone the repo:
    ```bash
    git clone https://github.com/your-username/medical-llm-finetuning.git
    ```
2.  Install dependencies:
    ```bash
    pip install torch transformers datasets accelerate peft
    ```
3.  Run the training script:
    ```bash
    python finetune_medical_gpt2.py
    ```
4.  Generate text:
    ```bash
    python generate_medical_text.py
    ```

## Files in this Repository
- `finetune_medical_gpt2.py` - The main script for fine-tuning.
- `generate_medical_text.py` - Script to generate text with the fine-tuned model.
- `requirements.txt` - List of Python dependencies.
- `README.md` - This file.