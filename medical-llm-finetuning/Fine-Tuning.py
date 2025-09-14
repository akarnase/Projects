# Step 1: Install the required libraries (run this in your terminal first)
# pip install torch transformers datasets accelerate

# Step 2: Import the necessary tools from the library
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Step 3: Load the Medical Textbook (Dataset)
# This loads our text file as a dataset object the computer can work with.
# We'll name the dataset 'train' because we're using it for training.
dataset = load_dataset("text", data_files={"train": "medical_reports.txt"})

# Step 4: Get the Smart Student and its Dictionary (Model & Tokenizer)
model_name = "gpt2"  # We're starting with the general-purpose GPT-2 model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 wasn't trained with a padding token, so we need to set one.
# We'll just use the End-Of-Sentence token for padding to avoid errors.
tokenizer.pad_token = tokenizer.eos_token

# Load the pre-trained GPT-2 model itself.
# This model is the "student" that has read the internet and now needs medical training.
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 5: Translate the Medical Textbook into Numbers (Tokenization)
def tokenize_medical_reports(examples):
    # This function takes a batch of text and converts it to token IDs (numbers).
    # We truncate and pad all examples to a length of 128 tokens for consistent batch sizes.
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Apply the tokenization function to our entire dataset
tokenized_dataset = dataset.map(
    tokenize_medical_reports,
    batched=True,
    remove_columns=["text"]  # Remove the original text column, we only need the tokens
)

# Step 6: Set Up the Medical Training Regime (Training Arguments)
# These are the settings for how the "tutor" will teach our "student".
training_args = TrainingArguments(
    output_dir="./medical_gpt2",  # Where to save the trained model and checkpoints
    overwrite_output_dir=True,    # Overwrite the output directory if it exists
    num_train_epochs=5,           # How many times to read the entire medical textbook
    per_device_train_batch_size=2,  # Number of examples to process at once (small due to memory)
    save_steps=500,               # Save a checkpoint every 500 steps
    save_total_limit=2,           # Only keep the last 2 checkpoints to save space
    prediction_loss_only=True,    # Only compute loss during training (faster)
    logging_dir="./logs",         # Directory for storing logs
)

# Step 7: Hire the Tutor (The Trainer)
# The Trainer class is our tutor. It handles all the complex training loops.
trainer = Trainer(
    model=model,                   # The student we want to teach (GPT-2)
    args=training_args,            # The lesson plan we defined above
    train_dataset=tokenized_dataset["train"], # The translated medical textbook
)

# Step 8: Start the Medical Class! (Train the Model)
print("Starting medical training...")
trainer.train()  # This is where the magic happens!
print("Training complete!")

# Step 9: Save the Graduated Medical Specialist
# Save the fine-tuned model and tokenizer so we can use it later without retraining.
trainer.save_model("./medical_gpt2_finetuned")
tokenizer.save_pretrained("./medical_gpt2_finetuned")
