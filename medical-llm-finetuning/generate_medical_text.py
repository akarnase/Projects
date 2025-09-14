from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your fine-tuned medical specialist
model_path = "./medical_gpt2_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create a medical prompt
prompt = "Patient presents with fever and a persistent cough"
# Alternatively, try: "A 60-year-old male with history of smoking presents with"

# Tokenize the prompt (translate to numbers)
inputs = tokenizer(prompt, return_tensors="pt")

# Generate a medical continuation
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,                # Maximum length of the generated text
    num_return_sequences=1,        # Number of different responses to generate
    temperature=0.7,               # Controls randomness: lower = more deterministic
    top_p=0.9,                     # Nucleus sampling: choose from top tokens making 90% of probability
    do_sample=True,                # Sample from the probability distribution
    pad_token_id=tokenizer.eos_token_id # Set the pad token to avoid warnings
)

# Decode the generated numbers back into text and print it
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)