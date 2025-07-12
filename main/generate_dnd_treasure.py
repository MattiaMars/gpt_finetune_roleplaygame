from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "./dnd_treasure_gpt2"
MAX_LENGTH = 50
NUM_RETURN_SEQUENCES = 5

# User input for initial text
prompt = input("Enter the beginning of a D&D casual treasure description: ")
if not prompt.strip():
    prompt = "A casual treasure: "

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)

# Generate text
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    input_ids,
    max_length=MAX_LENGTH,
    num_return_sequences=NUM_RETURN_SEQUENCES,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
)

print("\nGenerated D&D Casual Treasures:")
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"{i+1}: {text}") 