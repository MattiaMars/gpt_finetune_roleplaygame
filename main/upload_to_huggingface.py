from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, login
import os

def upload_model_to_hub():
    """Upload the fine-tuned model to Hugging Face Hub"""
    
    # Login to Hugging Face (you'll need to enter your token)
    print("Please enter your Hugging Face token (get it from https://huggingface.co/settings/tokens):")
    token = input().strip()
    login(token)
    
    # Model details
    model_name = "dnd-treasure-generator"  # You can change this
    model_path = "./dnd_treasure_gpt2"
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Create model card content
    model_card = """
# D&D Treasure Generator

A fine-tuned GPT-2 model specifically designed to generate Dungeons & Dragons casual treasure descriptions.

## Model Description

This model was fine-tuned on a corpus of D&D casual treasures to generate fantasy item descriptions with proper game mechanics and terminology.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/dnd-treasure-generator")
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/dnd-treasure-generator")

# Generate treasure
prompt = "A magical sword"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

- **Base Model**: GPT-2
- **Training Data**: D&D casual treasures corpus
- **Training Time**: ~1 hour 10 minutes
- **Final Loss**: 0.5184 (training), 0.4049 (validation)

## Examples

Input: "Boots of"
Output: "Boots of Speed – Grants the wearer temporary invisibility during combat."

Input: "Ring of"
Output: "Ring of Speed – Increases Intelligence score to 19 while worn."

## License

This model is intended for educational and entertainment purposes in D&D games.
"""
    
    # Upload to Hub
    print(f"Uploading model to {model_name}...")
    
    # Push model and tokenizer
    model.push_to_hub(model_name, private=False)
    tokenizer.push_to_hub(model_name, private=False)
    
    # Create model card
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(model_card)
    
    # Upload README
    api = HfApi()
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=model_name,
        repo_type="model"
    )
    
    print(f"✅ Model uploaded successfully!")
    print(f"🔗 View your model at: https://huggingface.co/{model_name}")
    print(f"📥 Download with: AutoModelForCausalLM.from_pretrained('{model_name}')")

if __name__ == "__main__":
    upload_model_to_hub() 