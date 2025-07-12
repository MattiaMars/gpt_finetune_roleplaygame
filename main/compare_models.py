from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_text(model, tokenizer, prompt, max_length=50, num_return_sequences=3, temperature=0.9):
    """Generate text using the given model and tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set pad_token for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

def main():
    print("Loading models...")
    
    # Load standard GPT-2
    print("Loading standard GPT-2...")
    standard_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    standard_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Load fine-tuned model
    print("Loading fine-tuned D&D model...")
    finetuned_tokenizer = AutoTokenizer.from_pretrained("./dnd_treasure_gpt2")
    finetuned_model = AutoModelForCausalLM.from_pretrained("./dnd_treasure_gpt2")
    
    # Test prompts
    test_prompts = [
        "A broken and ancient sword",
        "Boots of incredible ",
        "Ring of seductive ",
        "knife of the hidden"
    ]
    
    print("\n" + "="*80)
    print("COMPARISON: Standard GPT-2 vs Fine-tuned D&D Model")
    print("="*80)
    
    for prompt in test_prompts:
        print(f"\n📝 PROMPT: '{prompt}'")
        print("-" * 60)
        
        # Generate with standard GPT-2
        print("🔴 STANDARD GPT-2:")
        try:
            standard_outputs = generate_text(standard_model, standard_tokenizer, prompt)
            for i, text in enumerate(standard_outputs, 1):
                print(f"  {i}. {text}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\n🟢 FINE-TUNED D&D MODEL:")
        try:
            finetuned_outputs = generate_text(finetuned_model, finetuned_tokenizer, prompt)
            for i, text in enumerate(finetuned_outputs, 1):
                print(f"  {i}. {text}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main() 