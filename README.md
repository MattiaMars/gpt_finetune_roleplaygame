# D&D Treasure Generator

A fine-tuned GPT-2 model specifically designed to generate Dungeons & Dragons casual treasure descriptions with proper game mechanics and terminology.

## 🎯 Project Overview

This project fine-tunes a GPT-2 model on a corpus of D&D casual treasures to create a specialized text generation model for fantasy role-playing games.

## 📊 Model Performance

- **Base Model**: GPT-2
- **Training Data**: 1001 D&D casual treasures
- **Training Time**: ~1 hour 10 minutes
- **Final Training Loss**: 0.5184
- **Final Validation Loss**: 0.4049
- **Data Split**: 95% training, 5% validation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd 250711_Cursor_FineTuneDnDModel
```

2. **Set up virtual environment**
```bash
# Windows
setup_env.bat

# Or manually:
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. **Train the model**
```bash
python finetune_dnd_treasures.py
```

4. **Generate treasures**
```bash
python generate_dnd_treasure.py
```

## 📁 Project Structure

```
250711_Cursor_FineTuneDnDModel/
├── dnd_treasures.txt              # Training corpus (1001 D&D treasures)
├── finetune_dnd_treasures.py      # Training script
├── generate_dnd_treasure.py       # Text generation script
├── compare_models.py              # Model comparison script
├── plot_training_metrics.py       # Training visualization
├── upload_to_huggingface.py       # Hugging Face upload script
├── requirements.txt               # Python dependencies
├── setup_env.bat                  # Environment setup (Windows)
├── dnd_treasure_gpt2/            # Fine-tuned model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
└── README.md                      # This file
```

## 🎲 Usage Examples

### Generate D&D Treasures
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./dnd_treasure_gpt2")
model = AutoModelForCausalLM.from_pretrained("./dnd_treasure_gpt2")

# Generate treasure
prompt = "A magical sword"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Example Outputs

**Input:** "Boots of"
**Output:** "Boots of Speed – Grants the wearer temporary invisibility during combat."

**Input:** "Ring of"
**Output:** "Ring of Speed – Increases Intelligence score to 19 while worn."

**Input:** "A magical sword"
**Output:** "A magical sword of Giant Strength – Summons a spectral beast for 1 hour."

## 📈 Training Results

The model shows excellent convergence:
- Training loss decreased from ~1.13 to 0.52
- Validation loss decreased from ~0.41 to 0.40
- Model successfully learned D&D terminology and game mechanics

## 🔍 Model Comparison

The fine-tuned model significantly outperforms the base GPT-2 model:

| Aspect | Standard GPT-2 | Fine-tuned Model |
|--------|----------------|------------------|
| **Relevance** | ❌ Off-topic | ✅ D&D focused |
| **Terminology** | ❌ Generic | ✅ Fantasy/RPG terms |
| **Mechanics** | ❌ None | ✅ D&D game mechanics |
| **Consistency** | ❌ Random | ✅ Thematic consistency |
| **Usefulness** | ❌ Not usable | ✅ Ready for D&D games |

## 🛠️ Scripts

- **`finetune_dnd_treasures.py`**: Trains the model on D&D treasures corpus
- **`generate_dnd_treasure.py`**: Interactive text generation
- **`compare_models.py`**: Compares fine-tuned vs standard GPT-2
- **`plot_training_metrics.py`**: Visualizes training progress
- **`upload_to_huggingface.py`**: Uploads model to Hugging Face Hub

## 📦 Dependencies

- `transformers>=4.53.1`
- `datasets>=4.0.0`
- `torch>=2.7.1`
- `matplotlib>=3.10.3`
- `accelerate>=0.26.0`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is intended for educational and entertainment purposes in D&D games.

## 🙏 Acknowledgments

- Hugging Face for the transformers library
- OpenAI for the base GPT-2 model
- The D&D community for inspiration

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Happy adventuring! 🐉⚔️** 