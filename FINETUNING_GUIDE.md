# Local MTG Deck Fine-tuning Setup Guide

## Prerequisites

1. **Python 3.8+**
2. **GPU recommended** (NVIDIA with CUDA) but CPU works
3. **8GB+ RAM** (16GB+ recommended)
4. **5GB+ free disk space**

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements_finetuning.txt
```

### 2. Convert Training Data
```bash
python setup_local_finetuning.py
```
This converts your MTG training data to the format needed for fine-tuning.

### 3. Start Training
```bash
python train_mtg_model.py
```

### 4. Test Your Model
```bash
python test_mtg_model.py
```

## Model Options

Edit `MODEL_NAME` in `train_mtg_model.py`:

- `"microsoft/DialoGPT-medium"` - **Recommended** (117M params, good for dialogue)
- `"microsoft/DialoGPT-small"` - Faster training (117M params)
- `"gpt2"` - Classic choice (124M params)  
- `"EleutherAI/gpt-neo-125M"` - Slightly larger (125M params)

## Training Configuration

**Default settings** (adjust in `train_mtg_model.py`):
- **Epochs**: 3
- **Batch size**: 4 per device
- **Learning rate**: 5e-5
- **LoRA rank**: 16
- **Max sequence length**: 1024 tokens

## Expected Training Time

- **GPU (RTX 3080)**: ~30-60 minutes
- **CPU (8 cores)**: ~3-6 hours
- **Training examples**: 143 decks

## Monitoring Progress

Training will show:
```
Step 10/100: Loss: 2.345
Step 20/100: Loss: 2.123
...
```

Lower loss = better learning.

## Model Output Location

Trained model saved to: `./mtg-deck-model/`

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` from 4 to 2 or 1
- Reduce `max_length` from 1024 to 512
- Use `fp16=False` if on CPU

### Slow Training
- Use smaller model (`DialoGPT-small`)
- Reduce `num_train_epochs` from 3 to 1
- Increase `gradient_accumulation_steps`

### Poor Quality Output
- Increase training epochs
- Try different `temperature` values (0.3-1.0) during inference
- Add more training data

## Next Steps

1. **Test different prompts** with `test_mtg_model.py`
2. **Integrate into your main app** by loading the model
3. **Collect feedback** and retrain with better data
4. **Experiment with larger models** if results are promising

## Integration Example

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load your trained model
model = AutoPeftModelForCausalLM.from_pretrained("./mtg-deck-model")
tokenizer = AutoTokenizer.from_pretrained("./mtg-deck-model")

# Generate decks in your app
def generate_deck(archetype, colors):
    prompt = f"Build a competitive {archetype} deck for Standard format."
    # ... (rest of generation logic)
```
