# Quick Start Guide

Get started with GUI Test Automation in 5 minutes!

---

## 🚀 For Google Colab (Easiest)

### Step 1: Open the Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/diebold-cap/blob/main/notebooks/GUI_Test_Automation_Training.ipynb)

### Step 2: Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4)
3. Click **Save**

### Step 3: Run All Cells
1. Click **Runtime** → **Run all**
2. Wait ~2-3 hours for training to complete
3. Done! Model is trained and ready to use

---

## 💻 For Local/Kaggle Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB RAM

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/diebold-cap.git
cd diebold-cap
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Model
```bash
python train.py --evaluate
```

That's it! Training will start and save checkpoints to `./outputs/`

---

## 🎯 What You'll Get

After training completes, you'll have:

✅ **Trained model** saved in `./outputs/best_model/`
✅ **Evaluation metrics** (BLEU, ROUGE, Exact Match)
✅ **Checkpoints** for each epoch
✅ **Visualization** of predictions vs ground truth

---

## 🧪 Test Your Model

### Quick Test
```python
from src.models.blip2_model import GUITestBLIP2
from PIL import Image

# Load trained model
model = GUITestBLIP2(device='cuda')
model.load_model('./outputs/best_model')

# Load test image
image = Image.open('test_screenshot.png')

# Create prompt
prompt = """Task: Login to application
Previous steps:
1. Open homepage
Current action: None
Predict the next action:"""

# Generate prediction
prediction = model.generate([image], [prompt])
print(f"Predicted action: {prediction[0]}")
```

### Generate Full Sequence
```python
sequence = model.generate_sequence(
    initial_image=image,
    question="Login to the application and view dashboard",
    max_steps=10
)

for step in sequence:
    print(f"Step {step['step_num']}: {step['action']}")
```

---

## 📊 Understanding Results

### Training Output
```
Epoch 1/3: 100%|██████████| 500/500 [25:30<00:00]
  train_loss: 2.145
  val_loss: 1.823
✓ New best validation loss: 1.823

Epoch 2/3: 100%|██████████| 500/500 [25:28<00:00]
  train_loss: 1.687
  val_loss: 1.512
✓ New best validation loss: 1.512

Epoch 3/3: 100%|██████████| 500/500 [25:32<00:00]
  train_loss: 1.423
  val_loss: 1.389
✓ New best validation loss: 1.389
```

### Evaluation Metrics
```
============================================================
Evaluation Results
============================================================
exact_match         : 0.4530 (45.30%)
bleu                : 0.6210 (62.10%)
rouge1              : 0.7245 (72.45%)
rouge2              : 0.5834 (58.34%)
rougeL              : 0.7180 (71.80%)
action_type_acc     : 0.8340 (83.40%)
============================================================
```

**What do these mean?**

- **Exact Match (45%)**: 45% of predictions exactly match ground truth
- **BLEU (62%)**: Measures word overlap - 62% similarity
- **ROUGE-L (72%)**: Measures longest common subsequence
- **Action Type Acc (83%)**: 83% correct action type (click vs type vs verify)

---

## 🎨 Visualizations

The notebook includes several visualizations:

1. **Dataset Exploration** - View example screenshots and actions
2. **Training Progress** - Loss curves over epochs
3. **Prediction Comparison** - Side-by-side predictions vs ground truth
4. **Sequence Generation** - Full test workflows
5. **Per-Workflow Performance** - Metrics breakdown by application

---

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Training
training:
  num_epochs: 3          # Number of epochs
  learning_rate: 5e-5    # Learning rate
  batch_size: 4          # Batch size

# Model
model:
  name: "Salesforce/blip2-flan-t5-base"  # Model variant
  freeze_vision: true    # Freeze vision encoder
  freeze_qformer: false  # Train Q-Former

# Generation
generation:
  max_length: 128        # Max tokens to generate
  num_beams: 4           # Beam search width
  temperature: 1.0       # Sampling temperature
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
# In config.yaml:
batch_size: 2  # Instead of 4
gradient_accumulation_steps: 8  # Instead of 4
```

### Dataset Not Loading
```python
# Manually download dataset
from datasets import load_dataset
dataset = load_dataset("SuperAGI/GUIDE", split="train")
```

### CUDA Not Available
```python
# Use CPU (slower)
device = 'cpu'
model = GUITestBLIP2(device='cpu')
```

### ImportError
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 📖 Next Steps

1. **Experiment with hyperparameters** - Try different learning rates, batch sizes
2. **Use larger model** - Try `blip2-flan-t5-xl` for better performance
3. **Add chain-of-thought** - Set `use_cot: true` in config
4. **Fine-tune on your data** - Replace GUIDE with your own dataset
5. **Deploy model** - Create REST API or web interface

---

## 🤔 Common Questions

### Q: How long does training take?
**A:** ~2-3 hours on Colab T4 GPU for 3 epochs

### Q: Can I use CPU only?
**A:** Yes, but very slow (~20x slower). GPU strongly recommended.

### Q: What GPU do I need?
**A:** Minimum 8GB VRAM (T4, RTX 2070, etc.). 16GB+ for larger batches.

### Q: Can I pause and resume training?
**A:** Yes! Training saves checkpoints every epoch. Use `--resume` flag:
```bash
python train.py --resume ./outputs/checkpoint-epoch-2
```

### Q: How do I improve accuracy?
**A:**
- Train for more epochs
- Use larger model (T5-XL instead of T5-base)
- Increase batch size (if GPU allows)
- Add chain-of-thought (CoT) training

---

## 💡 Tips

1. **Save to Google Drive** - Mount Drive in Colab to persist models
2. **Use WandB** - Set `use_wandb: true` for experiment tracking
3. **Try beam search** - Increase `num_beams` for better quality
4. **Monitor GPU** - Run `nvidia-smi` to check memory usage
5. **Cache dataset** - First run downloads dataset, subsequent runs are faster

---

## 🆘 Get Help

- **GitHub Issues**: [Report a bug](https://github.com/YOUR_USERNAME/diebold-cap/issues)
- **Discussions**: [Ask questions](https://github.com/YOUR_USERNAME/diebold-cap/discussions)
- **Email**: your.email@example.com

---

Happy testing! 🚀
