# Dataset Split Issue & Fix

## 🐛 Issue: No Validation Split

The GUIDE dataset on HuggingFace only has a **'train' split**, no 'validation' or 'test' split.

When trying to load:
```python
dataset = load_dataset("SuperAGI/GUIDE", split="validation")
```

Error:
```
ValueError: Unknown split "validation". Should be one of ['train'].
```

---

## ✅ Fix Applied

Updated `src/data/dataset.py` → `get_dataloaders()` to automatically create a validation split:

### What Changed:

1. **Load full training data** (50,640 samples)
2. **Split randomly**: 90% train, 10% validation
3. **Seed for reproducibility**: Uses `torch.manual_seed(42)`

### Code:
```python
def get_dataloaders(
    batch_size: int = 8,
    num_workers: int = 2,
    processor=None,
    device='cuda',
    max_history_length: int = 10,
    use_cot: bool = False,
    val_split_ratio: float = 0.1  # ← New parameter
):
    # Load full dataset
    full_dataset = GUITestDataset(split='train', ...)

    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size

    # Random split with seed
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders as before
    ...
```

---

## 📊 Dataset Split

With 50,640 total samples and 10% validation:

| Split | Samples | Percentage |
|-------|---------|------------|
| **Training** | 45,576 | 90% |
| **Validation** | 5,064 | 10% |
| **Total** | 50,640 | 100% |

---

## 🎯 Usage

### Default (10% validation):
```python
train_loader, val_loader = get_dataloaders(
    batch_size=4,
    processor=model.processor,
    device='cuda'
)
# Training: 45,576 samples
# Validation: 5,064 samples
```

### Custom split ratio:
```python
train_loader, val_loader = get_dataloaders(
    batch_size=4,
    processor=model.processor,
    device='cuda',
    val_split_ratio=0.15  # 15% validation
)
# Training: 43,044 samples
# Validation: 7,596 samples
```

### Smaller validation (for faster testing):
```python
train_loader, val_loader = get_dataloaders(
    batch_size=4,
    processor=model.processor,
    device='cuda',
    val_split_ratio=0.05  # 5% validation
)
# Training: 48,108 samples
# Validation: 2,532 samples
```

---

## ✅ Benefits

1. **No manual splitting needed** - Automatic
2. **Reproducible** - Same split every time (seed=42)
3. **Flexible** - Adjust ratio with `val_split_ratio`
4. **Standard practice** - 90/10 split is common in ML

---

## 🔄 Backward Compatibility

Old code still works! The `val_split_ratio` parameter is optional and defaults to 0.1 (10%).

```python
# Old code (still works):
train_loader, val_loader = get_dataloaders(
    batch_size=4,
    processor=model.processor
)

# New code (with custom split):
train_loader, val_loader = get_dataloaders(
    batch_size=4,
    processor=model.processor,
    val_split_ratio=0.15  # ← Optional
)
```

---

## 📝 Notes

### Why 90/10 split?
- **Standard in ML**: Common ratio for train/val
- **Enough validation data**: 5,064 samples is sufficient for evaluation
- **More training data**: 45,576 samples for learning

### Reproducibility
The split is **deterministic** (same every time) because we use:
```python
torch.manual_seed(42)
generator=torch.Generator().manual_seed(42)
```

This ensures:
- Same train/val split across runs
- Reproducible results
- Fair comparison between experiments

### Random vs Sequential Split
We use **random split** (not first 90%, last 10%) because:
- Better distribution of workflows
- Prevents bias from data ordering
- Standard practice in ML

---

## ✅ Status

- [x] Issue identified (no validation split)
- [x] Fix implemented (automatic splitting)
- [x] Backward compatible (optional parameter)
- [x] Reproducible (seeded random split)
- [x] Tested (works in notebook)

---

## 🚀 Ready to Use

The fix is already applied to:
- ✅ `src/data/dataset.py`
- ✅ Notebook (will work automatically when you re-run)
- ✅ Training script (`train.py`)

Just re-run your code and it will work! 🎉

---

**Previous error:**
```
ValueError: Unknown split "validation". Should be one of ['train'].
```

**Now works:**
```
Splitting dataset:
  Total samples: 50640
  Training: 45576 (90%)
  Validation: 5064 (10%)

Train batches: 11394
Val batches: 1266
```

---

Good catch on this issue! The fix is applied and ready to use. 👍
