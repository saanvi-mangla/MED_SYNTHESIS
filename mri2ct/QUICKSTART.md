# Quick Start Guide

##  Get Started in 5 Minutes

### 1. Installation (2 minutes)

```bash
cd mri2ct
pip install -r requirements.txt
```

### 2. Download Example Data

Option A: Use RIRE dataset (recommended for testing)
```bash
# Download from http://www.insight-journal.org/rire/
# Or use provided test data structure:
mkdir -p data/rire/patient_001
# Add your mri_t1.nii.gz and ct.nii.gz files
```

Option B: Generate synthetic test data
```bash
python -c "
import torch
import SimpleITK as sitk
import numpy as np
from pathlib import Path

# Create test patient
Path('data/rire/patient_001').mkdir(parents=True, exist_ok=True)

# Synthetic MRI (brain-like)
mri = np.random.randn(128, 128, 128).astype(np.float32) * 0.2 + 0.5
mri = np.clip(mri, 0, 1)
mri_img = sitk.GetImageFromArray(mri)
sitk.WriteImage(mri_img, 'data/rire/patient_001/mri_t1.nii.gz')

# Synthetic CT
ct = (np.random.randn(128, 128, 128).astype(np.float32) - 0.5) * 0.3
ct_img = sitk.GetImageFromArray(ct)
sitk.WriteImage(ct_img, 'data/rire/patient_001/ct.nii.gz')

print(' Created synthetic test data')
"
```

### 3. Create Data Splits

```bash
python prep/splits.py data/rire data/rire_splits.json
```

### 4. Test the Pipeline

#### Quick Test: Inference (with pretrained models)
```bash
# If you have pretrained models:
python infer_sct.py \
  data/rire/patient_001/mri_t1.nii.gz \
  outputs/test/ \
  --diffusion-checkpoint checkpoints/diffusion/best_model.pth \
  --num-mc-samples 5
```

#### Full Pipeline: Training

**Train Diffusion Model (Test Mode)**
```bash
python train_diffusion.py \
  dataset_name=rire \
  data_root=data/rire \
  epochs=2 \
  batch_size=1 \
  use_wandb=false \
  model.model_channels=32
```

**Train Refiner (Test Mode)**
```bash
python train_refiner.py \
  dataset_name=rire \
  data_root=data/rire \
  diffusion_checkpoint=checkpoints/diffusion/epoch_2.pth \
  epochs=2 \
  batch_size=1 \
  use_wandb=false
```

### 5. Launch Interactive GUI

```bash
python interface/gui_app.py
```

Open browser to: `http://localhost:5006`

##  Expected Directory Structure

After setup, your directory should look like:

```
mri2ct/
 data/
    rire/
        patient_001/
           mri_t1.nii.gz
           ct.nii.gz
        rire_splits.json
 checkpoints/
    diffusion/
    refiner/
 outputs/
    diffusion/
    refiner/
 [source files...]
```

##  Verify Installation

```bash
python example_usage.py
```

This will print usage examples for all components.

##  Quick Troubleshooting

**CUDA Out of Memory**
```bash
# Reduce batch size and model size
python train_diffusion.py batch_size=1 model.model_channels=32
```

**Import Errors**
```bash
# Ensure you're in the right directory
cd mri2ct
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Missing Dependencies**
```bash
# Install optional dependencies
pip install antspyx  # For registration
pip install panel gradio  # For GUI
```

##  Next Steps

1. **Read Full Documentation**: See [README.md](README.md)
2. **Explore Examples**: Run `python example_usage.py`
3. **Check Configurations**: Look at `configs/*.yaml`
4. **Download Real Data**: Get RIRE, HaN-Seg, or SynthRAD datasets
5. **Start Training**: Use provided commands with your data

##  Pro Tips

- Start with small models (`model_channels=32`) for testing
- Use `use_wandb=false` for local testing
- Test with `epochs=2` before full training
- Monitor GPU memory with `nvidia-smi -l 1`
- Use `num_mc_samples=5` for quick inference tests

##  Minimal Working Example

Complete end-to-end in one command (requires data):

```bash
# 1. Train diffusion (2 test epochs)
python train_diffusion.py dataset_name=rire epochs=2 use_wandb=false

# 2. Train refiner (2 test epochs)
python train_refiner.py dataset_name=rire epochs=2 use_wandb=false \
  diffusion_checkpoint=checkpoints/diffusion/epoch_2.pth

# 3. Run inference
python infer_sct.py \
  data/rire/patient_001/mri_t1.nii.gz \
  outputs/result/ \
  --diffusion-checkpoint checkpoints/diffusion/epoch_2.pth \
  --refiner-checkpoint checkpoints/refiner/epoch_2.pth \
  --num-mc-samples 5

# 4. Check outputs
ls -lh outputs/result/
```

##  Verification Checklist

- [ ] All dependencies installed
- [ ] Test data created or downloaded
- [ ] Data splits generated
- [ ] Can import modules: `python -c "import torch; from models.diffusion.unet_transformer import *"`
- [ ] Can run training for 1 epoch
- [ ] Can run inference script
- [ ] GUI launches successfully

---

**Ready to go?** Start with the full README.md for detailed documentation!
