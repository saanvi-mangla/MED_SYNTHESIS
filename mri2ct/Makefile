.PHONY: help install test clean data train-diffusion train-refiner infer gui

help:
@echo "MRI2CT - Available Commands:"
@echo "  make install           - Install all dependencies"
@echo "  make data             - Create test data"
@echo "  make train-diffusion  - Train diffusion model (test mode)"
@echo "  make train-refiner    - Train GAN refiner (test mode)"
@echo "  make infer            - Run inference"
@echo "  make gui              - Launch interactive GUI"
@echo "  make test             - Run tests"
@echo "  make clean            - Clean generated files"

install:
pip install -r requirements.txt
pip install -e .

data:
mkdir -p data/rire/patient_001
python -c "import torch; import SimpleITK as sitk; import numpy as np; \
from pathlib import Path; \
mri = np.random.randn(128,128,128).astype(np.float32)*0.2+0.5; \
sitk.WriteImage(sitk.GetImageFromArray(mri), 'data/rire/patient_001/mri_t1.nii.gz'); \
ct = (np.random.randn(128,128,128).astype(np.float32)-0.5)*0.3; \
sitk.WriteImage(sitk.GetImageFromArray(ct), 'data/rire/patient_001/ct.nii.gz')"
python prep/splits.py data/rire data/rire_splits.json
@echo "✅ Test data created"

train-diffusion:
python train_diffusion.py \
dataset_name=rire \
epochs=2 \
batch_size=1 \
use_wandb=false \
model.model_channels=32

train-refiner:
python train_refiner.py \
dataset_name=rire \
epochs=2 \
batch_size=1 \
use_wandb=false \
diffusion_checkpoint=checkpoints/diffusion/epoch_2.pth

infer:
python infer_sct.py \
data/rire/patient_001/mri_t1.nii.gz \
outputs/test/ \
--diffusion-checkpoint checkpoints/diffusion/epoch_2.pth \
--num-mc-samples 5

gui:
python interface/gui_app.py

test:
python -m pytest tests/ -v

clean:
rm -rf outputs/* checkpoints/* __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

quickstart: install data train-diffusion
@echo "✅ Quick start complete! Models trained with test data."
@echo "Try: make infer"
