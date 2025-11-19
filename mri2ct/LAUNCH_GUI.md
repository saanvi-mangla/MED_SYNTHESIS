#  Launch the Interactive Web Interface

## Quick Start (3 Steps)

### Step 1: Navigate to the project directory
```bash
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct
```

### Step 2: Install required packages for GUI
```bash
pip install panel matplotlib numpy torch SimpleITK scipy
```

### Step 3: Launch the web interface
```bash
python interface/gui_app.py
```

The terminal will show something like:
```
Bokeh app running at: http://localhost:5006
```

### Step 4: Open your web browser
Go to: **http://localhost:5006**

---

##  What You'll See in the Interface

### Left Sidebar:
1. **Model Configuration**
   - Load diffusion model checkpoint
   - Load GAN refiner checkpoint (optional)
   
2. **Input Section**
   - Upload MRI file (.nii or .nii.gz)
   
3. **Parameters**
   - MC Samples: Number of uncertainty samples (5-50)
   - Confidence Level: 0.5 to 0.99
   - Apply N4 Bias Correction: Yes/No
   - **Generate CT Button** ← Click this to run!

### Main Display Area:
Four panels showing:
- **Input MRI**: Your uploaded MRI scan
- **Synthetic CT**: Generated CT image
- **Uncertainty Map**: Voxel-wise uncertainty (heatmap)
- **Coverage Mask**: Confidence coverage visualization

### Controls:
- **View**: Switch between Axial/Coronal/Sagittal
- **Slice Slider**: Navigate through 3D volume
- **Save Results**: Export generated images

---

##  Full Setup (If Starting Fresh)

```bash
# 1. Navigate to project
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct

# 2. Install ALL dependencies (takes 10-20 minutes)
pip install -r requirements.txt

# 3. Launch GUI
python interface/gui_app.py

# 4. Open browser to http://localhost:5006
```

---

##  Testing Without Trained Models

You can explore the GUI interface even without trained models:

```bash
# Just launch it
python interface/gui_app.py
```

The interface will load, and you can:
- Explore the layout
- Upload test MRI files
- See the interface design

**Note**: To actually generate CT images, you need trained model checkpoints.

---

##  Quick Demo Mode

Want to see it working immediately? Create a minimal demo:

### Option A: Launch with mock data
```bash
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct

# Create simple test data first
python << 'DEMO'
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Create demo MRI
Path('demo_data').mkdir(exist_ok=True)
demo_mri = np.random.randn(100, 100, 100).astype(np.float32) * 0.3 + 0.5
demo_mri = np.clip(demo_mri, 0, 1)
mri_img = sitk.GetImageFromArray(demo_mri)
sitk.WriteImage(mri_img, 'demo_data/demo_mri.nii.gz')
print(" Created demo MRI at: demo_data/demo_mri.nii.gz")
DEMO

# Launch GUI
python interface/gui_app.py
```

Then:
1. Open http://localhost:5006
2. Upload `demo_data/demo_mri.nii.gz`
3. Explore the interface!

---

##  Alternative: Simplified Viewer

If the full GUI doesn't work, try this simple viewer:

```bash
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct

# Create simple viewer
python << 'VIEWER'
import matplotlib.pyplot as plt
import numpy as np

print(" Simple MRI Viewer")
print("This creates a simple visualization...")

# Create sample data
mri = np.random.randn(100, 100, 100) * 0.3 + 0.5
ct = np.random.randn(100, 100, 100) * 0.3
uncertainty = np.random.rand(100, 100, 100) * 0.1

# Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
slice_idx = 50

axes[0, 0].imshow(mri[slice_idx], cmap='gray')
axes[0, 0].set_title('Input MRI')
axes[0, 0].axis('off')

axes[0, 1].imshow(ct[slice_idx], cmap='gray')
axes[0, 1].set_title('Synthetic CT')
axes[0, 1].axis('off')

axes[1, 0].imshow(uncertainty[slice_idx], cmap='hot')
axes[1, 0].set_title('Uncertainty Map')
axes[1, 0].axis('off')

axes[1, 1].imshow(uncertainty[slice_idx] < 0.05, cmap='RdYlGn')
axes[1, 1].set_title('Coverage Mask')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('demo_output.png', dpi=150, bbox_inches='tight')
print(" Saved visualization to: demo_output.png")
plt.show()
VIEWER
```

---

##  Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'panel'"
```bash
pip install panel matplotlib
```

### Issue: "Address already in use"
The port 5006 is busy. Try a different port:
```bash
# Edit interface/gui_app.py, change last line:
# app.show(port=5007)  # or any other port
python interface/gui_app.py
```

### Issue: GUI loads but "Models not loaded"
You need trained model checkpoints. For now, you can:
1. Explore the interface without models
2. Train models first (see training commands below)
3. Or just view the interface design

### Issue: Panel server won't start
Try the simpler Gradio interface (I can create one for you) or use the matplotlib viewer above.

---

##  Training Models First (Optional)

To get actual CT generation working, you need to train models:

```bash
# Quick test training (2 epochs, ~10 minutes)
python train_diffusion.py \
  dataset_name=rire \
  epochs=2 \
  batch_size=1 \
  use_wandb=false \
  model.model_channels=32

# This creates: checkpoints/diffusion/epoch_2.pth
```

Then use that checkpoint in the GUI!

---

##  Quick Commands Reference

```bash
# From anywhere:
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct

# Install GUI dependencies:
pip install panel matplotlib SimpleITK numpy torch scipy

# Launch interface:
python interface/gui_app.py

# Open browser:
# http://localhost:5006
```

---

##  Expected Result

When working correctly, you'll see:
-  Web page opens at localhost:5006
-  Interface with 4 image panels
-  Sidebar with controls
-  File upload button
-  Sliders for parameters
-  Generate CT button

**Even without trained models**, the interface will load and you can explore the design!

---

## Next Steps After Launching

1. **Explore the interface** - Click around, see the layout
2. **Upload a test MRI** - Use the file upload
3. **Adjust parameters** - Try different settings
4. **Train models** (optional) - For actual CT generation
5. **View documentation** - Read README.md for more details

---

**Ready to launch?** Run: `python interface/gui_app.py` 
