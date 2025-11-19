#  Complete Guide: How to Use the MRI→CT Interface

##  Quick Start (3 Commands)

```bash
# 1. Navigate to project
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct

# 2. Install required packages
pip install panel matplotlib numpy

# 3. Launch DEMO interface (works without models!)
python interface/demo_gui.py
```

**Then open:** http://localhost:5006

---

##  Two Versions Available

### Version 1: **DEMO GUI** (Recommended to Start)
-  Works immediately, no models needed
-  Shows how the interface works
-  Generates synthetic demo results
- **Launch:** `python interface/demo_gui.py`

### Version 2: **Full GUI** (Requires Trained Models)
-  Needs trained model checkpoints
-  Generates real CT from MRI
-  Real uncertainty quantification
- **Launch:** `python interface/gui_app.py`

**Start with the DEMO version!** 

---

##  Step-by-Step: Using the DEMO Interface

### Step 1: Launch the Interface

```bash
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct
python interface/demo_gui.py
```

You'll see:
```
========================================================
  MRI→CT DEMO INTERFACE
========================================================

 Starting demo interface...
 This version works WITHOUT trained models!

 Opening web browser...
 Go to: http://localhost:5006
```

### Step 2: Open Your Browser

- **Automatically:** Browser should open to http://localhost:5006
- **Manually:** Type http://localhost:5006 in your browser address bar

### Step 3: You'll See the Interface

**Left Sidebar (Controls):**
```

  Input                            
  
   Upload MRI                    
  (any image format)               
  
                                     
          **OR**                     
                                     
  
   Generate Random Demo MRI       CLICK THIS!
  
                                     
  Parameters                       
   MC Samples: [slider]              
   Confidence: [slider]              
                                     
  
   Generate Synthetic CT           THEN CLICK THIS!
  

```

**Right Area (Visualization):**
```

 View: [Axial|Coronal|Sagittal]  Slice: [===] 

   Input MRI     Synthetic CT            
  [image]         [image]                     

  Uncertainty    Coverage Mask           
  [heatmap]       [mask]                      

```

### Step 4: Generate Demo Data

**Option A: Click " Generate Random Demo MRI"**
- Creates synthetic brain-like MRI data
- Instantly ready to use
-  **Recommended for first try!**

**Option B: Upload Your Own File**
- Click " Upload MRI"
- Select any image file (doesn't have to be medical!)
- The demo will convert it to 3D data

### Step 5: Generate Synthetic CT

1. After loading/generating MRI, click **" Generate Synthetic CT"**
2. Wait 1-2 seconds
3. All 4 panels will populate with images!

### Step 6: Explore the Results

**Navigate Through Slices:**
- Use the **Slice slider** at the top to scroll through 3D volume
- Drag it left/right to see different slices

**Change View:**
- Click **Axial** (horizontal slices)
- Click **Coronal** (front-to-back)
- Click **Sagittal** (side-to-side)

**Adjust Parameters:**
- Change **MC Samples** (affects uncertainty calculation)
- Change **Confidence Level** (affects coverage mask)
- Click "Generate Synthetic CT" again to see updated results

### Step 7: Save Results

- Click **" Save Results"** button
- Files saved to `demo_outputs/` folder:
  - `demo_results.png` - All 4 panels in one image
  - `demo_mri.npy` - Raw MRI data
  - `demo_ct.npy` - Generated CT data
  - `demo_uncertainty.npy` - Uncertainty map
  - `demo_coverage.npy` - Coverage mask

---

##  Interface Controls Explained

### Left Sidebar

| Control | Purpose | How to Use |
|---------|---------|------------|
| **Upload MRI** | Load your own data | Click → Browse → Select file |
| **Generate Random** | Create demo data | Click button → Data appears |
| **MC Samples** | # of uncertainty samples | Drag slider (5-50) |
| **Confidence Level** | Coverage threshold | Drag slider (0.5-0.99) |
| **Generate CT** | Run the translation | Click to process |
| **Save Results** | Export outputs | Click to save to folder |

### Top Controls

| Control | Purpose | How to Use |
|---------|---------|------------|
| **View** | Change slice orientation | Click Axial/Coronal/Sagittal |
| **Slice Slider** | Navigate 3D volume | Drag left/right |

### Four Image Panels

| Panel | Shows | Color Map |
|-------|-------|-----------|
| **Input MRI** | Original scan | Grayscale |
| **Synthetic CT** | Generated CT | Grayscale |
| **Uncertainty Map** | Prediction confidence | Hot (red=uncertain) |
| **Coverage Mask** | Reliable regions | Green=good, Red=uncertain |

---

##  Using the FULL Interface (With Real Models)

Once you have trained models or have checkpoint files:

### Step 1: Launch Full Interface

```bash
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct
python interface/gui_app.py
```

### Step 2: Load Model Checkpoints

In the **Model Configuration** section:

1. **Diffusion Checkpoint:**
   - Type or paste path: `checkpoints/diffusion/best_model.pth`
   - Or full path: `/Users/saanvimangla/Desktop/genaiproj/mri2ct/checkpoints/diffusion/best_model.pth`

2. **Refiner Checkpoint (Optional):**
   - Type: `checkpoints/refiner/best_model.pth`
   - Leave empty if you only trained diffusion

3. Click **"Load Models"** button

4. Wait for: **" Models loaded successfully"**

### Step 3: Upload Real MRI

1. Click **" Upload MRI (NIfTI)"**
2. Browse to your `.nii.gz` or `.nii` file
3. Select and open

### Step 4: Adjust Parameters

- **MC Samples:** 20 (default) - More = better uncertainty, slower
- **Confidence Level:** 0.95 (95% confidence intervals)
- **Apply N4 Bias Correction:**  (recommended for real MRI)

### Step 5: Generate Real CT

1. Click **"Generate CT"** button
2. Wait (this takes 2-5 minutes for real generation)
3. Progress bar shows status
4. Results appear when complete!

---

##  File Formats Supported

### DEMO Interface:
-  Any image file (.png, .jpg, .jpeg)
-  NIfTI (.nii, .nii.gz)
- Converts everything to 3D for demo

### FULL Interface:
-  NIfTI (.nii, .nii.gz) - Standard medical imaging format
-  Compressed NIfTI (.nii.gz) - Recommended

---

##  Common Tasks

### Task 1: Just Want to See the Interface?
```bash
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct
python interface/demo_gui.py
# Click "Generate Random Demo MRI"
# Click "Generate Synthetic CT"
# Done!
```

### Task 2: Upload Your Own Image
```bash
# Launch demo
python interface/demo_gui.py
# Click "Upload MRI"
# Select any image
# Click "Generate Synthetic CT"
```

### Task 3: Save Results for Presentation
```bash
# After generating CT in the interface
# Click "Save Results"
# Check demo_outputs/demo_results.png
# This PNG contains all 4 panels!
```

### Task 4: Create a Screenshot
```bash
# 1. Generate results in the interface
# 2. Use your computer's screenshot tool:
#    Mac: Cmd + Shift + 4 (select area)
#    Windows: Windows + Shift + S
```

---

##  Troubleshooting

### Problem: Interface won't load

**Solution:**
```bash
# Install panel
pip install panel matplotlib numpy

# Try launching again
python interface/demo_gui.py
```

### Problem: "Address already in use"

**Solution:**
```bash
# Kill existing server
pkill -f panel

# Or use different port
# Edit demo_gui.py last line:
# app.show(port=5007)  # Change 5006 to 5007
```

### Problem: Can't see images

**Solution:**
1. Make sure you clicked "Generate Random Demo MRI" first
2. Then click "Generate Synthetic CT"
3. Refresh browser page (F5)

### Problem: "Generate CT" button is grayed out

**Solution:**
- You need to load/generate MRI data first!
- Click "Generate Random Demo MRI" button
- Button will become enabled

### Problem: Want to upload real NIfTI but only have DICOM

**Solution:**
```bash
# Convert DICOM to NIfTI first
python prep/convert_dicom.py \
  /path/to/dicom/folder \
  output.nii.gz
```

---

##  Tips & Tricks

### Tip 1: Fastest Way to See Results
```bash
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct
python interface/demo_gui.py
# Click: Generate Random Demo MRI
# Click: Generate Synthetic CT
# Done in 5 seconds!
```

### Tip 2: Create Multiple Views
- Change View to Axial, save results
- Change View to Coronal, save results
- Change View to Sagittal, save results
- Now you have 3 different angle views!

### Tip 3: Compare Confidence Levels
- Generate with Confidence = 0.68
- Save results
- Generate with Confidence = 0.95
- Save results
- Compare the coverage masks!

### Tip 4: Make Presentation Images
1. Generate results
2. Click "Save Results"
3. Open `demo_outputs/demo_results.png`
4. This 4-panel image is perfect for presentations!

---

##  Understanding the Output

### Input MRI (Top Left)
- Your original MRI scan
- Grayscale: bright = high signal, dark = low signal

### Synthetic CT (Top Right)
- Generated CT image
- Grayscale: bright = dense tissue (bone), dark = soft tissue

### Uncertainty Map (Bottom Left)
- Shows prediction confidence
- **Red/Yellow = High uncertainty** (model is unsure)
- **Dark = Low uncertainty** (model is confident)

### Coverage Mask (Bottom Right)
- Shows reliable predictions
- **Green = Inside confidence interval** (trustworthy)
- **Red = Outside confidence interval** (unreliable)

---

##  Quick Command Reference

```bash
# Navigate to project
cd /Users/saanvimangla/Desktop/genaiproj/mri2ct

# Install dependencies
pip install panel matplotlib numpy

# Launch DEMO (no models needed)
python interface/demo_gui.py

# Launch FULL (needs trained models)
python interface/gui_app.py

# Open in browser
http://localhost:5006

# Stop server
Press Ctrl+C in terminal
```

---

##  Success Checklist

- [ ] Launched the interface successfully
- [ ] Generated or uploaded MRI data
- [ ] Clicked "Generate Synthetic CT"
- [ ] Saw all 4 images populate
- [ ] Used slice slider to navigate
- [ ] Changed view (Axial/Coronal/Sagittal)
- [ ] Saved results to file
- [ ] Found output in demo_outputs/ folder

**All checked?** Congratulations!  You've successfully used the interface!

---

##  Next Steps

1. **Explore More:** Try different confidence levels and settings
2. **Upload Real Data:** Use actual MRI files if you have them
3. **Train Models:** Follow README.md to train real models
4. **Read Documentation:** Check out README.md for full details

---

**Need Help?** Check the troubleshooting section above or read the README.md file!

**Ready to start?** Run: `python interface/demo_gui.py` 
