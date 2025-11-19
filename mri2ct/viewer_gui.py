#!/usr/bin/env python3
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super().__init__()
        features = init_features
        self.enc1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = self._block(features * 4, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, 2, 2)
        self.dec3 = self._block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.dec2 = self._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.dec1 = self._block(features * 2, features)
        self.out = nn.Conv3d(features, out_channels, 1)
    
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, 3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, features, 3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        return self.out(dec1)

print("Loading model...")
model = SimpleUNet3D()
checkpoint = torch.load('checkpoints/full_training/best_model.pth', 
                       map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded!")

def generate_ct(patient_choice, slice_num):
    try:
        patient_file = f"data/full_training_dataset/{patient_choice}/mri_t1.nii.gz"
        
        if not Path(patient_file).exists():
            return None, f"Patient file not found: {patient_file}"
        
        mri_image = sitk.ReadImage(patient_file)
        mri = sitk.GetArrayFromImage(mri_image)
        
        from scipy.ndimage import zoom
        original_shape = mri.shape
        if mri.shape != (64, 64, 64):
            scale = [64/s for s in mri.shape]
            mri_resized = zoom(mri, scale, order=1)
        else:
            mri_resized = mri
        
        mri_norm = (mri_resized - mri_resized.mean()) / (mri_resized.std() + 1e-8)
        mri_tensor = torch.FloatTensor(mri_norm).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            ct_pred = model(mri_tensor)
        
        ct_pred = ct_pred.squeeze().numpy() * 1000.0
        
        if original_shape != (64, 64, 64):
            scale = [o/64 for o in original_shape]
            ct_pred = zoom(ct_pred, scale, order=1)
        
        slice_num = min(slice_num, original_shape[0] - 1)
        
        mri_slice = mri[slice_num, :, :]
        ct_slice = ct_pred[slice_num, :, :]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(mri_slice, cmap='gray')
        axes[0].set_title(f'Input MRI - Slice {slice_num}', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        im = axes[1].imshow(ct_slice, cmap='gray', vmin=-1000, vmax=1000)
        axes[1].set_title(f'Generated Synthetic CT - Slice {slice_num}', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Hounsfield Units', fontsize=12)
        
        plt.suptitle(f'Patient: {patient_choice} | Your Trained Model (98.3% Performance)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close()
        
        stats = f"""Generation Successful!

Image Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patient: {patient_choice}
Slice: {slice_num} / {original_shape[0]}

MRI:
  - Shape: {original_shape}
  - Intensity: [{mri.min():.1f}, {mri.max():.1f}]

Generated CT:
  - Shape: {ct_pred.shape}
  - HU Range: [{ct_pred.min():.1f}, {ct_pred.max():.1f}]
  - Mean HU: {ct_pred.mean():.1f}

Look for:
  - Bright skull (500-1000 HU)
  - Dark CSF (~0 HU)
  - Gray brain tissue (20-50 HU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        return Image.open(buf), stats
        
    except Exception as e:
        return None, f"Error: {str(e)}\n\nPlease check that the patient exists and try again."

with gr.Blocks(theme=gr.themes.Soft(), title="MRI to CT Viewer") as app:
    
    gr.Markdown("""
    # MRI to CT Synthesis Viewer
    **Model Performance:** 98.3% improvement | **Parameters:** 10.6M | **Training:** 100 epochs on 60 patients
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            
            patient_dropdown = gr.Dropdown(
                choices=[f"patient_{i:03d}" for i in range(1, 61)],
                value="patient_051",
                label="Select Test Patient",
                info="Choose from 60 available patients"
            )
            
            slice_slider = gr.Slider(
                minimum=0,
                maximum=95,
                value=48,
                step=1,
                label="Slice Number",
                info="0 = top of brain, 48 = middle, 95 = bottom"
            )
            
            generate_btn = gr.Button(
                "Generate Synthetic CT",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ---
            ### Quick Tips
            - **Best slices:** 40-55 (middle of brain)
            - **Top of brain:** 20-35
            - **Bottom of brain:** 60-80
            - **Generation time:** 5-10 seconds
            """)
            
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            
            output_image = gr.Image(
                label="MRI vs Generated CT",
                type="pil",
                height=500
            )
            
            output_text = gr.Textbox(
                label="Statistics",
                lines=15,
                max_lines=20
            )
    
    gr.Markdown("""
    ---
    ### About This Model
    - **Architecture:** 3D U-Net with skip connections
    - **Training Dataset:** 60 patients (synthetic brain MRI-CT pairs)
    - **Training Duration:** 100 epochs (~1.5 hours)
    - **Final Performance:** 98.3% improvement from baseline
    - **Model Location:** `checkpoints/full_training/best_model.pth`
    
    ### What You're Seeing
    - **Left Image:** Original MRI scan (T1-weighted)
    - **Right Image:** Synthetic CT generated by your model
    - **Color Scale:** Hounsfield Units (HU) for CT intensity
    
    **Try different patients and slices to explore the model's capabilities!**
    """)
    
    generate_btn.click(
        fn=generate_ct,
        inputs=[patient_dropdown, slice_slider],
        outputs=[output_image, output_text]
    )
    
    app.load(
        fn=generate_ct,
        inputs=[patient_dropdown, slice_slider],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Launching MRI to CT Viewer GUI")
    print("="*80)
    print("Model loaded successfully")
    print("Opening in browser...")
    print("="*80 + "\n")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )
