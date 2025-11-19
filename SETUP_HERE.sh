#!/bin/bash
# Simple setup script - Run this from the genaiproj directory

echo "================================="
echo "MRI2CT Project Setup"
echo "================================="
echo ""

# Navigate to mri2ct directory
echo "Navigating to mri2ct directory..."
cd mri2ct || { echo "Error: mri2ct directory not found!"; exit 1; }

echo "Current directory: $(pwd)"
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    exit 1
fi

echo "Installing Python dependencies..."
echo "This will take several minutes..."
echo ""

# Install dependencies
pip install -r requirements.txt

echo ""
echo "================================="
echo "Installation complete!"
echo "================================="
echo ""
echo "Next steps:"
echo "  1. cd mri2ct"
echo "  2. python test_installation.py     # Test the installation"
echo "  3. python example_usage.py         # See usage examples"
echo "  4. cat README.md                   # Read the documentation"
echo ""
