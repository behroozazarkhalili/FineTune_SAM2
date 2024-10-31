#!/bin/bash

#==============================================================================
# SAM2 Setup and Installation Script
# Purpose: Automate the setup process for SAM2 (Segment Anything Model 2)
# Author: Your Name
# Date: 2024
#==============================================================================

#------------------------------------------------------------------------------
# Function Definitions
#------------------------------------------------------------------------------

# Function to check if a command was successful
check_error() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

#------------------------------------------------------------------------------
# 1. Clone and Install SAM2
#------------------------------------------------------------------------------
echo "Cloning SAM2 repository..."
git clone https://github.com/facebookresearch/segment-anything-2
check_error "Failed to clone SAM2 repository"

# Change to SAM2 directory and install
echo "Installing SAM2..."
cd segment-anything-2
pip3 install -e .
check_error "Failed to install SAM2"

#------------------------------------------------------------------------------
# 2. Download Model Checkpoints
#------------------------------------------------------------------------------
echo "Downloading model checkpoints..."
cd checkpoints
./download_ckpts.sh
check_error "Failed to download checkpoints"
cd ../

#------------------------------------------------------------------------------
# 3. Install Additional Requirements
#------------------------------------------------------------------------------
echo "Installing additional Python packages..."
pip3 install pipreqs pipreqsnb
check_error "Failed to install pip requirements tools"

#------------------------------------------------------------------------------
# 4. Data Preparation
#------------------------------------------------------------------------------
# Create directory for training data
echo "Setting up training data directory..."
mkdir -p ./assets/Sam2-Train-Data

# Extract all zip files from training data directory
echo "Extracting training data..."
find ../Sam2-Training-Data -type f -name '*.zip' -exec unzip -o {} -d ./assets/Sam2-Train-Data \;
check_error "Failed to extract training data"

#------------------------------------------------------------------------------
# 5. Setup Finetuning Environment
#------------------------------------------------------------------------------
# Create and setup finetuning directory
echo "Setting up finetuning environment..."
mkdir -p src-finetuning
cp -r ../src/* src-finetuning/
check_error "Failed to copy source files"

# Install finetuning requirements
echo "Installing finetuning dependencies..."
cd src-finetuning
pip3 install -r requirements.txt
check_error "Failed to install finetuning requirements"

# Install system dependencies
echo "Installing system dependencies..."
apt install --fix-missing -y libgl1
check_error "Failed to install system dependencies"

#------------------------------------------------------------------------------
# Final Status Check
#------------------------------------------------------------------------------
echo "Setup completed successfully!"
echo "You can now proceed with training in the src-finetuning directory."

# End of script