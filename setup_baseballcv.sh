#!/bin/bash
#Executable to setup environment, model files and dataset for glove tracking fine tuning

# Create a new conda environment named "baseball_venv"
if ! conda env list | grep -q "^\s*baseball_venv\s"; then
    echo "Creating conda environment 'baseball_venv'..."
    conda create --name baseball_venv python=3.10 -y
else
    echo "Conda environment 'baseball_venv' already exists. Skipping creation."
fi

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate baseball_venv

# Clone the BaseballCV repository
if [ ! -d "BaseballCV" ]; then

    echo "Cloning BaseballCV repository..."
    git clone https://github.com/dylandru/BaseballCV.git

    # Navigate into the cloned repository
    cd BaseballCV

    # Install dependencies using poetry
    if ! pip show baseballcv &> /dev/null; then
        echo "Installing project dependencies with Poetry..."
        pip install poetry && poetry install
        echo "Installing optuna..."
        pip install optuna
    else
        echo "baseballcv package is already installed. Skipping dependency installation."
    fi

    echo "Navigating back to the parent directory..."
    cd ..
else
    echo "BaseballCV directory already exists. Skipping clone."
fi

DATASET_FILE="baseball_rubber_home_glove.zip"
MODEL_FILE="glove_tracking_v4_YOLOv11.pt"

# Define download URLs
DATASET_URL="https://data.balldatalab.com/index.php/s/pLy7sZqqMdx3jj7/download/baseball_rubber_home_glove.zip"
MODEL_URL="https://data.balldatalab.com/index.php/s/BwwWJbSsesFSBDa/download/glove_tracking_v4_YOLOv11.pt"

# Download dataset if it doesn't exist
if [ ! -f "$DATASET_FILE" ]; then
    echo "Downloading dataset..."
    wget -O "$DATASET_FILE" "$DATASET_URL"
    echo "Unzipping dataset..."
    unzip "$DATASET_FILE"
else
    echo "Dataset already exists. Skipping download."
fi

# Download model if it doesn't exist
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model..."
    wget -O "$MODEL_FILE" "$MODEL_URL"
else
    echo "Model file already exists. Skipping download."
fi

echo -e "\nSetup complete!"
echo "To begin working, activate the environment by running:"
echo "conda activate baseball_venv"
