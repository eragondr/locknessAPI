#!/bin/bash
#
#echo "========================================"
#echo "Starting 3DAIGC-API Installation"
#echo "========================================"
#echo "The installation may take a while, please wait..."
#echo ""
#




set -e

# Usage: bash install.sh 124   or   bash install.sh 126
CUDA_VER=$1
ISMESH=$2

if [ -z "$CUDA_VER" ]; then
    echo "[ERROR] Please provide CUDA version (e.g. 124 or 126)"
    echo "Usage: bash install.sh <cuda_version>"
    exit 1
fi
if [ -z "$ISMESH" ]; then
    echo "[ERROR] Missing ismesh argument (1 or 0)"
    echo "Usage: bash install.sh <cuda_version> <ismesh>"
    exit 1
fi
# Initialize conda
#conda init bash
##echo "[INFO] Creating conda environment 'locknessapi' with Python 3.10..."
#conda create -n locknessapi python=3.10 -y
#if [ $? -eq 0 ]; then
#    echo "[SUCCESS] Conda environment created successfully"
#else
#    echo "[ERROR] Failed to create conda environment"
#    exit 1
#fi


#conda activate locknessapi

echo "[INFO] Download models"
if [ "$ISMESH" == "1" ]; then
    bash downloadMesh.sh
elif [ "$ISMESH" == "0" ]; then
    bash downloadPaint.sh
else
    echo "no download"
fi

echo "[INFO] install Cuda"
if [ "$CUDA_VER" == "121" ]; then
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VER" == "124" ]; then
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
elif [ "$CUDA_VER" == "126" ]; then
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
elif [ "$CUDA_VER" == "128" ]; then
    pip3 install torch torchvision
elif [ "$CUDA_VER" == "13" ]; then
    pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130


else
    echo "[ERROR] Unsupported CUDA version: $CUDA_VER"
    echo "Supported values: 124, 126"
    exit 1
fi



cd ..


echo ""
echo "Current directory: $(pwd)"
echo "Installing Hunyuan3D21 Dependencies"
### installation for Hunyuan3D21  ###
echo "========================================"
echo "[INFO] Changing directory to thirdparty/Hunyuan3D21..."
echo "========================================"
cd thirdparty/Hunyuan3D21
echo "Current directory: $(pwd)"
echo "[INFO] Installing custom rasterizer for Hunyuan3D21..."
pip install -r requirements-inference.txt
pip install -r requirements.txt
pip install bpy==4.0 --extra-index-url https://download.blender.org/pypi/
pip install boto3==1.40.43
pip install pyglet==2.1.9
pip install torch-cluster==1.6.3
pip install easydict==1.13
pip install lightning==2.5.5
pip install plyfile==1.1.2
pip install kiui==0.2.18
pip install pydantic_settings
pip install sqlalchemy
pip install mmgp
sudo apt install libegl-mesa0
pip install kaolin
#pip install pybind11


echo "[INFO] Changing directory to hy3dpaint/custom_rasterizer..."
echo "Current directory: $(pwd)"
cd hy3dpaint/custom_rasterizer
set DISTUTILS_USE_SDK=1
pip install -e . --no-build-isolation
python setup.py install
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Hunyuan3D21 custom rasterizer installed"
else
    echo "[ERROR] Failed to install Hunyuan3D21 custom rasterizer"
    exit 1
fi
cd ..

echo "[INFO] Building differentiable renderer for Hunyuan3D21..."
echo "Before directory after cd: $(pwd)"
cd ..
cd hy3dpaint/DifferentiableRenderer
echo "Current directory after cd: $(pwd)"
bash compile_mesh_painter.sh
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Hunyuan3D21 differentiable renderer built successfully"
else
    echo "[ERROR] Failed to build Hunyuan3D21 differentiable renderer"
    exit 1
fi
cd ../..
echo "[INFO] Installing Hunyuan3D21 requirements..."

### installation for Hunyuan3D21 end ###
echo "[SUCCESS] Hunyuan3D21 installation completed"
echo "Before directory after cd: $(pwd)"
cd ../..


#echo ""
#echo "========================================"
#echo "Installing HoloPart Dependencies"
#echo "========================================"
#### holopart for part completion  ###
#echo "[INFO] Changing directory to thirdparty/HoloPart..."
#cd ../../thirdparty/HoloPart
#echo "[INFO] Installing HoloPart requirements..."
#pip install -r requirements.txt
#if [ $? -eq 0 ]; then
#    echo "[SUCCESS] HoloPart requirements installed"
#else
#    echo "[ERROR] Failed to install HoloPart requirements"
#    exit 1
#fi
#### holopart for part completion end ###
#echo "[SUCCESS] HoloPart installation completed"
#
#echo ""
#echo "========================================"
#echo "Installing UniRig Dependencies"
#echo "========================================"
#### unirig for auto-rigging  ###
#echo "[INFO] Changing directory to thirdparty/UniRig..."
#cd ../../thirdparty/UniRig
#echo "[INFO] Installing spconv-cu120 for UniRig..."
#pip install spconv-cu120
#pip install pyrender fast-simplification python-box timm
#if [ $? -eq 0 ]; then
#    echo "[SUCCESS] UniRig dependencies installed"
#else
#    echo "[ERROR] Failed to install UniRig dependencies"
#    exit 1
#fi
#
#echo ""
#echo "========================================"
#echo "Installing PartPacker Dependencies"
#echo "========================================"
#### part packer  ###
#echo "[INFO] Changing directory to thirdparty/PartPacker..."
#cd ../../thirdparty/PartPacker
#echo "[INFO] Installing PartPacker requirements..."
#pip install meshiki fpsample kiui pymcubes einops
#if [ $? -eq 0 ]; then
#    echo "[SUCCESS] PartPacker requirements installed"
#else
#    echo "[ERROR] Failed to install PartPacker requirements"
#    exit 1
#fi
#### part packer end ###
#echo "[SUCCESS] PartPacker installation completed"
#
#cd ../../

echo ""
echo "========================================"
echo "Installing Project Dependencies"
echo "========================================"
### for this project (fastapi / uvicorn relevant etc.)  ###
echo "[INFO] Installing main project requirements..."
#pip install -r requirements.txt
#
#if [ $? -eq 0 ]; then
#    echo "[SUCCESS] Main project requirements installed"
#else
#    echo "[ERROR] Failed to install main project requirements"
#    exit 1
#fi
#
#echo "[INFO] Installing test requirements..."
# testing 
#pip install -r requirements-test.txt
#if [ $? -eq 0 ]; then
#    echo "[SUCCESS] Test requirements installed"
#else
#    echo "[ERROR] Failed to install test requirements"
#    exit 1
#fi


echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo "All installation done successfully!"

echo "Checking CUDA availability..."
python -c "import torch; print(torch.cuda.is_available())" && echo "CUDA installed successfully" || echo "Failed"

echo "Checking PyTorch version..."
python -c "import torch; print(torch.__version__)" && echo "PyTorch installed successfully" || echo "Failed"

echo "Checking Blender availability..."
python -c "import bpy" && echo "Blender installed successfully" || echo "Failed"

echo "Checking Other Packages..."
python -c "import kaolin; print(kaolin.__version__)" && echo "Kaolin installed successfully" || echo "Failed"
python -c "import open3d; import pymeshlab" && echo "Open3D and pymeshlab installed successfully" || echo "Failed"

# install other runtime dependencies
sudo apt update
sudo apt install libsm6 libegl1 libegl1-mesa libgl1-mesa-dev -y # for rendering
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev

