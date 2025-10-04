#!/bin/bash

cd ..
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21 --include hunyuan3d-paintpbr-v2-1/*
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21 --include hy3dpaint/*
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21 --include hunyuan3d-vae-v2-1/*
huggingface-cli download facebook/dinov2-giant  --local-dir pretrained/tencent/Hunyuan3D21/dinov2-giant
wget -O "pretrained/tencent/Hunyuan3D21/misc" "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

