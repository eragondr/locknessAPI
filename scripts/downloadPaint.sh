#!/bin/bash

#!/bin/bash

cd ..
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21 --include hunyuan3d-paintpbr-v2-1/*
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21 --include hy3dpaint/*
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21 --include hunyuan3d-vae-v2-1/*
huggingface-cli download facebook/dinov2-giant  --local-dir pretrained/tencent/Hunyuan3D21/dinov2-giant

realesrgan_path="pretrained/tencent/Hunyuan3D21/misc/RealESRGAN_x4plus.pth"
if [ "$FORCE_DOWNLOAD" = false ] && verify_file "$realesrgan_path" 50000000; then # 50MB minimum
        print_info "RealESRGAN_x4plus already exists and verified"
    else
        mkdir -p pretrained/tencent/Hunyuan3D21/misc
        download_with_verify \
            wget -O "pretrained/tencent/Hunyuan3D21/misc/RealESRGAN_x4plus.pth" "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    fi


