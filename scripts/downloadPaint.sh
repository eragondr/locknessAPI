#!/bin/bash

huggingface-cli login --token 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
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21/hunyuan3d-paintpbr-v2-1 --include hunyuan3d-paintpbr-v2-1/*
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21/hy3dpaint --include hy3dpaint/*
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21/hy3dpaint --include hy3dpaint/*
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21/hunyuan3d-vae-v2-1 --include hunyuan3d-vae-v2-1/*
local dinov2_dir="pretrained/tencent/Hunyuan3D21/dinov2-giant"
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$dinov2_dir" 5; then
        print_info "DINOv2-giant model already exists and verified"
    else
        mkdir -p "$dinov2_dir"
        print_info "Downloading DINOv2-giant model..."
        if huggingface-cli download  facebook/dinov2-giant \
            --local-dir "$dinov2_dir" --exclude "*.bin"; then
            print_success "DINOv2-giant model downloaded successfully"
        else
            print_error "Failed to download DINOv2-giant model"
            return 1
        fi
    fi
local realesrgan_path="pretrained/tencent/Hunyuan3D21/misc/RealESRGAN_x4plus.pth"
if [ "$FORCE_DOWNLOAD" = false ] && verify_file "$realesrgan_path" 50000000; then # 50MB minimum
        print_info "RealESRGAN_x4plus already exists and verified"
    else
        mkdir -p pretrained/misc
        download_with_verify \
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
            "$realesrgan_path" \
            "RealESRGAN_x4plus model"
    fi


