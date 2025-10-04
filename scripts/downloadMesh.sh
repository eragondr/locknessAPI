#!/bin/bash

huggingface-cli login --token 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
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21/hunyuan3d-dit-v2-1 --include huhunyuan3d-dit-v2-1/*
huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir pretrained/tencent/Hunyuan3D21/hunyuan3d-vae-v2-1 --include hunyuan3d-vae-v2-1/*
