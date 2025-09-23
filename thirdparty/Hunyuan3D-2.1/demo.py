
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from PIL import Image
from hy3dshape.hy3dshape.rembg import BackgroundRemover
from hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

import logging
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

image_path = 'MaterialMVP/test_examples/image.png'
# image = Image.open(image_path).convert("RGBA")
# if image.mode == 'RGB':
#     rembg = BackgroundRemover()
#     image = rembg(image)
# # shape
# model_path = 'pretrained/tencent/Hunyuan3D-2.1'
# pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
# pipeline_shapegen.enable_flashvdm()
# print(f"image size: {image.size}")
# mesh = pipeline_shapegen(image=image)[0]
# print(f"mesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
# mesh.export('demowwwwww.glb')

#paint
max_num_view = 6  # can be 6 to 9
resolution = 768   # can be 768 or 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
conf.realesrgan_ckpt_path = "thirdparty/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "thirdparty/Hunyuan3D-2.1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "thirdparty/Hunyuan3D-2.1/hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)

output_mesh_path = 'aaaademo.glb'
from datetime import datetime
start_time = datetime.now()
logging.info(f"Start painting {start_time}")
print(f"Start painting {start_time}")
output_mesh_path = paint_pipeline(
    mesh_path = "MaterialMVP/test_examples/mesh.glb", 
    image_path = image_path,
    output_mesh_path = output_mesh_path
)
print(f"end painting {(datetime.now()-start_time).total_seconds()}")
