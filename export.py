import numpy as np
from tqdm import tqdm

import torch
from sgg_benchmark.config import cfg
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.structures.image_list import to_image_list

conf = "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/IndoorVG4/SGDET/penet-yolov8m/config.yml"


cfg.merge_from_file(conf)
cfg.MODEL.BACKBONE.NMS_THRESH = 0.001
cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 80
cfg.TEST.IMS_PER_BATCH = 1
# cfg.freeze()

cfg.TEST.CUSTUM_EVAL = True

model = build_detection_model(cfg)
checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
last_check = checkpointer.get_checkpoint_file()
if last_check == "":
    last_check = cfg.MODEL.WEIGHT
print("Loading last checkpoint from {}...".format(last_check))
_ = checkpointer.load(last_check)

model.to(cfg.MODEL.DEVICE)
model.roi_heads.eval()
model.backbone.eval()

# input img
img_path = "/home/maelic/Documents/DSC_0451.jpg"

# load as tensor
from PIL import Image
from torchvision import transforms
img = Image.open(img_path)
# resize to 640*640
img = img.resize((640, 640))
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)
# img = img.to(cfg.MODEL.DEVICE)
model.cpu()

# onnx_program = torch.onnx.dynamo_export(model, (input_img,None))
# onnx_program.save("my_image_classifier.onnx")

# Export the model
torch.onnx.export(model,               # model being run
                img,                         # model input (or a tuple for multiple inputs)
                "my_model.onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=17,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['boxes', 'rels'], # the model's output names
                )