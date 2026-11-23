import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from chrislib.data_util import load_image
from intrinsic.pipeline import load_models, run_pipeline
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from test_MT_util import test_all_case
# from networks.EGNet import build_model
from networks.MTMT import build_model
import labeler
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import onnxruntime
import time
import yaml
from torchvision.utils import save_image
import os
import onnxruntime
import csv


with open('./configurations/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

mode = config['tool_settings']['mode']
results_path = config['results_settings']['translated_images_path']
onnx_model = config['model_settings']['translation_model']
processed_dataset_path = config['dataset_settings']['processed_dataset_path']

os.makedirs(results_path, exist_ok=True)
os.makedirs("./tmp/", exist_ok=True)


onnx_model = config['model_settings']['translation_model']
target_width = int(config['image_settings']['width'])
target_height = int(config['image_settings']['height'])

images_path = config['dataset_settings']['images_path']
labels_path = config['dataset_settings']['labels_path']
predict_labels = bool(config['dataset_settings']['predict_labels'])



    
    

print("Clearing older ONNX profiles...")

file_list = os.listdir(os.getcwd())
for filename in file_list:
	if "onnxruntime_profile__" in filename:
	    file_path = os.path.join(os.getcwd(), filename)
	    os.remove(file_path)
	    print(f"Deleted: {file_path}")

opts = onnxruntime.SessionOptions()
opts.enable_profiling = True

session = onnxruntime.InferenceSession(
onnx_model,
sess_options=opts,
providers=[
    (
	"CUDAExecutionProvider",
	{
	    "device_id": 0,
	    "cudnn_conv_use_max_workspace": "1",
	    "cudnn_conv_algo_search": "DEFAULT",
	},
    ),
    "CPUExecutionProvider",
],
)

io_binding = session.io_binding()

   
   
def test(io_binding,img,gbuffers,label_map,file_name,out_path):
    global results_path

    gbuffers = np.transpose(gbuffers, (2, 0, 1))
    label_map = np.transpose(label_map, (2, 0, 1))


    img = np.expand_dims(img, axis=0)
    gbuffers = np.expand_dims(gbuffers, axis=0)
    label_map = np.expand_dims(label_map, axis=0)
    
    img = np.transpose(img, (0, 3, 1, 2)) / 255
    
    
    
    img = img.astype(np.float32)
    gbuffers = gbuffers.astype(np.float32)
    label_map = label_map.astype(np.float32)

    img = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
    gbuffers = onnxruntime.OrtValue.ortvalue_from_numpy(gbuffers, 'cuda', 0)
    label_map = onnxruntime.OrtValue.ortvalue_from_numpy(label_map, 'cuda', 0)
    

    tdtype = np.float32
    io_binding.bind_input(name='input', device_type=img.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=img.shape(), buffer_ptr=img.data_ptr())
    io_binding.bind_input(name='gbuffers', device_type=gbuffers.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=gbuffers.shape(), buffer_ptr=gbuffers.data_ptr())
    io_binding.bind_input(name='onnx::Gather_2', device_type=label_map.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=label_map.shape(), buffer_ptr=label_map.data_ptr())
    io_binding.bind_output('output', device_type='cuda', device_id=0, element_type=tdtype)

    infer_timer = time.time()
    session.run_with_iobinding(io_binding)
    print("Inference time: " + str(time.time() - infer_timer))
    new_img = torch.from_numpy(io_binding.copy_outputs_to_cpu()[0])
    translated_frame = (new_img[0, ...].clamp(min=0, max=1).permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(np.uint8)
    
    translated_frame = cv2.cvtColor(translated_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(results_path, file_name), translated_frame)

def generate_label_masks(label_path):
      labels = cv2.imread(label_path)
      return labeler.material_from_gt_label(labels), labels


def main(images_dir="./images"):

    for filename in tqdm(os.listdir(images_dir)):
        if not filename.lower().endswith(('.png')):
            print("Transform all the input images in PNG format...")
            exit(1)
        
        g_buffers = np.load(os.path.join(os.path.abspath(processed_dataset_path), "GBuffers", os.path.splitext(os.path.basename(filename))[0]+".npz"))['data']
        label_masks, label_map = generate_label_masks(os.path.join(os.path.abspath(processed_dataset_path), "Labels", filename))
        
        image = Image.open(os.path.join(images_dir, filename)).convert('RGB')
        
        test(io_binding, image, g_buffers, label_masks, filename, "./out/")


main(os.path.join(os.path.abspath(processed_dataset_path), "Images"))









    

