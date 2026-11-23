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
#import pygetwindow as gw
#import mss


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

window_name = bool(config['realtime']['window_name'])
capture_window = bool(config['realtime']['capture_window'])

model_albedo_shading = load_models('v2', device = config['model_settings']['albedo_shading_estimation_device'])



model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transformmds = midas_transforms.dpt_transform
else:
    transformmds = midas_transforms.small_transform
depth_device = torch.device(config['model_settings']['depth_estimation_device']) if torch.cuda.is_available() else torch.device("cpu")
midas.to(depth_device)
midas.eval()


model_normals = torch.hub.load('alexsax/omnidata_models', 'surface_normal_dpt_hybrid_384')
model_normals.eval()
normals_device = torch.device(config['model_settings']['normal_estimation_device'] if torch.cuda.is_available() else "cpu")
model_normals.to(normals_device)

# === Preprocessing ===
transform_normal = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


segformer_device = config['model_settings']['segformer_device'] if torch.cuda.is_available() else 'cpu'
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(segformer_device).eval()
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

trainId_to_id = {
     0: 7,    1: 8,    2: 11,   3: 12,   4: 13,
     5: 17,   6: 19,   7: 20,   8: 21,   9: 22,
    10: 23,  11: 24,  12: 25,  13: 26,  14: 27,
    15: 28,  16: 31,  17: 32,  18: 33,
}
lut = np.full(256, 255, dtype=np.uint8)
for trainId, cityscapes_id in trainId_to_id.items():
    lut[trainId] = cityscapes_id
    
    
if mode == 'infer':
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
    """options = onnxruntime.SessionOptions()
    options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL

    options.add_session_config_entry("tensorrt_engine_cache_path", "./engine_cache")
    options.add_session_config_entry("tensorrt_engine_decryption_enable", "1")

    options.add_session_config_entry("tensorrt_engine_cache_enable", "1")

    session = onnxruntime.InferenceSession(
        onnx_model,
        options,
        providers=[ 
             ("TensorrtExecutionProvider", {
               "trt_fp16_enable": True,
             }),
            "CUDAExecutionProvider"  # Fallback to CUDA if TensorRT fails
        ]
    )"""

    io_binding = session.io_binding()

    
snapshot_path = 'iter_10000.pth'
net = build_model('resnext101').cuda(config['model_settings']['shadows_estimation_device'])
net.load_state_dict(torch.load(snapshot_path))
print("init weight from {}".format(snapshot_path))
net.eval()


def get_albedo_shading(image_path):
    
    global model_albedo_shading, config

    image = load_image(image_path)  # Returns float32 array in [0, 1]

    results = run_pipeline(model_albedo_shading, image, device = config['model_settings']['albedo_shading_estimation_device'])

    albedo = results['hr_alb']
    albedo_uint8 = (np.clip(albedo, 0, 1) * 255).astype(np.uint8)
    albedo_img = Image.fromarray(albedo_uint8)

    diffuse_shading = results['dif_shd']
    shading_uint8 = (np.clip(diffuse_shading, 0, 1) * 255).astype(np.uint8)
    shading_img = Image.fromarray(shading_uint8)
    
    albedo_img = np.array(albedo_img)
    shading_img = np.array(shading_img)
    
    albedo_img = cv2.resize(albedo_img, (target_width, target_height))
    shading_img = cv2.resize(shading_img, (target_width, target_height))
    
    albedo = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
    shading = cv2.cvtColor(shading_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./tmp/shading.png", shading)
    cv2.imwrite("./tmp/albedo.png", albedo)
    
    #return albedo_img, shading_img


def depth_estimation(image_path):
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	input_batch = transformmds(img).to(depth_device)

	with torch.no_grad():
	    prediction = midas(input_batch)

	    prediction = torch.nn.functional.interpolate(
		prediction.unsqueeze(1),
		size=img.shape[:2],
		mode="bicubic",
		align_corners=False,
	    ).squeeze()

	output = prediction.cpu().numpy()


	depth_normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
	depth_uint8 = depth_normalized.astype(np.uint8)
	
	
	depth_uint8 = depth_uint8[..., np.newaxis]
	cv2.imwrite("./tmp/depth.png", depth_uint8)


def normal_estimation(image_path):
    global transform_normal

    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)

    input_tensor = transform_normal(img).unsqueeze(0).to(normals_device)

    with torch.no_grad():
        output = model_normals(input_tensor)
    output_resized = F.interpolate(output, size=(img.height, img.width), mode='bilinear', align_corners=True)
    normals = output_resized[0]
    norm = torch.norm(normals, dim=0, keepdim=True) + 1e-8
    normals_unit = normals / norm

    normals_vis = (normals_unit + 1) / 2  # [3, H, W]

    out_path = './tmp/normals.png'
    save_image(normals_vis, out_path)


def create_dataset_dirs(base_path):
    os.makedirs(base_path, exist_ok=True)

    subdirs = ["GBuffers", "Images", "Labels"]

    for subdir in subdirs:
        full_path = os.path.join(base_path, subdir)
        os.makedirs(full_path, exist_ok=True)
        print(f"Directory created or already exists: {full_path}")


def img_resizer(img_dir):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith(valid_exts):
            continue

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        save_path = os.path.join(img_dir, img_name)
        cv2.imwrite(save_path, resized_img)

def labels_resizer(labels_dir):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for img_name in os.listdir(labels_dir):
        if not img_name.lower().endswith(valid_exts):
            continue

        img_path = os.path.join(labels_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        save_path = os.path.join(labels_dir, img_name)
        cv2.imwrite(save_path, resized_img)


def generate_label_masks(label_path, img_path, predict_labels):
    if not predict_labels:
        labels = cv2.imread(label_path)
        return labeler.material_from_gt_label(labels), labels
    else:
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)

        # Preprocess
        inputs = feature_extractor(images=image, return_tensors="pt").to(segformer_device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits  # (1, num_classes=19, H, W)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize prediction back to original image size
        pred_resized = Image.fromarray(preds, mode='L')
        pred_resized = pred_resized.resize(original_size, resample=Image.NEAREST)
        pred_resized_np = np.array(pred_resized)

        # Map to Cityscapes ID
        remapped = lut[pred_resized_np]

        return labeler.material_from_gt_label(remapped), remapped


def inverse_rendering(image_path):
   global config
   print("Running inverse rendering...")
   
   get_albedo_shading(image_path)
   depth_estimation(image_path)
   normal_estimation(image_path)
   
   
   _, shadows = test_all_case(net, [(image_path,image_path)], num_classes=1, save_result=False, test_save_path="./", trans_scale=416, device_id=config['model_settings']['shadows_estimation_device'])
   shadows = np.stack((shadows,)*3, axis = -1)
   shadows = cv2.cvtColor(shadows, cv2.COLOR_BGR2RGB)
   cv2.imwrite("./tmp/shadows.png", shadows)
   
   

   albedo = cv2.imread("./tmp/albedo.png")
   normals = cv2.imread("./tmp/normals.png")
   depth = cv2.imread("./tmp/depth.png")   
   shading = cv2.imread("./tmp/shading.png") 
   shadows = cv2.imread("./tmp/shadows.png") 
   depth = np.expand_dims(depth[:,:,0], axis=-1)
   
   img = cv2.imread(image_path)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
   
   stacked = np.concatenate([img, albedo, normals, depth, shadows, shading], axis=-1)
   
   return stacked
   
   
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

    return translated_frame
   
def main(images_dir="./images", labels_dir="./labels", predict_labels=True, processed_dataset_path="./EPE_Dataset"):
    img_resizer(images_dir)
    labels_resizer(labels_dir)

    for filename in tqdm(os.listdir(images_dir)):
        if not filename.lower().endswith(('.png')):
            print("Transform all the input images in PNG format...")
            exit(1)
        
        g_buffers = inverse_rendering(os.path.join(images_dir, filename))
        label_masks, label_map = generate_label_masks(os.path.join(labels_dir, filename), os.path.join(images_dir, filename) , predict_labels)
        image = Image.open(os.path.join(images_dir, filename)).convert('RGB')
        
        if mode == 'infer':
            test(io_binding, image, g_buffers, label_masks, filename, "./out/")
        else:
            gbuffer_path = os.path.join(os.path.abspath(processed_dataset_path), "GBuffers", os.path.splitext(os.path.basename(filename))[0])
            image_path = os.path.join(os.path.abspath(processed_dataset_path), "Images", filename)
            labels_path = os.path.join(os.path.abspath(processed_dataset_path), "Labels", filename)
            robust_path = os.path.join(os.path.abspath(processed_dataset_path), "Robust", filename)
            
            np.savez_compressed(gbuffer_path, data=g_buffers)
            cv2.imwrite(labels_path,label_map)
            image.save(image_path)
            
            
            with open(os.path.join(os.path.abspath(processed_dataset_path), 'data.txt'), 'a') as f:
                   f.write(image_path+","+ robust_path+","+ gbuffer_path+".npz,"+ labels_path+  '\n')
            
            with open(os.path.join(os.path.abspath(processed_dataset_path), 'data.csv'), mode='a', newline='') as csvfile:
                   writer = csv.writer(csvfile)
                   writer.writerow([image_path])
            
            print("Robust images predicted by MSEG are required https://github.com/mseg-dataset/mseg-semantic. If you want to test/inference with the original EPE code then manually create a Robust directory and place inside the RGB images from Images directory to avoid running MSEG.")

if not capture_window:
    if mode == 'process':
       create_dataset_dirs(processed_dataset_path)
    main(images_path, labels_path, predict_labels, processed_dataset_path)


window_name = "Grand Theft Auto V"
if capture_window:
    try:
        game_window = gw.getWindowsWithTitle(window_name)[0]
    except IndexError:
        raise Exception("Window not found. Check the title.")

    if not game_window.isMinimized:
        # Bring the window to front
        game_window.activate()
        time.sleep(0.5)  # Allow time for the window to come to the foreground

    # Get window bounding box
    left, top, width, height = game_window.left, game_window.top, game_window.width, game_window.height

    # Start capturing the screen
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}

        while True:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)

            # Convert BGRA to BGR
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, (957, 526))
            cv2.imwrite("temp.png",frame)

            g_buffers = inverse_rendering("temp.png")
            label_masks, label_map = generate_label_masks("temp.png",
                                                          "temp.png", True)
            image = Image.open("temp.png").convert('RGB')

            translated_frame = test(io_binding, image, g_buffers, label_masks, "temp.png", "./out/")


            cv2.imshow("Game Capture", translated_frame)
            if cv2.waitKey(1) == 27:  # ESC to exit
                break
    
    cv2.destroyAllWindows()









    

