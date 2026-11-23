# RoboSAPIENS-Day2Night: INTO-CPS Case Study for Day-to-Night Image Translation

RoboSAPIENS-Day2Night presents an open-source tool for **unpaired day-to-night image-to-image translation using inverse rendering**, based on the work *Enhancing Unpaired Day-to-Night Image-to-Image Translation via Inverse Rendering* by Stefanos Pasios and Nikos Nikolaidis (EUSIPCO 2026). It generates additional scene representations from daytime RGB images and uses them to guide translation toward nighttime appearance while better preserving scene semantics.

---

## Relation to INTO-CPS

This repository is part of the INTO-CPS ecosystem and is associated with the **RoboSAPIENS** project (*Robotic Safe Adaptation In Unprecedented Situations*).

It supports research on dependable perception under changing environmental conditions by:

- generating nighttime-like images from daytime inputs for data augmentation and perception assessment;
- using scene geometry, material, and illumination cues to reduce visual artifacts and unintended semantic changes;
- enabling evaluation of perception systems under changes in illumination without requiring corresponding paired day/night images.

---

## Method Overview

The method adapts graphics-guided unpaired image translation to real-world daytime imagery. Rather than translating the RGB image in isolation, an inverse-rendering pipeline estimates several complementary representations:

1. **Albedo and shading**, describing appearance and illumination;
2. **Depth** and **surface normals**, providing scene geometry;
3. **Shadows**, representing additional illumination structure;
4. **Semantic labels**, either supplied by the user or estimated using a pretrained SegFormer network.

The RGB image, derived representations (G-buffers), and semantic information are used by the image translation network to produce a nighttime-like image. The repository also supports exporting these intermediate representations for training the [Enhancing Photorealism Enhancement (EPE)](https://github.com/isl-org/PhotorealismEnhancement) framework.

The accompanying paper reports improved visual similarity and semantic consistency relative to the evaluated unpaired translation baselines, measured using CMMD and a semantic segmentation experiment.

---

## Repository Structure

```text
robosapiens-Day2Night-Translation/
├── EPE_Modified/          # modified photorealism-enhancement components
├── Intrinsic/             # intrinsic image decomposition module
├── configurations/
│   └── config.yaml       # processing, inference and dataset configuration
├── images/               # input images (PNG)
├── labels/               # optional semantic label maps
├── materials/            # paper and additional resources
├── models/               # location for ONNX translation models
├── networks/             # neural network architectures
├── results/              # generated translated images
├── tmp/                  # intermediate inverse-rendering results
├── utils/                # supporting utilities
├── tool.py               # primary inverse-rendering and translation tool
├── infer.py              # inference using previously exported G-buffers
├── labeler.py            # semantic-label to material-label mapping
├── video_splitter.py     # extract input video frames
├── create_video.py       # combine translated frames into a video
├── test_MT_util.py       # shadow-estimation utilities
├── name_append.py        # auxiliary file-naming utility
├── LICENSE
├── NOTICE
├── CITATION.cff
├── CONTRIBUTING.md
└── README.md
```

*Pretrained weights are downloaded separately; they are not included in this repository.*

---

## Requirements

- **Python 3.8**, as specified by the original project environment.
- **PyTorch 2.4.1**, **torchvision 0.19.1**, and a compatible CUDA installation (the provided installation example uses CUDA 12.1).
- An **NVIDIA CUDA GPU** for the current ONNX inference and shadow-estimation paths.
- [Conda](https://www.anaconda.com/) (recommended), plus the Python dependencies listed below.
- Internet access for installing dependencies and obtaining additional pretrained models.

The implementation was written for a CUDA-based workflow; a CPU-only installation is **not** supported by the current inference path without source changes.

---

## Installation

From the root directory of the repository, create the environment and install the dependencies following the original project's instructions:

```bash
conda create --name day2night python=3.8 -y
conda activate day2night
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

cd Intrinsic
pip install .
cd ..

pip install --force-reinstall cython==0.29.36
pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git
pip install transformers==4.46.3 onnxruntime-gpu==1.19.2
pip install pygetwindow mss opencv-python
conda install -y -c conda-forge openssl=3.6.2
```

The `pydensecrf` source installation may require a C/C++ build toolchain (Microsoft C++ Build Tools on Windows). If an older environment cannot resolve the specified OpenSSL version, use a compatible version supported by the installed Conda channels.

### Pretrained Models

Download the pretrained weights from the [original project's model folder on Google Drive](https://drive.google.com/drive/folders/1cTppksDlnIXGAuThhAzBy3xDOp9ezb3p?usp=sharing) and place them as follows:

| File | Destination |
|---|---|
| `cs2dark.onnx` | `models/` |
| `pfd2vistas.onnx` | `models/` |
| `iter_10000.pth` | repository root |
| `resnext_101_32x4d.pth` | repository root |

Further weights may be downloaded automatically by the component libraries on first use. Keep the filenames and paths expected by the source code unchanged.

---

## Datasets

The tool accepts daytime images placed in the `images/` directory. The current main pipeline (`tool.py`) expects **PNG files**. When using existing semantic annotations, place the corresponding label maps in `labels/` with matching filenames and extensions.

The configuration file is **`configurations/config.yaml`** (plural `configurations`). Its key options are:

| Configuration field | Purpose |
|---|---|
| `tool_settings.mode` | `process` to prepare inverse-rendered training data, or `infer` to translate images |
| `dataset_settings.images_path` | input image directory |
| `dataset_settings.labels_path` | semantic label directory |
| `dataset_settings.predict_labels` | `True` to predict labels using SegFormer, `False` to read provided labels |
| `dataset_settings.processed_dataset_path` | output directory for exported training representations |
| `results_settings.translated_images_path` | translated-image output directory |
| `model_settings.translation_model` | path to the ONNX image-translation model |
| `video_settings.*` | video input, frame rate, and output path |
| `realtime.capture_window` | enable experimental real-time window capture |

The label-to-material conversion is implemented in `labeler.py`; labels from other datasets may require adapting `material_from_gt_label`.

---

## Running the Tool

Run the tool from the **repository root** so that its relative paths resolve correctly:

```bash
python tool.py
```

### Preparing Training Data

1. In `configurations/config.yaml`, set `tool_settings.mode: process`.
2. Set `dataset_settings.predict_labels: False` if matching ground-truth labels are available; otherwise set it to `True` to use the pretrained semantic segmentation model.
3. Place PNG images in `images/` and, if needed, their matching label maps in `labels/`.
4. Run `python tool.py`.

The tool creates `GBuffers/`, `Images/`, and `Labels/` inside `dataset_settings.processed_dataset_path`, together with dataset index files. These outputs can be used to train EPE by following the [EPE training instructions](https://github.com/isl-org/PhotorealismEnhancement/tree/main/code). **`tool.py` prepares data; it does not itself train the translation model.**

### Day-to-Night Inference

1. Set `tool_settings.mode: infer` in `configurations/config.yaml`.
2. Put input PNG files in `images/`.
3. Either supply matching semantic maps in `labels/` and set `predict_labels: False`, or set `predict_labels: True` to generate them with SegFormer.
4. Ensure `model_settings.translation_model` points to an installed ONNX checkpoint (for example, `models/cs2dark.onnx`).
5. Run:

```bash
python tool.py
```

Translated images are saved in the directory defined by `results_settings.translated_images_path` (default: `results/`). The separate `infer.py` script can perform translation using G-buffers and labels previously generated in the processed dataset.

### Video Processing

Set `video_settings.video_path` in the configuration and extract video frames:

```bash
python video_splitter.py
```

After processing the extracted frames using `python tool.py`, set `video_settings.video_fps` and `video_settings.video_export_path`, then recombine the translated images:

```bash
python create_video.py
```

### Synthetic Data / Real-Time Capture

The original project additionally includes a model that translates Grand Theft Auto V imagery toward the Mapillary Vistas appearance domain. For the experimental capture workflow, configure `realtime.capture_window` and the game window name in `configurations/config.yaml`. This path depends on GUI capture functionality and may require adapting the corresponding imports and platform-specific code in `tool.py`.

---

## Reproducibility

- The Python and major framework versions above follow the original project's installation instructions.
- Inference requires the pretrained weights available through the separately hosted download; they are **not** redistributed here.
- The configuration specifies input image dimensions (957 × 526 by default) and model/device settings that affect processing.
- The original paper evaluates translated image quality with **CLIP Maximum Mean Discrepancy (CMMD)** and examines downstream semantic segmentation consistency. Exact replication also depends on the original data and pretrained checkpoints.

---

## Known Limitations

- The supplied inference code relies on CUDA tensors and CUDA ONNX Runtime I/O binding; selecting the CPU provider alone is insufficient for CPU-only execution.
- The main preprocessing code resizes images and label files **in place** according to `image_settings`; keep copies of your original inputs.
- The main tool processes PNG files; convert other image formats to PNG before running.
- Existing semantic label conventions may not match the implemented material mapping and can require changes in `labeler.py`.
- The real-time window-capture section uses platform-specific GUI utilities and has commented-out imports in the current source; it may require code adjustments.
- Dependencies and pretrained weights from external projects have their own availability and usage conditions.

---

## License

This project retains the original **MIT License**. See [LICENSE](LICENSE) for the full license text and the original copyright notice (© 2026 Stefanos Pasios).

The project incorporates or relies upon components from third-party repositories, which may have **separate licensing terms**. See [NOTICE](NOTICE) and consult the corresponding upstream repositories before redistribution. Pretrained model weights and external datasets are subject to their own terms.

---

## Acknowledgment

This tool is presented as part of the **RoboSAPIENS** project (*Robotic Safe Adaptation In Unprecedented Situations*), funded by the European Union's Horizon Europe research and innovation programme under grant agreement **No. 101133807**. The views and opinions expressed are those of the authors and do not necessarily reflect those of the European Union or the European Commission.

---

## Citation

If you use this tool or its underlying method in your research, please cite the original publication:

```bibtex
@INPROCEEDINGS{11226280,
  author    = {Pasios, Stefanos and Nikolaidis, Nikos},
  booktitle = {2026 34th European Signal Processing Conference (EUSIPCO)},
  title     = {Enhancing Unpaired Day-to-Night Image-to-Image Translation via Inverse Rendering},
  year      = {2026},
  pages     = {616--620}
}
```

The [paper is also available in the EUSIPCO 2026 proceedings](https://eurasip.org/Proceedings/Eusipco/Eusipco2026/pdfs/0000616.pdf).
