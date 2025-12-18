# DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis

[![Pytorch](https://img.shields.io/badge/Implementation-PyTorch-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/ICCV2025-Paper-blue.svg)](https://openaccess.thecvf.com/content/ICCV2025/html/Cai_DeepShield_Fortifying_Deepfake_Video_Detection_with_Local_and_Global_Forgery_ICCV_2025_paper.html)
[![Arxiv](https://img.shields.io/badge/arXiv-2510.25237-B31B1B.svg)](https://arxiv.org/abs/2510.25237)

This is a PyTorch implementation of **"DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis"** accepted by **ICCV 2025**. 

---

## 🛠️ Environment Setup

Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt

```

---

## 📊 Data Preparation

### 1. Face Detection & Preprocessing

**Step 1: Download Datasets** Please download the following datasets and place them in the `./datasets/` directory:

* [FaceForensics++ (FF++)](https://github.com/ondyari/FaceForensics)
* [Celeb-DF (v2)](https://github.com/yuezunli/celeb-deepfakeforensics)
* [DFD](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
* [DFDCP](https://ai.facebook.com/datasets/dfdc/)
* [DFDC](https://ai.facebook.com/datasets/dfdc/)

**Step 2: Extract Facial Images** We use **RetinaFace** to extract facial images from videos. Please refer to the [RetinaFace GitHub repository](https://github.com/biubug6/Pytorch_Retinaface) for installation. Run the provided script to process the videos:

```bash
python ./preprocess/detect_faces.py

```

**Step 3: Prepare Data Text Files** Create data annotation files in `./datasets/FaceForensics++/data_txt` following this format:

`[path] [start_frame] [end_frame] [label]`

*Example (`train.txt` for FF++):*

```plaintext
original_sequences/youtube/c23/frames/071 0 452 0
original_sequences/youtube/c23/frames/054 0 367 0
manipulated_sequences/Deepfakes/c23/frames/071_054 0 452 1
manipulated_sequences/Face2Face/c23/frames/071_054 0 367 1
manipulated_sequences/FaceSwap/c23/frames/071_054 0 367 1
manipulated_sequences/NeuralTextures/c23/frames/071_054 0 367 1

```

### 2. Facial Landmark Detection

To analyze local forgery details, detect 81 facial landmarks using **dlib**:

1. Download `shape_predictor_81_face_landmarks.dat` and place it in the `./preprocess/` directory.
2. Run the landmark detection script:

```bash
python ./preprocess/detect_lands.py

```

### 3. Directory Structure

Ensure your data directory is structured as follows:

```text
datasets
└── FaceForensics++
    ├── manipulated_sequences
    │   └── Deepfakes
    │       └── c23
    │           └── frames
    │               └── 000_003
    │                   └── 0000.png  <-- Face frames
    └── original_sequences
        └── youtube
            └── c23
                ├── frames
                │   └── 000
                │       └── 0000.png
                └── landmarks
                    └── 000
                        └── 0000.npy   <-- Landmark files

```

---

## 🚀 Training

### Multi-GPU Training (DDP)

Run the following command for distributed training:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 16677 main.py \
 --dataset ffpp_combine --val_dataset ffpp --save_all_ckpt \
 --input-size 224 --num_clips 4 --output_dir [your_output_dir] --opt adam \
 --lr 0.0003 --momentum 0.9 --weight-decay 0.0005 --epochs 80 --sched cosine \
 --duration 12 --batch-size 2 --disable_scaleup --cutout True \
 --warmup-epochs 0 --model vitb_st \
 2>&1 | tee ./output/train_ffpp_`date +'%m_%d-%H_%M'`.log

```

### Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
 --dataset ffpp_combine --val_dataset ffpp --save_all_ckpt \
 --input-size 224 --num_clips 4 --output_dir [your_output_dir] --opt adam \
 --lr 0.0003 --momentum 0.9 --weight-decay 0.0005 --epochs 80 --sched cosine \
 --duration 12 --batch-size 2 --disable_scaleup --cutout True \
 --warmup-epochs 0 --model vitb_st \
 2>&1 | tee ./output/train_ffpp_`date +'%m_%d-%H_%M'`.log

```

---

## 🧪 Testing

To evaluate a trained model, use the following script:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
 --dataset cdfv2 --input-size 224 --num_clips 4 \
 --duration 12 --batch-size 4 --disable_scaleup --model vitb_st \
 --test_ckpt_dir [your_checkpoint_dir] \
 2>&1 | tee ./output/test_cdf_`date +'%m_%d-%H_%M'`.log

```

---

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{cai2025deepshield,
  title={DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis},
  author={Cai, Yinqi and Li, Jichang and Li, Zhaolun and Chen, Weikai and Lan, Rushi and Xie, Xi and Luo, Xiaonan and Li, Guanbin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={12524--12534},
  year={2025}
}

```

## Contact

Please feel free to contact the first author, namely [Li Jichang](https://lijichang.github.io/), with an Email address li.jichang@foxmail.com, if you have any questions.
