# LFEPose

About

A project to identify sheep pose to allow camera streams to be analzyed with custom trained models built for detecting sickness and other problems based off of sheep pose.
## 🚀 Installation
Follow the [MMPose](https://github.com/open-mmlab/mmpose) installation tutorial provided by OpenMMLab.


## 📦 Data Preparation
The LambingSheep dataset is available on [ModelScope](https://www.modelscope.cn/datasets/CondorG/LambingSheep) or [HuggingFace](https://huggingface.co/datasets/CondorG/LambingSheep).

## 🛠️ Usage
1. Download LFEPose, place the folder under projects/, and rename it to sheep_pose.
```
mmpose
└── projects
    └── sheep_pose
        ├── config
        ├── core
        └── sheep_config
```
2. Download the dataset and update the database path in LFEPose-CSP-m.py.
3. Train and test.
```
python tools/train.py projects/sheep_pose/config/LFEPose-CSP-m.py
python tools/test.py projects/sheep_pose/config/LFEPose-CSP-m.py work_dirs/LFEPose-CSP-m/best_coco_AP_epoch_600.pth
```
