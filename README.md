# YOLOv7 Lab 8 Training Guide

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yctsai0826/uav_Lab08.git
   cd uav_Lab08
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train your model!!!
   ```bash
   python train.py --device 0 --batch-size 8 --data data/data.yaml --img 1280 720 --cfg cfg/training/yolov7-tiny-lab8.yaml --weights 'yolov7-tiny.pt' --name yolov7-lab08 --hyp data/hyp.scratch.tiny.yaml --epoch 300
---

## Command Arguments

| Argument                 | Description                                                                                 | Example Value                |
|--------------------------|---------------------------------------------------------------------------------------------|------------------------------|
| `--device`               | Device to use for training (`0` for GPU, `cpu` for CPU, `mps` for Apple Silicon).          | `0`, `cpu`, `mps`           |
| `--batch-size`           | Number of samples per batch.                                                               | `8`                          |
| `--data`                 | Path to the dataset configuration YAML file.                                               | `data/data.yaml`             |
| `--img`                  | Image dimensions for training (width height).                                              | `1280 720`                   |
| `--cfg`                  | Path to the YOLOv7 model configuration file.                                               | `cfg/training/yolov7-tiny-lab8.yaml` |
| `--weights`              | Path to the pre-trained weights file (optional for transfer learning).                     | `'yolov7-tiny.pt'`           |
| `--name`                 | Name of the training run, used for saving checkpoints and logs.                            | `yolov7-lab08`               |
| `--hyp`                  | Path to the hyperparameter configuration YAML file.                                        | `data/hyp.scratch.tiny.yaml` |
| `--epoch`                | Number of training epochs.                                                                 | `300`                        |

---

## Results

After training, weights will be saved in the following directory:

```
.\runs\train\yolov7-lab08\weights
```

---
