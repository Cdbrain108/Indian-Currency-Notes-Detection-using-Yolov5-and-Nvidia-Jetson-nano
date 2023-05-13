# Indian Currency Detection using YOLOv5 on NVIDIA Jetson Nano

Welcome to our Indian Currency Notes Detection project! This repository contains code and data for detecting Indian currency notes using the YOLOv5 object detection algorithm on NVIDIA Jetson Nano.

## Overview
Our project aims to develop a robust and accurate deep learning model for detecting Indian currency notes in real-world scenarios. We use the YOLOv5 algorithm, which is known for its high performance and efficiency in object detection tasks and NVIDIA Jetson Nano is a popular edge device that is capable of running YOLOv5 model efficiently.

The repository includes the following:

1. YOLOv5 model for Indian currency detection

2. Jetson Nano setup and installation instructions

3. Instructions for running the model on Jetson Nano


Our model can detect and classify the following denominations of Indian currency notes:


Rs. 10

Rs. 20

Rs. 50

Rs. 100

Rs. 200

Rs. 500

Rs. 2000


## Example Results

![alt text](https://drive.google.com/uc?export=view&id=1x7OV_pzeV_s19J0QXyCH12a7PQgJWJaW)
![alt text](https://drive.google.com/uc?export=view&id=1BesAZxmfPUghMSl8lEUrhynn2EpONCkR)
![alt text](https://drive.google.com/uc?export=view&id=1MJmg5SmYC9-0BoBXz_Od3QO3Ns3usqB_)
![alt text](https://drive.google.com/uc?export=view&id=1CbSsdEVVjjuIkOPdq0GauTgky0hHFpyZ)
![alt text](https://drive.google.com/uc?export=view&id=1ZluWiUPnAvtsCtVMbhDUhKz_8KRoqcfh)


## Prerequisites

![alt text](https://drive.google.com/uc?export=view&id=12-u3_KzRAskUJNoMKXwi__Tbba2_Peyp)

NVIDIA Jetson Nano


Micro-SD Card (minimum 32GB

Power Supply

USB keyboard, mouse and monitor

Internet connection

Python 3.6 or later

USB Camera for live detection

## Installation

1. Download and flash the JetPack SDK onto the micro-SD card using the Etcher tool. The JetPack SDK includes the necessary drivers, libraries, and tools to run YOLOv5 on Jetson Nano.

2. Connect the power supply, keyboard, mouse and monitor to the Jetson Nano.

3. Follow the on-screen instructions to complete the Jetson Nano setup.

4. Install the necessary Python packages and dependencies using the following commands.

```
sudo apt-get update
sudo apt-get install -y python3-pip
sudo apt-get install -y libopenblas-dev liblapack-dev libatlas-base-dev libgfortran5
sudo pip3 install numpy torch torchvision

```

5. Clone the YOLOv5 repository using the following command:

```
git clone https://github.com/ultralytics/yolov5.git
```
6. Install the required Python packages using the following command.

``` 
pip install -r Requirement.text
```

## Usage
Navigate to the YOLOv5 directory using the following command.
```
cd yolov5
```
Load the YOLOv5 model using their pretrained weights:
```
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/best.pt', force_reload=True)
```
Run the following command to start the realtime object detection on the Jetson Nano using USB Camera.
```
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break
cap.release()
cv2.destroyAllWindows()
```
This will launch the currency tag detection process and the output will be displayed on the screen.

You can also specify the input video and image file or camera device number using the following command.
```
python3 detect.py --source <input-file-or-device-number> --weights path/to/weights

```
This will detect Indian currency notes in the specified image or video using our trained model and save the output to a new file.

To stop the object detection process, press `Ctrl+C`


## Model
We used YOLOv5, a popular deep learning algorithm for object detection, to detect Indian currency notes in our dataset. YOLOv5 is fast, accurate, and easy to train, making it an ideal choice for our project.





## Dataset

[Indian currency Detection](https://app.roboflow.com/ds/5UvQII9ghg?key=R9CBQ4uvpM)

## Used script for training

```
!python train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --device 0
```
### Trained weights for Indian Currency Detection Model

- ```Best.pt``` : https://drive.google.com/file/d/1mqeg950HGSFW4YHKl0yvo_nUR0Me7sjO/view?usp=share_link
- ```Last.pt``` : https://drive.google.com/file/d/1uHeVJK19TqbZPkZaqBSOEjXJsUdPPOJg/view?usp=share_link

## Contributing
We welcome contributions to our project! If you find a bug or want to add a new feature, please submit a pull request. We also welcome suggestions for improving our model and dataset.


### Special Thanks to [Ultralytics](https://github.com/ultralytics).

---

<div align="center">
<p>
   <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>
</p>
<br>
<div>
   <a href="https://github.com/ultralytics/yolov5/actions"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
   <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
   <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
   <br>
   <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
   <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
   <a href="https://join.slack.com/t/ultralytics/shared_invite/zt-w29ei8bp-jczz7QYUmDtgo6r6KcMIAg"><img src="https://img.shields.io/badge/Slack-Join_Forum-blue.svg?logo=slack" alt="Join Forum"></a>
</div>

<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<div align="center">
   <a href="https://github.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.linkedin.com/company/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://twitter.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.producthunt.com/@glenn_jocher">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-producthunt.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://youtube.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.facebook.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.instagram.com/ultralytics/">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="2%"/>
   </a>
</div>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Inference</summary>

Inference with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
. [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>



<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are
1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://github.com/ultralytics/yolov5/issues/475) times faster). Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.

```bash
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️
  RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Roboflow for Datasets, Labeling, and Active Learning](https://github.com/ultralytics/yolov5/issues/4975)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TFLite, ONNX, CoreML, TensorRT Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)
</details>


## License
This project is licensed under the MIT License.
