<h1 align="center">Object Detection with YOLOv8</h1>

<p align="center">
  <img src="https://github.com/mrkrisgee/object_detection_yolov8/blob/main/gifs/yolov8l_car_counter.gif" alt="YOLOv8 Vehicle Detection">
</p>
<p align="center">
  <img src="https://github.com/mrkrisgee/object_detection_yolov8/blob/main/gifs/yolov8l_people_counter.gif" alt="YOLOv8 Vehicle Detection">
</p>


## Overview

This repository contains code for object detection using the YOLOv8 model. The YOLO (You Only Look Once) model is a state-of-the-art, real-time object detection system.

## Usage

### Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/) installed on your system. Anaconda simplifies package management and deployment.

### Create a Virtual Environment

Create and activate a new conda environment by running the following commands in your terminal:

```
conda create -n yolov8
conda activate yolov8
```

### Clone the repository

Clone this repository to your local machine and navigate into the project directory:

```
git clone https://github.com/mrkrisgee/object_detection_yolov8.git
cd object_detection_yolov8
```

### Install Necessary Packages

Install the required Python packages using pip:

```
pip install -r requirements.txt
```

### Download CUDA Toolkit

If you have an NVIDIA GPU and want to utilize CUDA for acceleration, download and install the CUDA toolkit from the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) page.

```
https://developer.nvidia.com/cuda-downloads
```

### Run the Scripts

To run the vehicle detection script, execute the following command:

```
python car_counter.py
```
or
```
python people_counter.py
```

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): YOLOv8 is a real-time object detection model developed by Ultralytics.
- [Alex Bewley](https://github.com/abewley/sort): For providing the SORT (Simple Online and Realtime Tracking) algorithm used for object tracking.
- [Murtaza Hassan](https://github.com/murtazahassan): For his comprehensive Object Detection 101 course
