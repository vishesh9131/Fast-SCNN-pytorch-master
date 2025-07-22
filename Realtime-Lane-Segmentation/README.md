# Realtime Lane Segmentation with Semantic Navigation

A comprehensive semantic segmentation and navigation framework supporting **Fast-SCNN** and **BiSeNet V2** models with real-time lane detection, object detection, depth estimation, and intelligent robot navigation.

<p align="center"><img width="100%" src="./png/Fast-SCNN.png" /></p>

## 🚀 New Features

### 🔄 **Model Switching Support**
Now supports switching between **Fast-SCNN** and **BiSeNet V2** models:
- **Fast-SCNN**: Lightweight, optimized for speed (original)
- **BiSeNet V2**: Higher accuracy, better semantic understanding

### 🤖 **Advanced Navigation**
- Real-time semantic navigation with depth perception
- Robot-aware pathfinding considering physical dimensions
- Three-zone navigation decision making (Left/Center/Right)
- MiDaS depth estimation for smarter obstacle avoidance

### 📹 **Multi-Input Support**
- Webcam real-time processing
- Video file processing
- Single image inference
- ROS1 integration for robotics

## Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#model-selection'>Model Selection</a>
- <a href='#usage'>Usage</a>
- <a href='#ros-integration'>ROS Integration</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-fast-scnn'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#demo'>Demo</a>
- <a href='#results'>Results</a>
- <a href='#references'>Reference</a>

## Installation
1. **Python Environment**
   ```bash
   conda create -n lane_seg python=3.8
   conda activate lane_seg
   ```

2. **Install Dependencies**
   ```bash
   pip install torch torchvision opencv-python Pillow numpy
   ```

3. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Realtime-Lane-Segmentation
   ```

4. **Download Model Weights**
   - Place Fast-SCNN weights: `weights/fast_scnn_citys.pth`
   - Place BiSeNet V2 weights: `weights/BiSeNet_v2_coco.pth`

## Model Selection

### Available Models
- **`fast-scnn`**: Fast Semantic Segmentation Network (default)
- **`bisenet-v2`**: BiSeNet V2 for better accuracy

### Available Datasets
- **`citys`**: Cityscapes dataset (19 classes)
- **`coco`**: COCO dataset (182 classes)

### Testing Models
```bash
python test_models.py
```

### TensorRT Optimization (Optional)
For **massive speed improvements** on NVIDIA GPUs:
```bash
# Install TensorRT (requires CUDA)
pip install tensorrt pycuda

# Test TensorRT optimization and benchmarks
python test_tensorrt.py
```

## Usage

### Basic Semantic Segmentation
```bash
# Fast-SCNN with Cityscapes
python demo.py --model fast-scnn --dataset citys --show-segmentation

# BiSeNet V2 with COCO
python demo.py --model bisenet-v2 --dataset coco --show-segmentation
```

### TensorRT Accelerated Inference
```bash
# Fast-SCNN with TensorRT optimization (requires CUDA + TensorRT)
python demo.py --model fast-scnn --dataset citys --fast_load --show-segmentation

# BiSeNet V2 with TensorRT optimization
python demo.py --model bisenet-v2 --dataset coco --fast_load --navigate --detect-objects --use-depth
```

### Real-time Navigation
```bash
# Complete navigation with webcam
python demo.py --model bisenet-v2 --dataset coco \
               --navigate --detect-objects --use-depth \
               --show-segmentation --show-depth
```

### Video Processing
```bash
# Process video file
python demo.py --video-path input.mp4 --output-path output.avi \
               --model fast-scnn --navigate --show-segmentation
```

### All Arguments
```bash
python demo.py --help
```

## ROS Integration

### 1. Setup ROS Package
```bash
# Copy semantic_navigator_ros to your catkin workspace
cp -r semantic_navigator_ros ~/catkin_ws/src/
cd ~/catkin_ws && catkin_make
source devel/setup.bash
```

### 2. Configure Model in Launch File
Edit `semantic_navigator_ros/launch/navigate.launch`:
```xml
<!-- Model Selection -->
<param name="model" value="bisenet-v2"/>     <!-- fast-scnn or bisenet-v2 -->
<param name="dataset" value="coco"/>         <!-- citys or coco -->

<!-- TensorRT Acceleration (requires CUDA + TensorRT) -->
<param name="fast_load" value="true"/>      <!-- Enable TensorRT optimization -->
```

### 3. Launch Navigation
```bash
roslaunch semantic_navigator_ros navigate.launch
```

### 4. Remap Camera Topic (if needed)
```xml
<remap from="/usb_cam/image_raw" to="/your_camera/image_raw"/>
```

## Model Comparison

| Model | Parameters | Memory | PyTorch FPS | TensorRT FPS | Accuracy | Best Use Case |
|-------|------------|--------|-------------|--------------|----------|---------------|
| Fast-SCNN | 1.16M | 4.43 MB | ~15-25 | ~40-60* | Good | Real-time applications, embedded systems |
| BiSeNet V2 | 3.54M | 13.52 MB | ~8-15 | ~25-45* | Better | High-accuracy requirements, research |

*TensorRT performance depends on GPU. Results shown for typical modern NVIDIA GPUs.

### 🚀 **TensorRT Benefits**
- **2-4x faster inference** on NVIDIA GPUs
- **FP16 precision** for memory efficiency  
- **Automatic optimization** with `--fast_load`
- **Seamless fallback** to PyTorch if TensorRT unavailable

## Datasets
- You can download [cityscapes](https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/downloads/). Note: please download [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4) and [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).

## Training-Fast-SCNN
- By default, we assume you have downloaded the cityscapes dataset in the `./datasets/citys` dir.