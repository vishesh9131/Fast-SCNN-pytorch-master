# TensorRT Optimization Guide

This guide explains how to set up and use TensorRT optimization for **massive speed improvements** in your semantic segmentation models.

## 🚀 Performance Benefits

With TensorRT optimization, you can expect:
- **2-4x faster inference** on NVIDIA GPUs
- **Higher FPS** for real-time applications
- **FP16 precision** for memory efficiency
- **Automatic optimization** with zero code changes

## 📋 Prerequisites

### 1. Hardware Requirements
- **NVIDIA GPU** with CUDA Compute Capability 6.0+
- **CUDA Toolkit** 11.0 or later
- **At least 4GB GPU memory** (8GB+ recommended)

### 2. Software Requirements
- Python 3.7+
- PyTorch with CUDA support
- NVIDIA TensorRT 8.0+

## 🛠️ Installation

### Step 1: Install CUDA Toolkit
Download and install CUDA from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi
```

### Step 2: Install TensorRT
```bash
# Option 1: Using pip (recommended)
pip install tensorrt pycuda

# Option 2: Download from NVIDIA (more control)
# Go to https://developer.nvidia.com/tensorrt
# Download TensorRT for your CUDA version
```

### Step 3: Verify Installation
```bash
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

## 💡 Usage

### Command Line (demo.py)
```bash
# Basic usage with TensorRT acceleration
python demo.py --model fast-scnn --dataset citys --fast_load --show-segmentation

# Full navigation with TensorRT
python demo.py --model bisenet-v2 --dataset coco --fast_load --navigate --detect-objects --use-depth

# All available models with TensorRT
python demo.py --model fast-scnn --dataset citys --fast_load
python demo.py --model bisenet-v2 --dataset coco --fast_load
```

### ROS Integration
Edit `semantic_navigator_ros/launch/navigate.launch`:

```xml
<launch>
    <node name="semantic_navigator_node" pkg="semantic_navigator_ros" type="navigator_node.py" output="screen">
        <!-- Model Selection -->
        <param name="model" value="bisenet-v2"/>
        <param name="dataset" value="coco"/>
        
        <!-- Enable TensorRT Acceleration -->
        <param name="fast_load" value="true"/>
        
        <!-- Other parameters... -->
    </node>
</launch>
```

Then launch:
```bash
roslaunch semantic_navigator_ros navigate.launch
```

## 🧪 Performance Testing

### Benchmark Both Models
```bash
# Run comprehensive performance tests
python test_tensorrt.py
```

This will:
- Test both Fast-SCNN and BiSeNet V2
- Compare PyTorch vs TensorRT performance  
- Generate optimization engines
- Show detailed FPS comparisons

### Sample Output
```
🔸 FAST_SCNN_CITYS
   PyTorch: 23.4 FPS (42.7ms)
   TensorRT: 67.8 FPS (14.7ms)
   Speedup: 2.9x

🔸 BISENET_V2_COCO  
   PyTorch: 12.1 FPS (82.6ms)
   TensorRT: 34.2 FPS (29.2ms)
   Speedup: 2.8x
```

## 🔧 Technical Details

### How It Works
1. **First Run**: PyTorch model → ONNX → TensorRT engine (takes 2-5 minutes)
2. **Subsequent Runs**: Load pre-built TensorRT engine (instant)
3. **Automatic Fallback**: If TensorRT fails, uses standard PyTorch

### Engine Files
TensorRT engines are saved as:
- `weights/fast_scnn_citys_trt.engine`
- `weights/bisenet_v2_coco_trt.engine`

### Memory Usage
- **FP16 mode**: ~50% less GPU memory
- **Batch optimization**: Optimized for batch size = 1
- **Input shape**: Fixed at 480x640 (configurable)

## ❗ Troubleshooting

### Common Issues

**1. "TensorRT not available"**
```bash
pip install tensorrt pycuda
# or
pip install --upgrade tensorrt pycuda
```

**2. "CUDA not available"**
- Install CUDA Toolkit
- Verify with `nvidia-smi`
- Reinstall PyTorch with CUDA support

**3. "Engine building failed"**
- Check GPU memory (need 4GB+)
- Try with smaller batch size
- Verify model weights exist

**4. "ONNX export failed"**
- Update PyTorch: `pip install --upgrade torch torchvision`
- Check model compatibility

### Debug Mode
Add debug prints:
```bash
TENSORRT_LOGGER_LEVEL=VERBOSE python demo.py --fast_load
```

## 🎯 Best Practices

### For Maximum Performance
1. **Use CUDA**: TensorRT requires GPU
2. **Enable FP16**: Automatic with `--fast_load`
3. **Fixed input size**: 480x640 (default)
4. **Warm-up period**: First few frames may be slower

### For Development
1. **Test without TensorRT first**: Ensure model works
2. **Use `test_tensorrt.py`**: Verify optimization
3. **Monitor GPU memory**: Check with `nvidia-smi`

### For Production
1. **Pre-build engines**: Run once to create engines
2. **Deploy engines**: Include `.engine` files
3. **Graceful fallback**: Always handle TensorRT failures

## 📊 Supported Models

| Model | Dataset | TensorRT Support | Expected Speedup |
|-------|---------|------------------|------------------|
| Fast-SCNN | Cityscapes | ✅ | 2.5-3.5x |
| Fast-SCNN | COCO | ✅ | 2.5-3.5x |  
| BiSeNet V2 | Cityscapes | ✅ | 2.0-3.0x |
| BiSeNet V2 | COCO | ✅ | 2.0-3.0x |

## 🆘 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify CUDA/TensorRT installation
3. Test with `python test_tensorrt.py`
4. Check GPU memory with `nvidia-smi`

## 📚 Additional Resources

- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [PyTorch to TensorRT](https://pytorch.org/TensorRT/)
- [NVIDIA TensorRT Samples](https://github.com/NVIDIA/TensorRT)

---

**🎉 With TensorRT optimization, your semantic navigation will run 2-4x faster!** 