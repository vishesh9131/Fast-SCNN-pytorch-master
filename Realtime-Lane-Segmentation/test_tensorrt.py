#!/usr/bin/env python3
"""
TensorRT optimization and performance testing script
"""

import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from models.bisenet_v2 import get_bisenet_v2
from models.tensorrt_utils import optimize_model_with_tensorrt, TensorRTInference, is_tensorrt_available

def benchmark_model(model, trt_engine, input_tensor, num_iterations=100):
    """Benchmark PyTorch vs TensorRT inference speed"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Warm up
    print("Warming up models...")
    for _ in range(10):
        if trt_engine:
            _ = trt_engine.infer(input_tensor.cpu().numpy())
        else:
            with torch.no_grad():
                _ = model(input_tensor.to(device))
    
    # PyTorch benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    pytorch_times = []
    print(f"Benchmarking PyTorch inference ({num_iterations} iterations)...")
    
    for i in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor.to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        pytorch_times.append(end_time - start_time)
        
        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{num_iterations} iterations")
    
    # TensorRT benchmark
    tensorrt_times = []
    if trt_engine:
        print(f"Benchmarking TensorRT inference ({num_iterations} iterations)...")
        input_np = input_tensor.cpu().numpy().astype(np.float32)
        
        for i in range(num_iterations):
            start_time = time.time()
            _ = trt_engine.infer(input_np)
            end_time = time.time()
            tensorrt_times.append(end_time - start_time)
            
            if (i + 1) % 25 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations")
    
    return pytorch_times, tensorrt_times

def test_tensorrt_optimization():
    """Test TensorRT optimization for both models"""
    
    print("=" * 60)
    print("TENSORRT OPTIMIZATION AND PERFORMANCE TEST")
    print("=" * 60)
    
    # Check TensorRT availability
    print(f"\n📋 System Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"TensorRT available: {is_tensorrt_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. TensorRT requires CUDA.")
        return
    
    if not is_tensorrt_available():
        print("❌ TensorRT not available. Install with: pip install tensorrt pycuda")
        print("💡 Continuing with PyTorch-only benchmark...")
    
    device = torch.device('cuda')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'weights')
    
    # Test input
    input_shape = (1, 3, 480, 640)
    test_input = torch.randn(*input_shape)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    models_to_test = [
        ('fast-scnn', 'citys'),
        ('bisenet-v2', 'coco')
    ]
    
    results = {}
    
    for model_type, dataset in models_to_test:
        print(f"\n🔧 Testing {model_type.upper()} with {dataset} dataset...")
        
        # Load model
        if model_type == 'fast-scnn':
            model = get_fast_scnn(dataset, pretrained=True, root=weights_path, map_cpu=False).to(device)
            model_name = f"fast_scnn_{dataset}"
        else:
            model = get_bisenet_v2(dataset, pretrained=True, root=weights_path, map_cpu=False).to(device)
            model_name = f"bisenet_v2_{dataset}"
        
        model.eval()
        
        # TensorRT optimization
        trt_engine = None
        engine_path = None
        
        if is_tensorrt_available():
            print(f"🚀 Optimizing {model_name} with TensorRT...")
            engine_path = optimize_model_with_tensorrt(
                model, model_name, weights_path, input_shape, fp16=True
            )
            
            if engine_path:
                try:
                    trt_engine = TensorRTInference(engine_path)
                    print(f"✅ TensorRT engine created successfully!")
                except Exception as e:
                    print(f"❌ TensorRT engine loading failed: {e}")
            else:
                print(f"❌ TensorRT optimization failed for {model_name}")
        
        # Performance benchmark
        print(f"\n⏱️  Performance Benchmark for {model_name}:")
        pytorch_times, tensorrt_times = benchmark_model(model, trt_engine, test_input, num_iterations=50)
        
        # Calculate statistics
        pytorch_avg = np.mean(pytorch_times) * 1000  # Convert to ms
        pytorch_fps = 1.0 / np.mean(pytorch_times)
        
        print(f"\n📊 Results for {model_name}:")
        print(f"PyTorch - Avg: {pytorch_avg:.2f}ms, FPS: {pytorch_fps:.1f}")
        
        if tensorrt_times:
            tensorrt_avg = np.mean(tensorrt_times) * 1000
            tensorrt_fps = 1.0 / np.mean(tensorrt_times)
            speedup = pytorch_avg / tensorrt_avg
            
            print(f"TensorRT - Avg: {tensorrt_avg:.2f}ms, FPS: {tensorrt_fps:.1f}")
            print(f"🚀 Speedup: {speedup:.2f}x faster with TensorRT!")
            
            results[model_name] = {
                'pytorch_ms': pytorch_avg,
                'pytorch_fps': pytorch_fps,
                'tensorrt_ms': tensorrt_avg,
                'tensorrt_fps': tensorrt_fps,
                'speedup': speedup,
                'engine_path': engine_path
            }
        else:
            results[model_name] = {
                'pytorch_ms': pytorch_avg,
                'pytorch_fps': pytorch_fps,
                'tensorrt_ms': None,
                'tensorrt_fps': None,
                'speedup': None,
                'engine_path': None
            }
    
    # Summary
    print(f"\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for model_name, stats in results.items():
        print(f"\n🔸 {model_name.upper()}")
        print(f"   PyTorch: {stats['pytorch_fps']:.1f} FPS ({stats['pytorch_ms']:.2f}ms)")
        if stats['tensorrt_fps']:
            print(f"   TensorRT: {stats['tensorrt_fps']:.1f} FPS ({stats['tensorrt_ms']:.2f}ms)")
            print(f"   Speedup: {stats['speedup']:.2f}x")
            if stats['engine_path']:
                print(f"   Engine: {os.path.basename(stats['engine_path'])}")
        else:
            print(f"   TensorRT: Not available")
    
    print(f"\n💡 Usage with TensorRT:")
    print(f"python demo.py --model fast-scnn --dataset citys --fast_load --navigate")
    print(f"python demo.py --model bisenet-v2 --dataset coco --fast_load --navigate")
    
    print(f"\n🤖 ROS Usage:")
    print(f"<param name=\"fast_load\" value=\"true\"/>")
    
    print(f"\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)

if __name__ == '__main__':
    test_tensorrt_optimization() 