#!/usr/bin/env python3
"""
Test script to demonstrate model switching between Fast-SCNN and BiSeNet V2
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from models.fast_scnn import get_fast_scnn
from models.bisenet_v2 import get_bisenet_v2
from utils.visualize import get_color_pallete
from torchvision import transforms

def test_model_switching():
    """Test both Fast-SCNN and BiSeNet V2 models on a sample image"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a sample image (you can replace this with an actual image path)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Or load an actual image
    # test_image = cv2.imread('path_to_your_image.jpg')
    
    if test_image is None:
        print("Using random test image")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Convert to PIL Image
    image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'weights')
    
    print("\n" + "="*60)
    print("TESTING MODEL SWITCHING FEATURE")
    print("="*60)
    
    # Test 1: Fast-SCNN with Cityscapes
    print("\n1. Testing Fast-SCNN with Cityscapes dataset...")
    try:
        model_fastscnn_citys = get_fast_scnn('citys', pretrained=True, root=weights_path, map_cpu=(device.type == 'cpu')).to(device)
        model_fastscnn_citys.eval()
        
        with torch.no_grad():
            input_tensor = transform(image_pil).unsqueeze(0).to(device)
            output = model_fastscnn_citys(input_tensor)
            prediction = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy().astype(np.uint8)
            
        print(f"✓ Fast-SCNN (Cityscapes) - Output shape: {prediction.shape}")
        
        # Save visualization
        mask = get_color_pallete(prediction, 'citys', pil=True)
        result_image = cv2.addWeighted(test_image, 0.7, cv2.cvtColor(np.array(mask.convert('RGB')), cv2.COLOR_RGB2BGR), 0.3, 0)
        cv2.imwrite('test_fastscnn_citys.jpg', result_image)
        print("  Saved result: test_fastscnn_citys.jpg")
        
    except Exception as e:
        print(f"✗ Fast-SCNN (Cityscapes) failed: {e}")
    
    # Test 2: BiSeNet V2 with COCO
    print("\n2. Testing BiSeNet V2 with COCO dataset...")
    try:
        model_bisenet_coco = get_bisenet_v2('coco', pretrained=True, root=weights_path, map_cpu=(device.type == 'cpu')).to(device)
        model_bisenet_coco.eval()
        
        with torch.no_grad():
            input_tensor = transform(image_pil).unsqueeze(0).to(device)
            output = model_bisenet_coco(input_tensor)
            prediction = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy().astype(np.uint8)
            
        print(f"✓ BiSeNet V2 (COCO) - Output shape: {prediction.shape}")
        
        # Save visualization  
        mask = get_color_pallete(prediction, 'coco', pil=True)
        result_image = cv2.addWeighted(test_image, 0.7, cv2.cvtColor(np.array(mask.convert('RGB')), cv2.COLOR_RGB2BGR), 0.3, 0)
        cv2.imwrite('test_bisenetv2_coco.jpg', result_image)
        print("  Saved result: test_bisenetv2_coco.jpg")
        
    except Exception as e:
        print(f"✗ BiSeNet V2 (COCO) failed: {e}")
    
    # Test 3: Model Memory Usage Comparison
    print("\n3. Memory usage comparison...")
    try:
        def get_model_size(model):
            param_size = 0
            param_sum = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
                param_sum += param.nelement()
            buffer_size = 0
            buffer_sum = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
                buffer_sum += buffer.nelement()
            all_size = (param_size + buffer_size) / 1024 / 1024
            return all_size, param_sum + buffer_sum
        
        fastscnn_size, fastscnn_params = get_model_size(model_fastscnn_citys)
        bisenet_size, bisenet_params = get_model_size(model_bisenet_coco)
        
        print(f"Fast-SCNN: {fastscnn_size:.2f} MB, {fastscnn_params:,} parameters")
        print(f"BiSeNet V2: {bisenet_size:.2f} MB, {bisenet_params:,} parameters")
        
    except Exception as e:
        print(f"Memory comparison failed: {e}")
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    print("\n# For demo.py (standalone):")
    print("python demo.py --model fast-scnn --dataset citys --show-segmentation")
    print("python demo.py --model bisenet-v2 --dataset coco --show-segmentation --navigate")
    
    print("\n# For ROS (edit launch file parameters):")
    print("<param name=\"model\" value=\"fast-scnn\"/>")
    print("<param name=\"dataset\" value=\"citys\"/>")
    print("# OR")
    print("<param name=\"model\" value=\"bisenet-v2\"/>")
    print("<param name=\"dataset\" value=\"coco\"/>")
    
    print("\n" + "="*60)
    print("TEST COMPLETED!")
    print("="*60)

if __name__ == '__main__':
    test_model_switching() 