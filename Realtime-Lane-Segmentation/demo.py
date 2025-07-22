DEPTH_THREAT_THRESHOLD = 1000 # threshold for threat -MiDaS Model 
STOP_THRESHOLD = 8000000 # threshold for stopping

# --- Robot Physical Configuration ---
ROBOT_WIDTH_PERCENT = 0.4
ROBOT_SCAN_HEIGHT_PERCENT = 0.25

import os
import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from models.fast_scnn import get_fast_scnn
from models.bisenet_v2 import get_bisenet_v2
from models.tensorrt_utils import optimize_model_with_tensorrt, TensorRTInference, is_tensorrt_available
from utils.visualize import get_color_pallete

def main():
    parser = argparse.ArgumentParser(description='Intelligent Semantic Navigation with Depth Perception')
    # --- General arguments ---
    parser.add_argument('--video-path', type=str, default=None, help='Path to a video file. If not given, webcam will be used.')
    parser.add_argument('--output-path', type=str, default='output.avi', help='Path to save the output video.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU for inference.')
    parser.add_argument('--inference-interval', type=int, default=5, help='Run heavy inference every N frames for performance.')

    # --- Model selection ---
    parser.add_argument('--model', type=str, choices=['fast-scnn', 'bisenet-v2'], default='fast-scnn', 
                       help='Choose segmentation model: fast-scnn or bisenet-v2 (default: fast-scnn)')
    parser.add_argument('--dataset', type=str, choices=['citys', 'coco'], default='citys',
                       help='Dataset for model weights: citys or coco (default: citys)')

    # --- Performance optimization ---
    parser.add_argument('--fast_load', action='store_true', 
                       help='Use TensorRT optimization for faster inference (requires CUDA and TensorRT)')

    # --- Feature flags ---
    parser.add_argument('--navigate', action='store_true', help='Enable semantic navigation decision-making.')
    parser.add_argument('--detect-objects', action='store_true', help='Enable YOLO object detection.')
    parser.add_argument('--use-depth', action='store_true', help='Enable monocular depth estimation for smarter obstacle avoidance.')
    
    # --- Visualization flags ---
    parser.add_argument('--show-segmentation', action='store_true', help='Overlay the semantic segmentation mask.')
    parser.add_argument('--show-depth', action='store_true', help='Display the visual depth map in a separate window.')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # --- TensorRT Optimization Check ---
    use_tensorrt = args.fast_load and is_tensorrt_available() and device.type == 'cuda'
    if args.fast_load and not use_tensorrt:
        if not is_tensorrt_available():
            print("Warning: TensorRT not available. Install with: pip install tensorrt pycuda")
        elif device.type == 'cpu':
            print("Warning: TensorRT requires CUDA. Running on CPU instead.")
        print("Falling back to standard PyTorch inference.")

    # --- Load Models ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Semantic Segmentation Model (Fast-SCNN or BiSeNet V2)
    scnn_weights_path = os.path.join(script_dir, 'weights')
    
    if args.model == 'fast-scnn':
        scnn_model = get_fast_scnn(args.dataset, pretrained=True, root=scnn_weights_path, map_cpu=(device.type == 'cpu')).to(device)
        model_name = f"fast_scnn_{args.dataset}"
        print(f"Fast-SCNN model loaded for {args.dataset} dataset.")
    elif args.model == 'bisenet-v2':
        scnn_model = get_bisenet_v2(args.dataset, pretrained=True, root=scnn_weights_path, map_cpu=(device.type == 'cpu')).to(device)
        model_name = f"bisenet_v2_{args.dataset}"
        print(f"BiSeNet V2 model loaded for {args.dataset} dataset.")
    
    scnn_model.eval()

    # TensorRT Optimization
    trt_engine = None
    if use_tensorrt:
        print("🚀 Optimizing model with TensorRT for faster inference...")
        input_shape = (1, 3, 480, 640)  # (batch, channels, height, width)
        engine_path = optimize_model_with_tensorrt(
            scnn_model, 
            model_name, 
            scnn_weights_path, 
            input_shape, 
            fp16=True
        )
        
        if engine_path:
            try:
                trt_engine = TensorRTInference(engine_path)
                print("✅ TensorRT optimization successful! Using accelerated inference.")
            except Exception as e:
                print(f"Warning: TensorRT engine loading failed: {e}")
                print("Falling back to standard PyTorch inference.")
                trt_engine = None
        else:
            print("TensorRT optimization failed. Using standard PyTorch inference.")

    # 2. YOLO (Object Detection)
    yolo_net, yolo_classes = None, None
    if args.detect_objects:
        try:
            yolo_weights_path = os.path.join(script_dir, "yolov3.weights")
            yolo_cfg_path = os.path.join(script_dir, "yolov3.cfg")
            yolo_names_path = os.path.join(script_dir, "coco.names")
            if not os.path.exists(yolo_weights_path) or not os.path.exists(yolo_cfg_path):
                 raise FileNotFoundError("YOLOv3 weights or cfg not found. Please download them.")
            yolo_net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
            with open(yolo_names_path, "r") as f:
                yolo_classes = [line.strip() for line in f.readlines()]
            print("YOLO model loaded.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            args.detect_objects = False
            
    # 3. MiDaS (Depth Estimation)
    midas, midas_transform = None, None
    if args.use_depth:
        try:
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
            midas.eval()
            midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            print("MiDaS depth model loaded.")
        except Exception as e:
            print(f"Error loading MiDaS model: {e}. Depth estimation will be disabled.")
            args.use_depth = False

    # --- Setup Video I/O ---
    input_source = args.video_path if args.video_path else 0
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {input_source}")
        return

    frame_width, frame_height = 640, 480
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    scnn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- Navigation Parameters ---
    cost_map_config = {0: 0, 1: 0, 8: 10, 9: 10, 11: 250, 12: 250, 13: 250, 14: 250, 15: 250}
    default_cost = 100
    depth_threat_threshold = DEPTH_THREAT_THRESHOLD
    stop_threshold = STOP_THRESHOLD

    frame_count = 0
    last_pred, last_boxes, last_depth = None, None, None

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))

        # --- Staggered Inference for Performance ---
        if frame_count % args.inference_interval == 0:
            image_rgb_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb_numpy)
            
            with torch.no_grad():
                # Semantic Segmentation Inference (Fast-SCNN or BiSeNet V2)
                if trt_engine is not None:
                    # TensorRT accelerated inference
                    image_tensor = scnn_transform(image_pil).unsqueeze(0)
                    input_np = image_tensor.cpu().numpy().astype(np.float32)
                    trt_output = trt_engine.infer(input_np)
                    last_pred = np.argmax(trt_output, axis=1).squeeze(0).astype(np.uint8)
                else:
                    # Standard PyTorch inference
                    image_tensor = scnn_transform(image_pil).unsqueeze(0).to(device)
                    outputs = scnn_model(image_tensor)
                    last_pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy().astype(np.uint8)

                # MiDaS Depth Inference
                if args.use_depth and midas:
                    midas_input = midas_transform(image_rgb_numpy).to(device)
                    prediction = midas(midas_input)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1), size=image_pil.size[::-1], mode="bicubic", align_corners=False
                    ).squeeze()
                    depth_map = prediction.cpu().numpy()
                    depth_min = depth_map.min()
                    depth_max = depth_map.max()
                    if depth_max - depth_min > 0:
                        last_depth = 1000 * (depth_map - depth_min) / (depth_max - depth_min)
                    else:
                        last_depth = np.zeros(depth_map.shape)

            # YOLO Inference
            if args.detect_objects and yolo_net:
                last_boxes = []
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                yolo_net.setInput(blob)
                layer_names = yolo_net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
                yolo_outs = yolo_net.forward(output_layers)
                for yolo_output in yolo_outs:
                    for detection in yolo_output:
                        if detection[5:].max() > 0.5:
                            last_boxes.append(detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height]))

        output_frame = np.copy(frame)
        vis_overlay = output_frame.copy()

        # --- Navigation Logic ---
        if args.navigate and last_pred is not None:
            numeric_cost_map = np.full(last_pred.shape, default_cost, dtype=np.int32)
            for class_id, cost in cost_map_config.items():
                numeric_cost_map[last_pred == class_id] = cost
            
            if last_boxes:
                for box in last_boxes:
                    is_threat = True
                    if args.use_depth and last_depth is not None:
                        center_x, center_y, w, h = box.astype('int')
                        x1, y1 = max(0, int(center_x - w/2)), max(0, int(center_y - h/2))
                        x2, y2 = min(frame_width, x1 + w), min(frame_height, y1 + h)
                        if x1 < x2 and y1 < y2:
                            depth_roi = last_depth[y1:y2, x1:x2]
                            mean_depth = np.mean(depth_roi) if depth_roi.size > 0 else 0
                            if mean_depth < depth_threat_threshold:
                                is_threat = False
                    if is_threat:
                        cv2.rectangle(numeric_cost_map, (x1, y1), (x2, y2), 1000, -1)

            # --- Robot-Aware Pathfinding ---
            zone_height_pixels = int(frame_height * ROBOT_SCAN_HEIGHT_PERCENT)
            zone_width_pixels = int(frame_width * ROBOT_WIDTH_PERCENT)
            scan_area = numeric_cost_map[-zone_height_pixels:, :]
            perspective_weights = np.linspace(1, 5, zone_height_pixels).reshape(zone_height_pixels, 1)
            weighted_scan_area = scan_area * perspective_weights
            
            path_scores = []
            for x_start in range(frame_width - zone_width_pixels):
                path_zone = weighted_scan_area[:, x_start : x_start + zone_width_pixels]
                path_scores.append(np.sum(path_zone))

            best_path_score = float('inf')
            best_path_start_x = 0
            if path_scores:
                best_path_index = np.argmin(path_scores)
                best_path_score = path_scores[best_path_index]
                best_path_start_x = best_path_index
            
            # --- Decision Making ---
            decision = "STOP"
            if best_path_score <= stop_threshold:
                path_center_x = best_path_start_x + (zone_width_pixels / 2)
                center_zone_start = frame_width * 0.3
                center_zone_end = frame_width * 0.7
                if path_center_x < center_zone_start:
                    decision = "Go Left"
                elif path_center_x > center_zone_end:
                    decision = "Go Right"
                else:
                    decision = "Go Center"
            
            # --- Visualization of 3 Zones ---
            zone_width = frame_width // 3
            zone_start_y = frame_height - zone_height_pixels
            zone_colors = {"Left": (0, 255, 255), "Center": (0, 255, 0), "Right": (255, 0, 255)}

            # Draw all three zones with default brightness
            cv2.rectangle(vis_overlay, (0, zone_start_y), (zone_width, frame_height), zone_colors["Left"], 2)
            cv2.rectangle(vis_overlay, (zone_width, zone_start_y), (2*zone_width, frame_height), zone_colors["Center"], 2)
            cv2.rectangle(vis_overlay, (2*zone_width, zone_start_y), (frame_width, frame_height), zone_colors["Right"], 2)

            # If a decision is made, "press" the corresponding box by making it darker/filled
            if decision.startswith("Go"):
                chosen_zone = decision.split(" ")[1]
                if chosen_zone == "Left":
                    cv2.rectangle(vis_overlay, (0, zone_start_y), (zone_width, frame_height), zone_colors["Left"], -1)
                elif chosen_zone == "Center":
                    cv2.rectangle(vis_overlay, (zone_width, zone_start_y), (2*zone_width, frame_height), zone_colors["Center"], -1)
                elif chosen_zone == "Right":
                    cv2.rectangle(vis_overlay, (2*zone_width, zone_start_y), (frame_width, frame_height), zone_colors["Right"], -1)

            output_frame = cv2.addWeighted(vis_overlay, 0.4, output_frame, 0.6, 0)
            
            cv2.putText(output_frame, f"Decision: {decision}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(output_frame, f"Decision: {decision}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            score_text = f"Best Path Score: {int(best_path_score)}"
            cv2.putText(output_frame, score_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(output_frame, score_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        if args.show_segmentation and last_pred is not None:
            mask = get_color_pallete(last_pred, 'citys', pil=True)
            output_frame = cv2.addWeighted(output_frame, 1, cv2.cvtColor(np.array(mask.convert('RGB')), cv2.COLOR_RGB2BGR), 0.4, 0)
        
        if args.detect_objects and last_boxes is not None:
            for box in last_boxes:
                center_x, center_y, w, h = box.astype('int')
                x, y = int(center_x - w/2), int(center_y - h/2)
                color = (0, 255, 0)
                closeness_text = ""
                if args.use_depth and last_depth is not None:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame_width, x1 + w), min(frame_height, y1 + h)
                    if x1 < x2 and y1 < y2:
                        depth_roi = last_depth[y1:y2, x1:x2]
                        mean_depth = np.mean(depth_roi) if depth_roi.size > 0 else 0
                        closeness_text = f"Closeness: {int(mean_depth)}"
                        if mean_depth >= depth_threat_threshold:
                            color = (0, 0, 255)
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
                if closeness_text:
                    text_y = y - 10 if y > 20 else y + 20
                    cv2.putText(output_frame, closeness_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if args.show_depth and last_depth is not None:
            depth_display = cv2.normalize(last_depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow('Depth Map', depth_display)

        out.write(output_frame)
        cv2.imshow('Semantic Navigation', output_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()