import os
import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

# --- Main Application ---

def main():
    parser = argparse.ArgumentParser(description='Fast-SCNN Semantic Navigation')
    parser.add_argument('--video-path', type=str, default=None, help='Path to a video file for navigation.')
    parser.add_argument('--output-path', type=str, default='output.avi', help='Path to save the output video.')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for inference.')
    parser.add_argument('--navigate', action='store_true', help='Enable semantic navigation decision-making.')
    parser.add_argument('--detect-objects', action='store_true', help='Enable YOLO object detection for obstacle avoidance.')
    parser.add_argument('--show-segmentation', action='store_true', help='Overlay the semantic segmentation mask.')
    parser.add_argument('--inference-interval', type=int, default=5, help='Run heavy inference every N frames for performance.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # --- Load Models ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load Fast-SCNN
    scnn_weights_path = os.path.join(script_dir, 'weights')
    scnn_model = get_fast_scnn('citys', pretrained=True, root=scnn_weights_path, map_cpu=(device.type == 'cpu')).to(device)
    scnn_model.eval()
    print("Fast-SCNN model loaded.")
    
    # Load YOLO
    yolo_net, yolo_classes = None, None
    if args.detect_objects:
        yolo_net = cv2.dnn.readNet(os.path.join(script_dir, "yolov3.weights"), os.path.join(script_dir, "yolov3.cfg"))
        with open(os.path.join(script_dir, "coco.names"), "r") as f:
            yolo_classes = [line.strip() for line in f.readlines()]
        print("YOLO model loaded.")

    # --- Setup Video Source and Display ---
    input_source = args.video_path if args.video_path else 0
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {input_source}")
        return

    frame_width, frame_height = 640, 480
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    scnn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- Define Navigation Costs ---
    # The cost map assigns a risk value to each semantic class. Lower is safer.
    # Class 0 (road) and 1 (sidewalk) are safe (cost 0).
    # Other classes have higher costs. Obstacles are assigned a very high cost.
    cost_map_config = { 0: 0, 1: 0, 8: 10, 9: 10, 11: 250, 12: 250, 13: 250, 14: 250, 15: 250 }
    default_cost = 100
    stop_threshold = 2000000 # Stop if the safest zone's score is above this.

    frame_count = 0
    last_pred, last_boxes = None, None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (frame_width, frame_height))

        # --- Staggered Inference for Performance ---
        if frame_count % args.inference_interval == 0:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = scnn_transform(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = scnn_model(image_tensor)
            last_pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy().astype(np.uint8)

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

        # --- Navigation Logic ---
        if args.navigate and last_pred is not None:
            numeric_cost_map = np.full(last_pred.shape, default_cost, dtype=np.int32)
            for class_id, cost in cost_map_config.items():
                numeric_cost_map[last_pred == class_id] = cost
            
            # Add YOLO objects to the cost map
            if last_boxes is not None:
                for box in last_boxes:
                    center_x, center_y, w, h = box.astype('int')
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    cv2.rectangle(numeric_cost_map, (x, y), (x + w, y + h), 1000, -1) # High cost for objects

            # Analyze safety zones
            zone_height = int(frame_height * 0.4)
            scan_area = numeric_cost_map[-zone_height:, :]
            
            zone_width = frame_width // 3
            left_zone = scan_area[:, :zone_width]
            center_zone = scan_area[:, zone_width : 2*zone_width]
            right_zone = scan_area[:, 2*zone_width:]

            left_score = np.sum(left_zone)
            center_score = np.sum(center_zone)
            right_score = np.sum(right_zone)
            
            # Make decision: Find the safest zone (lowest score)
            scores = {"Left": left_score, "Center": center_score, "Right": right_score}
            best_zone = min(scores, key=scores.get)
            
            # Decide to GO or STOP based on the risk score of the best available zone
            if scores[best_zone] > stop_threshold:
                decision = "STOP"
            else:
                decision = f"Go {best_zone}"

            # --- Visualization ---
            # Draw Zones
            cv2.rectangle(output_frame, (0, frame_height - zone_height), (zone_width, frame_height), (0, 255, 255), 2)
            cv2.rectangle(output_frame, (zone_width, frame_height - zone_height), (2*zone_width, frame_height), (0, 255, 0), 2)
            cv2.rectangle(output_frame, (2*zone_width, frame_height - zone_height), (frame_width, frame_height), (255, 0, 255), 2)
            # Display Text
            cv2.putText(output_frame, f"Decision: {decision}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output_frame, f"Decision: {decision}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output_frame, f"L:{left_score} C:{center_score} R:{right_score}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output_frame, f"L:{left_score} C:{center_score} R:{right_score}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the segmentation mask from Fast-SCNN
        if args.show_segmentation and last_pred is not None:
            mask = get_color_pallete(last_pred, 'citys', pil=True)
            output_frame = cv2.addWeighted(output_frame, 1, cv2.cvtColor(np.array(mask.convert('RGB')), cv2.COLOR_RGB2BGR), 0.4, 0)
        
        # Show the bounding boxes from YOLO
        if args.detect_objects and last_boxes is not None:
            for box in last_boxes:
                center_x, center_y, w, h = box.astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(output_frame)
        cv2.imshow('Semantic Navigation', output_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
