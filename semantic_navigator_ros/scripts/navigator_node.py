#!/usr/bin/env python
import rospy
import cv2
import torch
import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from PIL import Image as PILImage
from torchvision import transforms

# Import model-related code from the parent directory
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models.fast_scnn import get_fast_scnn
from models.bisenet_v2 import get_bisenet_v2
from models.tensorrt_utils import optimize_model_with_tensorrt, TensorRTInference, is_tensorrt_available
from utils.visualize import get_color_pallete

class SemanticNavigatorNode:
    def __init__(self):
        rospy.init_node('semantic_navigator_node')
        rospy.loginfo("Initializing Semantic Navigator Node...")

        # --- Load Parameters from ROS Param Server ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_interval = rospy.get_param('~inference_interval', 5)
        self.show_segmentation = rospy.get_param('~show_segmentation', True)
        self.detect_objects = rospy.get_param('~detect_objects', True)
        self.use_depth = rospy.get_param('~use_depth', True)
        
        # Model selection parameters
        self.model_type = rospy.get_param('~model', 'fast-scnn')  # 'fast-scnn' or 'bisenet-v2'
        self.dataset = rospy.get_param('~dataset', 'citys')  # 'citys' or 'coco'
        
        # TensorRT optimization parameter
        self.fast_load = rospy.get_param('~fast_load', False)
        
        # Robot and Navigation parameters
        self.robot_width_percent = rospy.get_param('~robot_width_percent', 0.4)
        self.scan_height_percent = rospy.get_param('~scan_height_percent', 0.25)
        self.stop_threshold = rospy.get_param('~stop_threshold', 8000000)
        self.depth_threat_threshold = rospy.get_param('~depth_threat_threshold', 800)
        
        # --- ROS Communications ---
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.vis_pub = rospy.Publisher('/semantic_navigation/visualization', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/semantic_navigation/depth_map', Image, queue_size=1)
        self.point_cloud_pub = rospy.Publisher('/safe_zone_cloud', PointCloud2, queue_size=1)

        # --- Load Models ---
        self.load_models()

        # --- Image Processing ---
        self.scnn_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.frame_count = 0
        self.last_pred, self.last_boxes, self.last_depth = None, None, None
        rospy.loginfo("Semantic Navigator Node is ready.")

    def load_models(self):
        models_path = os.path.join(script_dir, '..', 'models')
        
        # 1. Semantic Segmentation Model (Fast-SCNN or BiSeNet V2)
        if self.model_type == 'fast-scnn':
            self.scnn_model = get_fast_scnn(self.dataset, pretrained=True, root=models_path, map_cpu=(self.device.type == 'cpu')).to(self.device)
            self.scnn_model.eval()
            model_name = f"fast_scnn_{self.dataset}"
            rospy.loginfo(f"Fast-SCNN model loaded for {self.dataset} dataset.")
        elif self.model_type == 'bisenet-v2':
            self.scnn_model = get_bisenet_v2(self.dataset, pretrained=True, root=models_path, map_cpu=(self.device.type == 'cpu')).to(self.device)
            self.scnn_model.eval()
            model_name = f"bisenet_v2_{self.dataset}"
            rospy.loginfo(f"BiSeNet V2 model loaded for {self.dataset} dataset.")
        else:
            rospy.logerr(f"Unknown model type: {self.model_type}. Supported: fast-scnn, bisenet-v2")
            return
            
        # TensorRT Optimization
        self.trt_engine = None
        use_tensorrt = self.fast_load and is_tensorrt_available() and self.device.type == 'cuda'
        
        if self.fast_load and not use_tensorrt:
            if not is_tensorrt_available():
                rospy.logwarn("TensorRT not available. Install with: pip install tensorrt pycuda")
            elif self.device.type == 'cpu':
                rospy.logwarn("TensorRT requires CUDA. Running on CPU instead.")
            rospy.logwarn("Falling back to standard PyTorch inference.")
        
        if use_tensorrt:
            rospy.loginfo("🚀 Optimizing model with TensorRT for faster inference...")
            input_shape = (1, 3, 480, 640)  # (batch, channels, height, width)
            engine_path = optimize_model_with_tensorrt(
                self.scnn_model, 
                model_name, 
                models_path, 
                input_shape, 
                fp16=True
            )
            
            if engine_path:
                try:
                    self.trt_engine = TensorRTInference(engine_path)
                    rospy.loginfo("✅ TensorRT optimization successful! Using accelerated inference.")
                except Exception as e:
                    rospy.logwarn(f"TensorRT engine loading failed: {e}")
                    rospy.logwarn("Falling back to standard PyTorch inference.")
                    self.trt_engine = None
            else:
                rospy.logwarn("TensorRT optimization failed. Using standard PyTorch inference.")
            
        # Setup transforms for the segmentation model
        self.scnn_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # 2. YOLO
        self.yolo_net = None
        if self.detect_objects:
            try:
                self.yolo_net = cv2.dnn.readNet(os.path.join(models_path, "yolov3.weights"), os.path.join(models_path, "yolov3.cfg"))
                rospy.loginfo("YOLO model loaded.")
            except Exception as e:
                rospy.logerr(f"Failed to load YOLO: {e}")
                self.detect_objects = False
                
        # 3. MiDaS
        self.midas, self.midas_transform = None, None
        if self.use_depth:
            try:
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
                self.midas.eval()
                self.midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
                rospy.loginfo("MiDaS depth model loaded.")
            except Exception as e:
                rospy.logerr(f"Failed to load MiDaS: {e}. Disabling depth perception.")
                self.use_depth = False

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        frame_height, frame_width, _ = cv_image.shape

        if self.frame_count % self.inference_interval == 0:
            self.run_inference(cv_image)

        output_frame = self.run_navigation_and_visualization(cv_image, data.header)
        
        try:
            self.vis_pub.publish(self.bridge.cv2_to_imgmsg(output_frame, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(e)

        self.frame_count += 1
    
    def run_inference(self, frame):
        image_rgb_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb_numpy)
        
        with torch.no_grad():
            # Semantic Segmentation Inference
            if self.trt_engine is not None:
                # TensorRT accelerated inference
                image_tensor = self.scnn_transform(image_pil).unsqueeze(0)
                input_np = image_tensor.cpu().numpy().astype(np.float32)
                trt_output = self.trt_engine.infer(input_np)
                self.last_pred = np.argmax(trt_output, axis=1).squeeze(0).astype(np.uint8)
            else:
                # Standard PyTorch inference
                image_tensor = self.scnn_transform(image_pil).unsqueeze(0).to(self.device)
                outputs = self.scnn_model(image_tensor)
                self.last_pred = outputs[0].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            if self.use_depth and self.midas:
                midas_input = self.midas_transform(image_rgb_numpy).to(self.device)
                prediction = self.midas(midas_input)
                prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=image_pil.size[::-1], mode="bicubic", align_corners=False).squeeze()
                depth_map = prediction.cpu().numpy()
                depth_min, depth_max = depth_map.min(), depth_map.max()
                self.last_depth = 1000 * (depth_map - depth_min) / (depth_max - depth_min) if depth_max > depth_min else np.zeros_like(depth_map)
                
                depth_display = cv2.normalize(self.last_depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_display, "bgr8"))


            if self.detect_objects and self.yolo_net:
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                self.yolo_net.setInput(blob)
                layer_names = self.yolo_net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
                yolo_outs = self.yolo_net.forward(output_layers)
                self.last_boxes = [det[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]) for out in yolo_outs for det in out if det[5:].max() > 0.5]

    def run_navigation_and_visualization(self, frame, header):
        output_frame = frame.copy()
        vis_overlay = output_frame.copy()

        if self.last_pred is not None:
            numeric_cost_map = np.full(self.last_pred.shape, 100, dtype=np.int32)
            cost_map_config = {0: 0, 1: 0, 8: 10, 9: 10, 11: 250, 12: 250, 13: 250, 14: 250, 15: 250}
            for class_id, cost in cost_map_config.items():
                numeric_cost_map[self.last_pred == class_id] = cost
            
            if self.last_boxes:
                for box in self.last_boxes:
                    is_threat = not (self.use_depth and self.last_depth is not None and self.is_object_safe(box, self.last_depth))
                    if is_threat:
                        x1, y1, x2, y2 = self.get_box_coords(box, frame.shape)
                        cv2.rectangle(numeric_cost_map, (x1, y1), (x2, y2), 1000, -1)
            
            self.publish_safe_zone_cloud(numeric_cost_map, header)

            decision, best_path_start_x, zone_height_pixels = self.find_best_path(numeric_cost_map, frame.shape)
            self.publish_robot_command(decision)
            self.draw_navigation_visuals(vis_overlay, decision, best_path_start_x, zone_height_pixels, frame.shape)
            output_frame = cv2.addWeighted(vis_overlay, 0.4, output_frame, 0.6, 0)
        
        if self.show_segmentation and self.last_pred is not None:
            mask = get_color_pallete(self.last_pred, 'citys', pil=True)
            output_frame = cv2.addWeighted(output_frame, 1, cv2.cvtColor(np.array(mask.convert('RGB')), cv2.COLOR_RGB2BGR), 0.4, 0)
        
        if self.detect_objects and self.last_boxes:
            self.draw_object_boxes(output_frame, self.last_boxes, self.last_depth)
            
        return output_frame

    def is_object_safe(self, box, depth_map):
        x1, y1, x2, y2 = self.get_box_coords(box, depth_map.shape)
        depth_roi = depth_map[y1:y2, x1:x2]
        return np.mean(depth_roi) < self.depth_threat_threshold if depth_roi.size > 0 else True

    def get_box_coords(self, box, shape):
        center_x, center_y, w, h = box.astype('int')
        x1 = max(0, int(center_x - w/2))
        y1 = max(0, int(center_y - h/2))
        x2 = min(shape[1], x1 + w)
        y2 = min(shape[0], y1 + h)
        return x1, y1, x2, y2

    def find_best_path(self, cost_map, shape):
        zone_height_pixels = int(shape[0] * self.scan_height_percent)
        zone_width_pixels = int(shape[1] * self.robot_width_percent)
        scan_area = cost_map[-zone_height_pixels:, :]
        perspective_weights = np.linspace(1, 5, zone_height_pixels).reshape(zone_height_pixels, 1)
        weighted_scan_area = scan_area * perspective_weights
        
        path_scores = [np.sum(weighted_scan_area[:, x:x+zone_width_pixels]) for x in range(shape[1] - zone_width_pixels)]
        
        best_path_start_x = np.argmin(path_scores) if path_scores else 0
        best_path_score = path_scores[best_path_start_x] if path_scores else float('inf')
        
        decision = "STOP"
        if best_path_score <= self.stop_threshold:
            path_center_x = best_path_start_x + (zone_width_pixels / 2)
            if path_center_x < shape[1] * 0.3: decision = "Go Left"
            elif path_center_x > shape[1] * 0.7: decision = "Go Right"
            else: decision = "Go Center"
            
        return decision, best_path_start_x, zone_height_pixels

    def publish_robot_command(self, decision):
        twist_msg = Twist()
        if decision.startswith("Go"):
            twist_msg.linear.x = 0.5
            if "Left" in decision: twist_msg.angular.z = 0.5
            elif "Right" in decision: twist_msg.angular.z = -0.5
        self.cmd_vel_pub.publish(twist_msg)

    def draw_navigation_visuals(self, overlay, decision, path_start_x, zone_height, shape):
        zone_width = shape[1] // 3
        zone_start_y = shape[0] - zone_height
        colors = {"Left": (0, 255, 255), "Center": (0, 255, 0), "Right": (255, 0, 255)}
        for zone, color in colors.items():
            start_x = 0 if zone == "Left" else (zone_width if zone == "Center" else 2*zone_width)
            end_x = zone_width if zone == "Left" else (2*zone_width if zone == "Center" else shape[1])
            is_chosen = decision.endswith(zone)
            cv2.rectangle(overlay, (start_x, zone_start_y), (end_x, shape[0]), color, -1 if is_chosen else 2)
            
        cv2.putText(overlay, f"Decision: {decision}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    def draw_object_boxes(self, frame, boxes, depth_map):
        for box in boxes:
            x1, y1, x2, y2 = self.get_box_coords(box, frame.shape)
            is_safe = self.is_object_safe(box, depth_map) if self.use_depth and depth_map is not None else True
            color = (0, 255, 0) if is_safe else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if self.use_depth and depth_map is not None:
                depth_roi = depth_map[y1:y2, x1:x2]
                mean_depth = np.mean(depth_roi) if depth_roi.size > 0 else -1
                text = f"Closeness: {int(mean_depth)}"
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def publish_safe_zone_cloud(self, cost_map, header):
        safe_pixels = np.argwhere(cost_map == 0)
        points = [[(p[1]/cost_map.shape[1])*2-1, (p[0]/cost_map.shape[0])*2-1, 0] for p in safe_pixels]
        if points:
            cloud_msg = pc2.create_cloud_xyz32(header, points)
            self.point_cloud_pub.publish(cloud_msg)

if __name__ == '__main__':
    try:
        SemanticNavigatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 