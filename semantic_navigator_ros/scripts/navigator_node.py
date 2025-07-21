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

# Import model-related code
import sys
# This is a bit of a hack to get the model files imported correctly
# A better solution would be to make this a proper python package
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from models.fast_scnn import get_fast_scnn
from utils.visualize import get_color_pallete
from PIL import Image as PILImage
from torchvision import transforms

class SemanticNavigator:
    def __init__(self):
        rospy.init_node('semantic_navigator', anonymous=True)

        # --- Parameters ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_interval = rospy.get_param('~inference_interval', 5)
        self.show_segmentation = rospy.get_param('~show_segmentation', True)
        self.detect_objects = rospy.get_param('~detect_objects', True)
        
        # --- ROS Publishers and Subscribers ---
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.point_cloud_pub = rospy.Publisher('/safe_zone_cloud', PointCloud2, queue_size=1)
        self.vis_pub = rospy.Publisher('/semantic_navigation/visualization', Image, queue_size=1)

        # --- Load Models ---
        models_path = os.path.join(script_dir, '..', 'models')
        
        # Load Fast-SCNN
        self.scnn_model = get_fast_scnn('citys', pretrained=True, root=models_path, map_cpu=(self.device.type == 'cpu')).to(self.device)
        self.scnn_model.eval()
        rospy.loginfo("Fast-SCNN model loaded.")

        # Load YOLO
        self.yolo_net, self.yolo_classes = None, None
        if self.detect_objects:
            self.yolo_net = cv2.dnn.readNet(os.path.join(models_path, "yolov3.weights"), os.path.join(models_path, "yolov3.cfg"))
            with open(os.path.join(models_path, "coco.names"), "r") as f:
                self.yolo_classes = [line.strip() for line in f.readlines()]
            rospy.loginfo("YOLO model loaded.")

        # --- Image Processing and Navigation ---
        self.scnn_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Navigation Costs
        self.cost_map_config = { 0: 0, 1: 0, 8: 10, 9: 10, 11: 250, 12: 250, 13: 250, 14: 250, 15: 250 }
        self.default_cost = 100
        self.stop_threshold = 8000000 

        self.frame_count = 0
        self.last_pred, self.last_boxes = None, None

        rospy.loginfo("Semantic Navigator is ready.")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        frame_height, frame_width, _ = cv_image.shape

        # Staggered Inference
        if self.frame_count % self.inference_interval == 0:
            image_pil = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            image_tensor = self.scnn_transform(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.scnn_model(image_tensor)
            self.last_pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy().astype(np.uint8)

            if self.detect_objects and self.yolo_net:
                self.last_boxes = []
                blob = cv2.dnn.blobFromImage(cv_image, 1/255.0, (416, 416), swapRB=True, crop=False)
                self.yolo_net.setInput(blob)
                layer_names = self.yolo_net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
                yolo_outs = self.yolo_net.forward(output_layers)
                for yolo_output in yolo_outs:
                    for detection in yolo_output:
                        if detection[5:].max() > 0.5:
                            self.last_boxes.append(detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height]))

        output_frame = np.copy(cv_image)

        if self.last_pred is not None:
            numeric_cost_map = np.full(self.last_pred.shape, self.default_cost, dtype=np.int32)
            for class_id, cost in self.cost_map_config.items():
                numeric_cost_map[self.last_pred == class_id] = cost
            
            if self.last_boxes:
                for box in self.last_boxes:
                    center_x, center_y, w, h = box.astype('int')
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    cv2.rectangle(numeric_cost_map, (x, y), (x + w, y + h), 1000, -1)
            
            self.publish_safe_zone_cloud(numeric_cost_map, data.header)

            zone_height = int(frame_height * 0.4)
            scan_area = numeric_cost_map[-zone_height:, :]
            perspective_weights = np.linspace(1, 5, zone_height).reshape(zone_height, 1)
            weighted_scan_area = scan_area * perspective_weights
            
            zone_width = frame_width // 3
            left_zone = weighted_scan_area[:, :zone_width]
            center_zone = weighted_scan_area[:, zone_width : 2*zone_width]
            right_zone = weighted_scan_area[:, 2*zone_width:]

            left_score = np.sum(left_zone)
            center_score = np.sum(center_zone)
            right_score = np.sum(right_zone)
            
            scores = {"Left": left_score, "Center": center_score, "Right": right_score}
            best_zone = min(scores, key=scores.get)
            
            twist_msg = Twist()
            if scores[best_zone] > self.stop_threshold:
                decision = "STOP"
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
            else:
                decision = f"Go {best_zone}"
                twist_msg.linear.x = 0.5 # Constant forward speed
                if best_zone == "Left":
                    twist_msg.angular.z = 0.5
                elif best_zone == "Right":
                    twist_msg.angular.z = -0.5
                else: # Center
                    twist_msg.angular.z = 0.0
            
            self.cmd_vel_pub.publish(twist_msg)

            # Visualization
            cv2.putText(output_frame, f"Decision: {decision}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if self.show_segmentation:
                mask = get_color_pallete(self.last_pred, 'citys', pil=True)
                output_frame = cv2.addWeighted(output_frame, 1, cv2.cvtColor(np.array(mask.convert('RGB')), cv2.COLOR_RGB2BGR), 0.4, 0)
            if self.detect_objects and self.last_boxes:
                for box in self.last_boxes:
                    center_x, center_y, w, h = box.astype('int')
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            self.vis_pub.publish(self.bridge.cv2_to_imgmsg(output_frame, "bgr8"))

        self.frame_count += 1
    
    def publish_safe_zone_cloud(self, cost_map, header):
        # Create a point cloud of the safe-to-drive areas
        # For simplicity, we'll create a flat point cloud.
        # A more advanced implementation would project this into 3D space.
        safe_pixels = np.argwhere(cost_map == 0)
        
        points = []
        for p in safe_pixels:
            # We'll put all points on the z=0 plane
            # The x and y are normalized to be between -1 and 1
            x = (p[1] / float(cost_map.shape[1])) * 2 - 1
            y = (p[0] / float(cost_map.shape[0])) * 2 - 1
            z = 0
            points.append([x, y, z])

        if not points: return

        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.point_cloud_pub.publish(cloud_msg)

if __name__ == '__main__':
    try:
        navigator = SemanticNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 