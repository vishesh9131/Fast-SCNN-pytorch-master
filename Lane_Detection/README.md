# Lane and Object Detection

[![Build Status](https://img.shields.io/travis/com/your-username/Lane-Detection-master.svg?style=for-the-badge)](https://travis-ci.com/your-username/Lane-Detection-master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project performs real-time lane detection and object detection on a video stream or from a camera feed. It uses computer vision techniques to identify lane lines on the road and a pre-trained YOLO model to detect objects like cars, pedestrians, etc.

## Project Structure

```
.
├── .gitignore
├── LICENSE
├── README.md
├── coco.names
├── lane_detection_version.py
├── requirements.txt
└── utils.py
```

## Prerequisites

*   Python 3.x
*   pip

## Dependencies

*   [NumPy](https://numpy.org/)
*   [OpenCV](https://opencv.org/)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YOUR-USERNAME]/Lane-Detection-master.git
    cd Lane-Detection-master
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Download the YOLOv3 weights and configuration file:**

    *   **YOLOv3 Weights (237 MB):**
        ```bash
        curl -O https://pjreddie.com/media/files/yolov3.weights
        ```

    *   **YOLOv3 Configuration file (8 KB):**
        ```bash
        curl -O https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
        ```

    *   **COCO Names file:**
        The `coco.names` file, which contains the names of the objects that the YOLO model can detect, is already included in this repository.

2.  **Run the lane detection script:**

    *   **On a video file:**
        ```bash
        python lane_detection_version.py --model_cfg yolov3.cfg --model_weights yolov3.weights --video /path/to/your/video.mp4 --output_dir /path/to/output/directory
        ```

    *   **From a camera feed:**
        ```bash
        python lane_detection_version.py --model_cfg yolov3.cfg --model_weights yolov3.weights --src 0 --output_dir /path/to/output/directory
        ```

## Results

Here is an example of the output of the lane detection script:

![Lane Detection Results](path/to/your/results.gif)

## Troubleshooting

*   **`wget: command not found`**: This error occurs when `wget` is not installed on your system. You can use `curl` instead, as shown in the "Usage" section.
*   **`ModuleNotFoundError: No module named 'cv2'`**: This error occurs when OpenCV is not installed correctly. You can install it by running `pip install opencv-python`.

## To-Do

*   [ ] Add support for more YOLO versions.
*   [ ] Implement a more robust lane detection algorithm.
*   [ ] Add a GUI for easier use.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

*   This project uses the YOLOv3 model, created by Joseph Redmon.
*   The lane detection algorithm is based on the concepts of computer vision and image processing.

## Contact

Created by [@your-username](https://github.com/your-username) - feel free to contact me! 