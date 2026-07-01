import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ultralytics import YOLO

class CameraYoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        self.declare_parameter("model_path",   "ai.pt")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("conf_thresh",  0.5)
        self.declare_parameter("fps",          30)

        model_path   = "/home/jetson/Desktop/ai.pt"
        camera_index = 0
        self.conf    = self.get_parameter("conf_thresh").value
        fps          = self.get_parameter("fps").value

        self.get_logger().info(f"YOLO 모델 로드: {model_path}")
        self.model       = YOLO(model_path)
        self.class_names = self.model.names

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().fatal("카메라를 열 수 없습니다!")
            raise RuntimeError("Camera open failed")

        self.img_pub = self.create_publisher(CompressedImage, "/yolo/image_raw", 10)
        self.det_pub = self.create_publisher(String, "/yolo/detections", 10)

        self.create_timer(1.0 / fps, self.timer_callback)
        self.get_logger().info("camera_yolo_node 시작!")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("프레임 읽기 실패")
            return

        results    = self.model.predict(source=frame, conf=self.conf, verbose=False)
        detections = self._parse_and_rank(results)

        # 이미지 publish
        _, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_msg = CompressedImage()
        img_msg.header.stamp    = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera"
        img_msg.format          = "jpeg"
        img_msg.data            = buf.tobytes()
        self.img_pub.publish(img_msg)

        # 감지 결과 publish
        det_msg      = String()
        det_msg.data = json.dumps(detections, ensure_ascii=False)
        self.det_pub.publish(det_msg)

    def _parse_and_rank(self, results):
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area":  (x2 - x1) * (y2 - y1),
                    "conf":  round(conf, 4),
                    "label": self.class_names.get(int(box.cls[0]), "unknown"),
                })

        detections.sort(key=lambda d: d["area"], reverse=True)
        prev_area, prev_rank = None, 0
        for i, d in enumerate(detections):
            if d["area"] != prev_area:
                prev_rank = i + 1
            d["rank"]  = prev_rank
            prev_area  = d["area"]

        return detections

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()