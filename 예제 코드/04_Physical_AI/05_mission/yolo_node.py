#!/usr/bin/env python3
"""
yolo_node.py
============
역할:
  - 웹캠에서 프레임을 읽어 YOLO 추론 수행
  - 검출된 모든 박스를 JSON 직렬화하여 /agv/detection 에 publish
  - 어노테이션된 프레임을 /agv/frame (CompressedImage) 에 publish

Subscribe : 없음
Publish   :
  /agv/detection  (std_msgs/String)       ← JSON 직렬화된 박스 목록
  /agv/frame      (sensor_msgs/CompressedImage) ← 어노테이션 프레임 (JPEG)

JSON 구조 (/agv/detection):
  {
    "all_boxes": [
      {"conf": 0.91, "xyxy": [x1,y1,x2,y2], "label": "can"},
      ...
    ],
    "stamp": 1700000000.123   ← time.time()
  }
"""

import json
import time
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ai.pt 기본 경로 — 이 파일과 같은 폴더
_DEFAULT_MODEL_PATH = "/home/jetson/agv_ws/build/agv_tracker/agv_tracker/ai.pt"
from agv_tracker.config import (
    MODEL_PATH, CAM_INDEX, CAM_W, CAM_H,
    CONF_THRESHOLD, CLOSE_Y_THRESH_MAP,
    CLOSE_Y_THRESH_DEFAULT, DEAD_ZONE,
    STEER_GAIN, MAX_STEER,
)


class YoloNode(Node):
    """YOLO 추론 + 카메라 프레임 publish 노드."""

    def __init__(self):
        super().__init__("yolo_node")

        # ── 파라미터 선언 (launch 파일에서 override 가능) ──────────
        self.declare_parameter("model_path",    _DEFAULT_MODEL_PATH)
        self.declare_parameter("cam_index",     CAM_INDEX)
        self.declare_parameter("cam_w",         CAM_W)
        self.declare_parameter("cam_h",         CAM_H)
        self.declare_parameter("conf_threshold", CONF_THRESHOLD)

        model_path     = self.get_parameter("model_path").value
        cam_index      = self.get_parameter("cam_index").value
        self.cam_w     = self.get_parameter("cam_w").value
        self.cam_h     = self.get_parameter("cam_h").value
        conf_threshold = self.get_parameter("conf_threshold").value

        # ── Publisher ─────────────────────────────────────────────
        self.pub_det   = self.create_publisher(String,           "/agv/detection", 10)
        self.pub_frame = self.create_publisher(CompressedImage,  "/agv/frame",     10)

        # ── YOLO 모델 로드 ─────────────────────────────────────────
        self.get_logger().info(f"YOLO 모델 로드 중: {model_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.get_logger().info("YOLO 모델 로드 완료")
        except Exception as e:
            self.get_logger().error(f"YOLO 로드 실패: {e}")
            self.model = None

        # ── 카메라 열기 ───────────────────────────────────────────
        self.cap = None
        for idx in [cam_index, 0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                self.cap = cap
                self.get_logger().info(f"카메라 열기 성공: /dev/video{idx}")
                break
            cap.release()

        if self.cap is None:
            self.get_logger().error("카메라를 열 수 없습니다!")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(f"실제 카메라 해상도: {actual_w}x{actual_h}")

        # ── 추론 타이머 (30 Hz) ───────────────────────────────────
        self.create_timer(1.0 / 30.0, self._inference_cb)

    # ──────────────────────────────────────────────────────────────
    def _inference_cb(self):
        """매 타이머 콜백: 프레임 읽기 → 추론 → publish."""
        if self.cap is None or self.model is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # 해상도 강제 통일
        if frame.shape[1] != self.cam_w or frame.shape[0] != self.cam_h:
            frame = cv2.resize(frame, (self.cam_w, self.cam_h))

        # ── YOLO 추론 ─────────────────────────────────────────────
        results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)

        all_boxes = []
        for r in results:
            for box in r.boxes:
                conf  = float(box.conf[0])
                xyxy  = box.xyxy[0].cpu().numpy().astype(int)
                label = self.model.names[int(box.cls[0])]
                x1 = int(np.clip(xyxy[0], 0, self.cam_w - 1))
                y1 = int(np.clip(xyxy[1], 0, self.cam_h - 1))
                x2 = int(np.clip(xyxy[2], 0, self.cam_w - 1))
                y2 = int(np.clip(xyxy[3], 0, self.cam_h - 1))
                all_boxes.append({"conf": conf, "xyxy": [x1, y1, x2, y2], "label": label})

        # ── /agv/detection publish ────────────────────────────────
        payload = {"all_boxes": all_boxes, "stamp": time.time()}
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_det.publish(msg)

        # ── 어노테이션 프레임 생성 + /agv/frame publish ────────────
        annotated = self._annotate(frame, all_boxes)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_msg = CompressedImage()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.format = "jpeg"
        img_msg.data   = buf.tobytes()
        self.pub_frame.publish(img_msg)

    # ──────────────────────────────────────────────────────────────
    def _annotate(self, frame: np.ndarray, all_boxes: list) -> np.ndarray:
        """박스 + 가이드라인 어노테이션."""
        annotated  = frame.copy()
        cx_center  = self.cam_w // 2

        # 가이드선 — 수직 중앙선
        cv2.line(annotated, (cx_center, 0), (cx_center, self.cam_h), (0, 255, 255), 1)

        # 물체별 close_y 가이드선 (색상 구분)
        label_colors = {
            "can":     (255,  50,  50),
            "paper":   ( 50, 255,  50),
            "plastic": ( 50,  50, 255),
        }
        for lbl_key, y_thresh in CLOSE_Y_THRESH_MAP.items():
            color = label_colors.get(lbl_key, (200, 200, 200))
            cv2.line(annotated, (0, y_thresh), (self.cam_w, y_thresh), color, 1)
            cv2.putText(annotated, f"{lbl_key} y={y_thresh}",
                        (4, y_thresh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 박스
        for b in all_boxes:
            x1, y1, x2, y2 = b["xyxy"]
            label = b["label"]
            conf  = b["conf"]
            bcx   = (x1 + x2) // 2
            bcy   = (y1 + y2) // 2
            error = bcx - cx_center

            # 해당 label의 close_y 기준
            close_y = CLOSE_Y_THRESH_MAP.get(label.strip().lower(), CLOSE_Y_THRESH_DEFAULT)

            # y2 가장 큰 것(가장 가까운 것) 강조
            is_closest = all(b["xyxy"][3] >= other["xyxy"][3] for other in all_boxes)
            color     = (0, 220, 255) if is_closest else (80, 80, 80)
            thickness = 2 if is_closest else 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(annotated, (bcx, bcy), 5, (255, 107, 53), -1)

            steer = 0.0
            if abs(error) >= DEAD_ZONE:
                steer = max(-MAX_STEER, min(MAX_STEER, STEER_GAIN * error))

            # cy가 close_y 넘으면 CLOSE! 표시
            close_tag = " [CLOSE!]" if bcy > close_y else ""
            txt = f"{label} {conf:.2f} | err:{error:+d} | cy:{bcy}{close_tag}"
            cv2.putText(annotated, txt, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return annotated
    # ──────────────────────────────────────────────────────────────
    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()


# ──────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
