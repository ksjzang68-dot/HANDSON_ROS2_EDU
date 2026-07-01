"""
YOLO Node
=========
역할
  - 카메라 프레임 캡처
  - YOLO 추론 → 객체 감지
  - rank=1(가장 가까운 = 가장 큰 객체) 기준으로 랭킹 부여
  - /yolo/image_raw    CompressedImage 발행  (gui_node 용)
  - /yolo/detections   String(JSON)   발행  (motor_node + gui_node 용)

랭킹 기준
  - 바운딩박스 면적(pixel²) 내림차순 → 크면 클수록 가깝다고 간주
  - 동일 면적이면 동일 rank

토픽
  발행: /yolo/image_raw   (sensor_msgs/CompressedImage)
        /yolo/detections  (std_msgs/String  JSON array)

JSON 스키마 (1개 원소 예시)
  {
    "rank": 1,
    "label": "person",
    "conf": 0.92,
    "x1": 120, "y1": 80, "x2": 520, "y2": 700,
    "area": 193600,
    "box_h": 620,        ← 박스 높이(px)  motor_node 에서 정지 판정에 사용
    "cx": 320            ← 박스 중심 x(px) motor_node 에서 조향에 사용
  }
"""

import json
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ── 더미 감지 (YOLO 없을 때 시뮬레이션) ─────────────────────────

class _DummyDetection:
    """ultralytics Box 와 동일한 인터페이스"""
    def __init__(self, x1, y1, x2, y2, conf, cls_id):
        import numpy as np
        self.xyxy = [np.array([x1, y1, x2, y2])]
        self.conf = [conf]
        self.cls  = [cls_id]


class CameraYoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        # ── 파라미터 ──────────────────────────────────────────────
        self.declare_parameter('model_path',   '/home/jetson/Desktop/ai.pt')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('conf_thresh',  0.5)
        self.declare_parameter('fps',          30)

        model_path   = self.get_parameter('model_path').value
        camera_index = self.get_parameter('camera_index').value
        self.conf    = self.get_parameter('conf_thresh').value
        fps          = self.get_parameter('fps').value

        # ── YOLO 모델 ─────────────────────────────────────────────
        if YOLO_AVAILABLE:
            self.get_logger().info(f'YOLO 모델 로드: {model_path}')
            self.model       = YOLO(model_path)
            self.class_names = self.model.names
        else:
            self.model       = None
            self.class_names = {0: 'object'}
            self.get_logger().warn('ultralytics 없음 → 시뮬레이션 감지 사용')

        # ── 카메라 ────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().fatal('카메라를 열 수 없습니다!')
            raise RuntimeError('Camera open failed')

        self._frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f'카메라 해상도: {self._frame_w}×{self._frame_h}')

        # ── 퍼블리셔 ─────────────────────────────────────────────
        self.img_pub = self.create_publisher(
            CompressedImage, '/yolo/image_raw',  10)
        self.det_pub = self.create_publisher(
            String,          '/yolo/detections', 10)

        self.create_timer(1.0 / fps, self._timer_cb)
        self.get_logger().info('yolo_node 시작!')

    # ── 타이머 콜백 ───────────────────────────────────────────────

    def _timer_cb(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('프레임 읽기 실패')
            return

        # YOLO 추론
        if self.model:
            results    = self.model.predict(
                source=frame, conf=self.conf, verbose=False)
            detections = self._parse_and_rank(results, frame)
        else:
            detections = self._sim_detections(frame)

        # ── 이미지 발행 ──────────────────────────────────────────
        _, buf  = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_msg                  = CompressedImage()
        img_msg.header.stamp     = self.get_clock().now().to_msg()
        img_msg.header.frame_id  = 'camera'
        img_msg.format           = 'jpeg'
        img_msg.data             = buf.tobytes()
        self.img_pub.publish(img_msg)

        # ── 감지 결과 발행 ───────────────────────────────────────
        det_msg      = String()
        det_msg.data = json.dumps(detections, ensure_ascii=False)
        self.det_pub.publish(det_msg)

    # ── 파싱 + 랭킹 ──────────────────────────────────────────────

    def _parse_and_rank(self, results, frame) -> list:
        fh, fw = frame.shape[:2]
        items  = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, min(fw, x1))
                x2 = max(0, min(fw, x2))
                y1 = max(0, min(fh, y1))
                y2 = max(0, min(fh, y2))
                items.append({
                    'x1':    x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'area':  (x2 - x1) * (y2 - y1),
                    'box_h': y2 - y1,
                    'cx':    (x1 + x2) // 2,
                    'conf':  round(conf, 4),
                    'label': self.class_names.get(int(box.cls[0]), 'unknown'),
                    'frame_h': fh,
                    'frame_w': fw,
                })

        # 면적 내림차순 → rank 부여
        items.sort(key=lambda d: d['area'], reverse=True)
        prev_area, prev_rank = None, 0
        for i, d in enumerate(items):
            if d['area'] != prev_area:
                prev_rank = i + 1
            d['rank']  = prev_rank
            prev_area  = d['area']

        return items

    def _sim_detections(self, frame) -> list:
        """YOLO 없을 때 화면 중앙에 가상 객체 1개 반환"""
        fh, fw = frame.shape[:2]
        w, h   = fw // 3, fh // 3
        x1 = fw // 2 - w // 2
        y1 = fh // 2 - h // 2
        return [{
            'rank':    1,
            'label':   'sim_object',
            'conf':    0.99,
            'x1': x1, 'y1': y1, 'x2': x1+w, 'y2': y1+h,
            'area':    w * h,
            'box_h':   h,
            'cx':      fw // 2,
            'frame_h': fh,
            'frame_w': fw,
        }]

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


if __name__ == '__main__':
    main()
