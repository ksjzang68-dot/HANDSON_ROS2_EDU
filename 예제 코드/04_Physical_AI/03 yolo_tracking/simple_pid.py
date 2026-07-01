from HandsON_BuildHat_API import MotorPair
from ultralytics import YOLO
import cv2
import time

# ---------------- 설정값 ----------------
MODEL_PATH   = "best.pt"   # 사용할 YOLO 모델 (경량 모델 권장: yolov8n)
TARGET_CLASS = 0              # 추적할 객체 클래스 id (COCO 기준 0 = person)
CAM_INDEX    = 0              # 카메라 장치 번호 (CSI 카메라면 gstreamer 파이프라인 필요)
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240

BASE_SPEED = 30                # 기본 전진 속도
MAX_SPEED  = 60                # 모터 속도 상한(절대값)

# PID 게인 (환경에 맞게 튜닝 필요)
Kp = 0.15
Ki = 0.0
Kd = 0.05

LOST_TARGET_TIMEOUT = 1.0      # 이 시간(초) 이상 타겟을 못 찾으면 정지

# ---------------- 초기화 ----------------
robot = MotorPair('E', 'F')
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

frame_center_x = FRAME_WIDTH // 2

prev_error = 0.0
integral = 0.0
prev_time = time.time()
last_seen_time = time.time()


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


print("YOLO 기반 PID 조향 주행을 시작합니다. (종료: q)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        results = model(frame, verbose=False)[0]

        # 신뢰도가 가장 높은 target 클래스 박스 선택
        target_box = None
        best_conf = 0.0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == TARGET_CLASS and conf > best_conf:
                best_conf = conf
                target_box = box

        now = time.time()
        dt = now - prev_time if now - prev_time > 0 else 1e-3

        if target_box is not None:
            last_seen_time = now

            x1, y1, x2, y2 = target_box.xyxy[0]
            cx = int((x1 + x2) / 2)

            # 화면 중심 대비 오차 (양수: 물체가 오른쪽, 음수: 왼쪽)
            error = cx - frame_center_x

            integral += error * dt
            derivative = (error - prev_error) / dt
            correction = Kp * error + Ki * integral + Kd * derivative

            prev_error = error

            # 오차가 오른쪽(+)이면 왼쪽 모터를 더 빠르게 돌려 우회전
            left_speed = clamp(BASE_SPEED + correction, -MAX_SPEED, MAX_SPEED)
            right_speed = clamp(BASE_SPEED - correction, -MAX_SPEED, MAX_SPEED)

            robot.start_tank(int(left_speed), int(right_speed))

            print(f"cx={cx:4d}  error={error:5.1f}  corr={correction:6.2f}  "
                  f"L={left_speed:5.1f} R={right_speed:5.1f}")

        else:
            # 일정 시간 이상 타겟을 못 찾으면 정지 및 적분항 초기화
            if now - last_seen_time > LOST_TARGET_TIMEOUT:
                robot.stop()
                integral = 0.0
                prev_error = 0.0
                print("타겟 미검출 - 정지")

        prev_time = now

        # 디버그용 화면 표시 (필요 없으면 주석 처리)
        annotated = results.plot()
        cv2.imshow("YOLO PID Steering", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    robot.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("종료되었습니다.")