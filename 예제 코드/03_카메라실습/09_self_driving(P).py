from HandsON_BuildHat_API import MotorPair
import cv2
import numpy as np
import time

# ------------------------- 설정값 (환경에 맞게 조정) -------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SHOW_WINDOW = True

ROI_TOP_Y = 0.45
ROI_TOP_LEFT_X = 0.30
ROI_TOP_RIGHT_X = 0.70
ROI_BOTTOM_LEFT_X = 0.05
ROI_BOTTOM_RIGHT_X = 0.95

CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 100

SLOPE_MIN_THRESHOLD = 0.3
SINGLE_LANE_OFFSET_RATIO = 0.15

# [P 제어 파라미터] -----------------------------------------------------------
BASE_SPEED = 40
KP = 0.3          # 비례 이득 (Proportional Gain) - 환경에 맞게 미세 조정 필요
MAX_STEER = 35     # 모터 보호를 위한 최대 조향 보정치 제한
# -------------------------------------------------------------------------

# 하드웨어 선언 (포트 E/F가 실제 배선상 좌/우 모터와 일치하는지 확인 필요)
robot = MotorPair('E', 'F')


# ----------------------------------------------------------
def make_roi_mask(frame_shape):
    h, w = frame_shape[:2]
    top_y = int(h * ROI_TOP_Y)
    vertices = np.array([[
        (int(w * ROI_BOTTOM_LEFT_X), h),
        (int(w * ROI_TOP_LEFT_X), top_y),
        (int(w * ROI_TOP_RIGHT_X), top_y),
        (int(w * ROI_BOTTOM_RIGHT_X), h),
    ]], dtype=np.int32)
    return vertices


def apply_roi(edges, vertices):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return edges


def detect_lines(roi_edges):
    return cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )


def separate_and_average_lines(lines, frame_shape):
    h, w = frame_shape[:2]
    left_pts = []
    right_pts = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < SLOPE_MIN_THRESHOLD:
            continue

        if slope < 0:
            left_pts.append((x1, y1, x2, y2))
        else:
            right_pts.append((x1, y1, x2, y2))

    def fit_average_line(points):
        if not points:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in points:
            xs += [x1, x2]
            ys += [y1, y2]
        poly = np.polyfit(ys, xs, 1)
        m, b = poly

        y_bottom = h
        y_top = int(h * ROI_TOP_Y)
        x_bottom = int(m * y_bottom + b)
        x_top = int(m * y_top + b)
        return (x_bottom, y_bottom, x_top, y_top)

    left_line = fit_average_line(left_pts)
    right_line = fit_average_line(right_pts)
    return left_line, right_line


def calculate_lane_center(left_line, right_line, frame_width):
    """
    차선 정보를 바탕으로 이상적인 차선 중심(target_x)을 계산합니다.
    차선이 감지되지 않으면 None을 반환합니다.
    """
    frame_center = frame_width // 2

    # 1. 양쪽 차선이 다 보일 때
    if left_line is not None and right_line is not None:
        return (left_line[0] + right_line[0]) // 2

    # 2. 왼쪽 차선만 보일 때
    elif left_line is not None:
        # 감지된 왼쪽 차선 바닥 위치에 가상의 반대편 차선 간격 가산
        return left_line[0] + int(frame_width * SINGLE_LANE_OFFSET_RATIO * 2)

    # 3. 오른쪽 차선만 보일 때
    elif right_line is not None:
        # 감지된 오른쪽 차선 바닥 위치에서 가상의 반대편 차선 간격 감산
        return right_line[0] - int(frame_width * SINGLE_LANE_OFFSET_RATIO * 2)

    # 4. 차선이 안 보일 때
    else:
        return None


def draw_overlay(frame, roi_vertices, left_line, right_line, lane_center, steering):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    cv2.polylines(overlay, roi_vertices, isClosed=True, color=(255, 255, 0), thickness=2)

    if left_line is not None:
        cv2.line(overlay, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 5)
    if right_line is not None:
        cv2.line(overlay, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)

    frame_center = w // 2
    cv2.line(overlay, (frame_center, h), (frame_center, int(h * ROI_TOP_Y)), (200, 200, 200), 1)

    if lane_center is not None:
        cv2.circle(overlay, (lane_center, h - 20), 8, (0, 255, 255), -1)
        # 현재 오차 상태에 따른 진행 방향 텍스트 표시
        direction_text = "STRAIGHT" if abs(steering) < 2 else ("RIGHT" if steering > 0 else "LEFT")
        cv2.putText(overlay, f"DIR: {direction_text} (Steer: {steering:.1f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(overlay, "DIRECTION: NO_LANE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

    return overlay


def motor_moves_p_control(lane_center, frame_width):
    """
    오차(Error)에 비례(P)하여 좌우 모터의 속도를 제어합니다.
    """
    if lane_center is None:
        robot.stop()
        return 0

    frame_center = frame_width // 2
    
    # 1. 오차 계산 (목표 중심 - 현재 중심)
    # lane_center가 우측에 있으면 error > 0 -> 우회전 필요
    error = lane_center - frame_center

    # 2. 제어량(Steering) 연산: 제어량 = 오차 * Kp
    steering = error * KP

    # 3. 조향 제한 (최대 속도 범위 보호)
    steering = max(min(steering, MAX_STEER), -MAX_STEER)

    # 4. 좌우 모터 속도 할당 (start_tank 방식에 맞춰 가감산 적용)
    # 조향값이 양수(우회전)이면 왼쪽 바퀴 속도가 빨라지고 오른쪽 바퀴는 느려집니다.
    left_speed = int(BASE_SPEED + steering)
    right_speed = int(BASE_SPEED - steering)

    # 5. 모터 구동
    robot.start_tank(left_speed, right_speed)
    
    return steering


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[에러] 카메라(index={CAMERA_INDEX})를 열 수 없습니다. /dev/video* 번호를 확인하세요.")
        return

    print("차선 검출 + 비례제어(P-Control) 모터 제어 시작 (종료: 'q' 키 또는 Ctrl+C)")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[경고] 프레임을 읽지 못했습니다.")
                break

            roi_vertices = make_roi_mask(frame.shape)

            edges = preprocess(frame)
            roi_edges = apply_roi(edges, roi_vertices)
            lines = detect_lines(roi_edges)
            left_line, right_line = separate_and_average_lines(lines, frame.shape)
            
            # 차선 중심 구하기
            lane_center = calculate_lane_center(left_line, right_line, frame.shape[1])

            # P 제어를 통한 모터 제어 수행 및 조향값 리턴
            steering = motor_moves_p_control(lane_center, frame.shape[1])

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            print(f"[FPS {fps:5.1f}] Center: {lane_center}, Steer: {steering:.1f}")

            if SHOW_WINDOW:
                result = draw_overlay(frame, roi_vertices, left_line, right_line, lane_center, steering)
                cv2.imshow("Lane Detection", result)
                cv2.imshow("ROI Edges", roi_edges)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n중단됨 (Ctrl+C)")

    finally:
        robot.stop()          # 종료 시 반드시 모터 정지 (안전)
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
