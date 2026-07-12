from HandsON_BuildHat_API import MotorPair
import cv2
import numpy as np
import time

# ------------------------- 설정값 (환경에 맞게 조정) -------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SHOW_WINDOW = True

ROI_TOP_Y = 0.55
ROI_TOP_LEFT_X = 0.40
ROI_TOP_RIGHT_X = 0.60
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
DIRECTION_DEADZONE_PX = 25
SINGLE_LANE_OFFSET_RATIO = 0.15

BASE_SPEED = 30
TURN_BOOST = 15  # 회전 시 안쪽/바깥쪽 바퀴 속도 차이

# ----------------------------------------------------------
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


def decide_direction(left_line, right_line, frame_width):
    frame_center = frame_width // 2

    if left_line is not None and right_line is not None:
        lane_center = (left_line[0] + right_line[0]) // 2
        offset = lane_center - frame_center

        if offset > DIRECTION_DEADZONE_PX:
            return "RIGHT", lane_center
        elif offset < -DIRECTION_DEADZONE_PX:
            return "LEFT", lane_center
        else:
            return "STRAIGHT", lane_center

    elif left_line is not None:
        offset = left_line[0] - frame_center
        if offset > -frame_width * SINGLE_LANE_OFFSET_RATIO:
            return "LEFT", None
        return "STRAIGHT", None

    elif right_line is not None:
        offset = right_line[0] - frame_center
        if offset < frame_width * SINGLE_LANE_OFFSET_RATIO:
            return "RIGHT", None
        return "STRAIGHT", None

    else:
        return "NO_LANE", None


def draw_overlay(frame, roi_vertices, left_line, right_line, direction, lane_center):
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

    color_map = {
        "LEFT": (0, 0, 255),
        "RIGHT": (0, 165, 255),
        "STRAIGHT": (0, 255, 0),
        "NO_LANE": (128, 128, 128),
    }
    color = color_map.get(direction, (255, 255, 255))
    cv2.putText(overlay, f"DIRECTION: {direction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    arrow_y = 90
    center_x = w // 2
    if direction == "LEFT":
        cv2.arrowedLine(overlay, (center_x + 40, arrow_y), (center_x - 40, arrow_y), color, 4, tipLength=0.4)
    elif direction == "RIGHT":
        cv2.arrowedLine(overlay, (center_x - 40, arrow_y), (center_x + 40, arrow_y), color, 4, tipLength=0.4)
    elif direction == "STRAIGHT":
        cv2.arrowedLine(overlay, (center_x, arrow_y + 30), (center_x, arrow_y - 30), color, 4, tipLength=0.4)

    return overlay


def motor_moves(direction):
    if direction == "LEFT":
        robot.start(BASE_SPEED, BASE_SPEED + TURN_BOOST)   # 왼쪽으로 회전
    elif direction == "RIGHT":
        robot.start(BASE_SPEED + TURN_BOOST, BASE_SPEED)   # 오른쪽으로 회전
    elif direction == "STRAIGHT":
        robot.start(BASE_SPEED, BASE_SPEED)                # 직진
    else:
        robot.stop()                                       # 차선 없음 -> 정지


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[에러] 카메라(index={CAMERA_INDEX})를 열 수 없습니다. /dev/video* 번호를 확인하세요.")
        return

    print("차선 검출 + 모터 제어 시작 (종료: 화면에서 'q' 키, 또는 Ctrl+C)")

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
            direction, lane_center = decide_direction(left_line, right_line, frame.shape[1])

            # ---- 여기가 누락돼 있던 부분: 실제로 모터에 방향을 반영 ----
            motor_moves(direction)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            print(f"[FPS {fps:5.1f}] direction = {direction}"
                  + (f", lane_center_x = {lane_center}" if lane_center is not None else ""))

            if SHOW_WINDOW:
                result = draw_overlay(frame, roi_vertices, left_line, right_line, direction, lane_center)
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