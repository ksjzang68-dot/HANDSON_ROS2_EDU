import cv2
import numpy as np
import time

# ------------------------- 설정값 (환경에 맞게 조정) -------------------------
CAMERA_INDEX = 0          # /dev/video0 이 아니면 1, 2 등으로 변경
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SHOW_WINDOW = True        # 디스플레이 없는 원격 환경이면 False로 변경

# ROI(관심영역) - 화면 하단 사다리꼴, 비율(0~1)로 지정 (해상도가 바뀌어도 그대로 동작)
ROI_TOP_Y = 0.55          # ROI 윗변 y위치 (화면 세로 비율)
ROI_TOP_LEFT_X = 0.40
ROI_TOP_RIGHT_X = 0.60
ROI_BOTTOM_LEFT_X = 0.05
ROI_BOTTOM_RIGHT_X = 0.95

# Canny / Hough 파라미터
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 100

# 기울기가 너무 작으면(수평선) 노이즈로 간주하고 버림
SLOPE_MIN_THRESHOLD = 0.3

# 방향 판단 민감도: 차선 중점이 화면 중앙에서 이 픽셀 이상 벗어나야 좌/우로 판단
DIRECTION_DEADZONE_PX = 25

# 좌/우 차선 중 하나만 검출됐을 때, 화면 중앙 대비 어느 정도 치우쳐야 좌/우 판단할지
SINGLE_LANE_OFFSET_RATIO = 0.15  # 화면 폭의 15%


def make_roi_mask(frame_shape):
    """비율 기반으로 ROI 사다리꼴 마스크의 꼭짓점 좌표를 계산."""
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
    """그레이스케일 -> 블러 -> Canny 엣지."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return edges


def detect_lines(roi_edges):
    lines = cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    return lines


def separate_and_average_lines(lines, frame_shape):
    """Hough 직선들을 기울기 부호로 좌/우로 나누고 각각 하나의 대표 직선으로 평균."""
    h, w = frame_shape[:2]
    left_pts = []   # 왼쪽 차선 후보 (x1,y1,x2,y2)
    right_pts = []  # 오른쪽 차선 후보

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # 수직선은 기울기 계산 불가 -> 스킵
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < SLOPE_MIN_THRESHOLD:
            continue  # 거의 수평인 노이즈 제거

        # 이미지 좌표계: y가 아래로 갈수록 증가
        # 왼쪽 차선은 기울기 음수(좌상->우하로 갈수록 x 감소), 오른쪽은 양수
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
        # y = m*x + b 형태로 1차 피팅 (x를 y의 함수로 놓으면 수직에 가까운 선도 안정적으로 처리됨)
        poly = np.polyfit(ys, xs, 1)  # x = m*y + b
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
    """
    좌/우 차선 위치를 바탕으로 이동 방향 결정.
    반환값: (direction:str, lane_center_x:int or None)
    """
    frame_center = frame_width // 2

    if left_line is not None and right_line is not None:
        # 둘 다 검출: 두 차선 하단 x좌표의 중점 = 차선 중심
        lane_center = (left_line[0] + right_line[0]) // 2
        offset = lane_center - frame_center

        if offset > DIRECTION_DEADZONE_PX:
            return "RIGHT", lane_center
        elif offset < -DIRECTION_DEADZONE_PX:
            return "LEFT", lane_center
        else:
            return "STRAIGHT", lane_center

    elif left_line is not None:
        # 왼쪽 차선만 보임 -> 차가 오른쪽으로 붙어있는 상태 -> 왼쪽으로 조향 필요
        offset = left_line[0] - frame_center
        if offset > -frame_width * SINGLE_LANE_OFFSET_RATIO:
            return "LEFT", None
        return "STRAIGHT", None

    elif right_line is not None:
        # 오른쪽 차선만 보임 -> 오른쪽으로 조향 필요
        offset = right_line[0] - frame_center
        if offset < frame_width * SINGLE_LANE_OFFSET_RATIO:
            return "RIGHT", None
        return "STRAIGHT", None

    else:
        return "NO_LANE", None


def draw_overlay(frame, roi_vertices, left_line, right_line, direction, lane_center):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # ROI 표시
    cv2.polylines(overlay, roi_vertices, isClosed=True, color=(255, 255, 0), thickness=2)

    # 차선 표시
    if left_line is not None:
        cv2.line(overlay, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 5)
    if right_line is not None:
        cv2.line(overlay, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)

    # 화면 중앙선 (기준선)
    frame_center = w // 2
    cv2.line(overlay, (frame_center, h), (frame_center, int(h * ROI_TOP_Y)), (200, 200, 200), 1)

    # 차선 중심점
    if lane_center is not None:
        cv2.circle(overlay, (lane_center, h - 20), 8, (0, 255, 255), -1)

    # 방향 텍스트 + 화살표
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


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[에러] 카메라(index={CAMERA_INDEX})를 열 수 없습니다. /dev/video* 번호를 확인하세요.")
        return

    print("차선 검출 시작 (종료: 화면에서 'q' 키, 또는 Ctrl+C)")

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

            # FPS 계산
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
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()