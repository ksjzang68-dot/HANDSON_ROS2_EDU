from ultralytics import YOLO

# 학습된 커스텀 가중치 파일 로드
model = YOLO('runs/detect/train5/weights/best.pt')

# 실시간 웹캠 영상을 통한 객체 탐지 수행
results = model.predict(source=0, show=True, conf=0.6)