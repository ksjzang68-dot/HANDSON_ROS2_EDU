from ultralytics import YOLO

# YOLOv8 nano 사전 학습 모델 로드
model = YOLO('yolov8n.pt')

# 모델 학습(Training) 파라미터 설정
results = model.train(
    data='/home/jetson/Desktop/dataset/data.yaml',
    epochs=50,
    imgsz=416,
    workers=2,
    batch=2,
    device=0
)

print("학습이 모두 완료되었습니다")