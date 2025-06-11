from ultralytics import YOLO


model = YOLO("yolo11n.yaml")


train_results = model.train(
data="coco.yaml", # path to dataset YAML
epochs=600, # number of training epochs
imgsz=640, # training image size
device="mps",
)
