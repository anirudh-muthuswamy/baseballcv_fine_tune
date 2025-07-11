from ultralytics import YOLO
from utils import get_torch_device

device = get_torch_device()

def get_metrics(model_path, dataset_path, split='test'):

    model = YOLO(model_path).to(device)
    results = model.val(data=dataset_path, split=split, device=device)
    
    # Extract and return the metrics
    metrics = results.pandas().xywh
    return metrics


if __name__ == "__main__":
    model_path = 'glove_tracking_v4_YOLOv11.pt'
    dataset_path = 'baseball_data.yaml'
    
    baseline_metrics = get_metrics(model_path=model_path, dataset_path=dataset_path)
    print("Baseline Model Performance:")
    print(f"mAP@0.5:0.95: {baseline_metrics.box.map}")
    print(f"mAP@0.5: {baseline_metrics.box.map50}")
    print(baseline_metrics)