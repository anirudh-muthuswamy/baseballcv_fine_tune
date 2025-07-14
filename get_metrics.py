from ultralytics import YOLO
import os
from utils import get_torch_device
import argparse

DEVICE = get_torch_device()

def print_metrics(metrics):
    print("Class indices with average precision:", metrics.ap_class_index)
    print("Average precision:", metrics.box.ap)
    print("Average precision at IoU=0.50:", metrics.box.ap50)
    print("F1 score:", metrics.box.f1)
    print("Mean average precision:", metrics.box.map)
    print("Mean average precision at IoU=0.50:", metrics.box.map50)
    print("Mean average precision at IoU=0.75:", metrics.box.map75)
    print("Mean average precision for different IoU thresholds:", metrics.box.maps)
    print("Mean precision:", metrics.box.mp)
    print("Mean recall:", metrics.box.mr)
    print("Precision:", metrics.box.p)
    print("Recall:", metrics.box.r)

def get_metrics(model_path, dataset_path, name, split='test'):

    model = YOLO(model_path).to(DEVICE)
    metrics = model.val(data=dataset_path, split=split, device=DEVICE, name=name)
    
    return metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate YOLO model metrics.")
    parser.add_argument('--model_path', type=str, default='glove_tracking_v4_YOLOv11.pt', help='Path to baseline model')
    parser.add_argument('--fine_tuned_model_path', type=str, default='fine_tuned_model_results/weights/best.pt', help='Path to fine-tuned model')
    parser.add_argument('--dataset_path', type=str, default='baseball_data.yaml', help='Path to dataset YAML file')
    args = parser.parse_args()

    model_path = args.model_path
    fine_tuned_model_path = args.fine_tuned_model_path
    dataset_path = args.dataset_path

    # Get baseline metrics for the model
    baseline_metrics = get_metrics(model_path=model_path, dataset_path=dataset_path, name='baseline_val')
    print("Baseline Model Performance:")
    print_metrics(baseline_metrics)

    print("------------------------------")

    if os.path.exists(fine_tuned_model_path):
    #Compare baseline metrics with fine tuned model
        fine_tuned_metrics = get_metrics(model_path=fine_tuned_model_path, dataset_path=dataset_path, name='fine_tuned_val')
        print("Fine-tuned Model Performance:")
        print_metrics(fine_tuned_metrics)


