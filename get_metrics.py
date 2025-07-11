from ultralytics import YOLO
from utils import get_torch_device

device = get_torch_device()

def print_metrics(metrics):
    print("Class indices with average precision:", metrics.ap_class_index)
    print("Average precision for all classes:", metrics.box.all_ap)
    print("Average precision:", metrics.box.ap)
    print("Average precision at IoU=0.50:", metrics.box.ap50)
    print("Class indices for average precision:", metrics.box.ap_class_index)
    print("Class-specific metrics:", metrics.box.class_result)
    print("F1 score:", metrics.box.f1)
    print("F1 score curve:", metrics.box.f1_curve)
    print("Overall fitness score:", metrics.box.fitness)
    print("Mean average precision:", metrics.box.map)
    print("Mean average precision at IoU=0.50:", metrics.box.map50)
    print("Mean average precision at IoU=0.75:", metrics.box.map75)
    print("Mean average precision for different IoU thresholds:", metrics.box.maps)
    print("Mean results for different metrics:", metrics.box.mean_metrics)
    print("Mean precision:", metrics.box.mp)
    print("Mean recall:", metrics.box.mr)
    print("Precision:", metrics.box.p)
    print("Precision curve:", metrics.box.p_curve)
    print("Precision values:", metrics.box.prec_values)
    print("Specific precision metrics:", metrics.box.px)
    print("Recall:", metrics.box.r)
    print("Recall curve:", metrics.box.r_curve)

def get_metrics(model_path, dataset_path, split='test'):

    model = YOLO(model_path).to(device)
    results = model.val(data=dataset_path, split=split, device=device)
    
    # Extract and return the metrics
    metrics = results.pandas().xywh
    return metrics

if __name__ == "__main__":
    
    model_path = 'glove_tracking_v4_YOLOv11.pt'
    fine_tuned_model_path = 'fine_tuned_glove_tracking_YOLOv11.pt'
    dataset_path = 'baseball_data.yaml'

    # Get baseline metrics for the model
    baseline_metrics = get_metrics(model_path=model_path, dataset_path=dataset_path, name='baseline_val')
    print("Baseline Model Performance:")
    print_metrics(baseline_metrics)


    #Compare baseline metrics with fine tuned model
    fine_tuned_metrics = get_metrics(model_path=fine_tuned_model_path, dataset_path=dataset_path, name='fine_tuned_val')
    print("Fine-tuned Model Performance:")
    print_metrics(fine_tuned_metrics)


