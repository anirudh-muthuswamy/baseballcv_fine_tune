import optuna
import yaml
from ultralytics import YOLO
from utils import get_torch_device
import argparse
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
import cv2
DEVICE = get_torch_device()

#Defining objective function for optuma for hyperparameter optimization 
#It takes in trial object, dataset path, and model path as parameters
#It returns the validation accuracy of the model trained with the hyperparameters suggested by the trial object

def create_glove_focused_data_config():
    """Create training configuration optimized for glove detection"""
    return {
        'imgsz': 800,

        'hsv_h': 0.02,      # more hue variation
        'hsv_s': 0.8,       # higher saturation variation
        'hsv_v': 0.5,       # more brightness variation
        
        'degrees': 15.0,     # more rotation for glove orientations
        'translate': 0.15,   # higher translation
        'scale': 0.7,        # more scale variation
        'shear': 5.0,        # more shear for glove deformation
        'perspective': 0.0005, # slight change in perspective
        
        'fliplr': 0.3,       # increased horizontal flip
        'flipud': 0.1,       # added vertical flip for catching positions
        
        'mosaic': 0.6,       # increase mosaic for context
        'close_mosaic': 10,  # close mosaic later for better convergence
        'multi_scale': True,  # enable multi-scale training
    }


def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        self.contains_spatial = True 
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A         
            T = [
                # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomRain(p=0.02),
                A.RandomSunFlare(p=0.02),
                A.RandomShadow(p=0.02),
                A.OneOf([A.MotionBlur(blur_limit=5, p=0.1),
                         A.GaussianBlur(blur_limit=3, p=0.1),
                         A.MedianBlur(blur_limit=3, p=0.1)
                        ], p=0.3),
                A.CoarseDropout(num_holes_range=[1, 2],hole_height_range=[0.01, 0.05],hole_width_range=[0.01, 0.05],
                                fill=100),
                A.CLAHE(p=0.05),
            ]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")


def objective(trial, data_args, dataset_path='baseball_data.yaml', model_path='glove_tracking_v4_YOLOv11.pt',
              epochs=5):
    
    # Define the hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [4, 8])
    optimizer = trial.suggest_categorical('optimizer', ['AdamW', 'SGD', 'Adam'])
    dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4])

    print(f"Trial {trial.number} hyperparameters: lr={lr}, batch_size={batch_size}, optimizer={optimizer}, dropout={dropout}")

    # Load pretrained YOLO model
    model = YOLO(model_path).to(device=DEVICE)
    print('Pretrained model loaded')

    # Train YOLO model with the trial's hyperparameters for 5 epochs
    model.train(data=dataset_path,
                epochs=epochs,
                batch=batch_size,
                optimizer=optimizer,
                lr0=lr,
                dropout=dropout,
                plots=True,
                val=True, 
                device=DEVICE,
                augment = True,
                name = f'trial_{trial.number}',  # Name the run for clarity,
                **data_args
                )

    # Evaluate the model and return validation accuracy
    metrics = model.val()
    return metrics.box.map  # Maximize accuracy (mean average precision)


def optimize(n_trials):
    # Run Optuna study with n_trials
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, create_glove_focused_data_config()), n_trials=n_trials)

    # Get the best trial's hyperparameters
    best_params = study.best_trial.params
    print('Best hyperparams found:', best_params)
    #save params as yaml file
    with open('best_hyperparams.yaml', 'w') as f:
        yaml.dump(best_params, f)
    return best_params

def fine_tune_model(data_args, model_path='glove_tracking_v4_YOLOv11.pt', dataset_path='baseball_data.yaml', fine_tune_epochs=25, 
                    best_params=None, freeze_layers = 8, run_name='fine_tuned_run'):
    
    model = YOLO(model_path).to(device=DEVICE)
    print('Pretrained model loaded')

    # Train the model with the best hyperparameters or default parameters
    model.train(data=dataset_path,
                epochs=fine_tune_epochs,
                batch=best_params['batch_size'] if best_params else 4,  # Use best batch size or default
                optimizer=best_params['optimizer'] if best_params else 'auto',
                lr0=best_params['lr'] if best_params else 0.001,
                dropout=best_params['dropout'] if best_params else 0.2,
                warmup_epochs=0,
                plots=True,
                val=True,
                device=DEVICE,
                augment=True,
                freeze=freeze_layers,  # Freeze specified layers
                name=run_name,  # Name the run for clarity
                **data_args
                )
    print(f'Model trained for {fine_tune_epochs} epochs with hyperparameters: {best_params if best_params else "default"}')
    
    # Save the fine-tuned model
    model.save('fine_tuned_glove_tracking_YOLOv11.pt')
    print('Fine-tuned model saved as fine_tuned_glove_tracking_YOLOv11.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune YOLO model with optional hyperparameter optimization.')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of Optuna trials for hyperparameter optimization')
    parser.add_argument('--optimize_hyperpaterameters', action='store_true', help='Set this flag to optimize hyperparameters')
    parser.add_argument('--model_path', type=str, default='glove_tracking_v4_YOLOv11.pt', help='Path to the pretrained model')
    parser.add_argument('--dataset_path', type=str, default='baseball_data.yaml', help='Path to the dataset YAML file')
    parser.add_argument('--fine_tune_epochs', type=int, default=25, help='Number of epochs for fine-tuning')
    parser.add_argument('--freeze_layers', type=int, default=8, help='set layers to freeze during fine-tuning, e.g., 10 for freezing first 10 layers')
    parser.add_argument('--modify_albumentation', action='store_true',default=True, help='change default albumentations for additional data augmentation')

    args = parser.parse_args()
    data_config = create_glove_focused_data_config()

    n_trials = args.n_trials
    optimize_hyperpaterameters = args.optimize_hyperpaterameters
    model_path = args.model_path
    dataset_path = args.dataset_path
    fine_tune_epochs = args.fine_tune_epochs
    freeze_layers = args.freeze_layers
    modify_albumentation = args.modify_albumentation

    if modify_albumentation:
        print('Modifying albumentations for additional data augmentation...')
        Albumentations.__init__ = __init__

    if optimize_hyperpaterameters:
        print(f'Starting hyperparameter optimization with {n_trials} trials...')
        best_params = optimize(n_trials)
        print(f'Best hyperparameters after {n_trials} trials: {best_params}')
    else:
        print('Skipping hyperparameter optimization, using default parameters.')
        best_params = None # use default parameters of the YOLO model

    # fine tune model
    fine_tune_model(data_args=data_config, model_path=model_path, dataset_path=dataset_path, 
                    fine_tune_epochs=fine_tune_epochs, 
                    best_params=best_params, freeze_layers=freeze_layers, 
                    run_name='fine_tuned_run')