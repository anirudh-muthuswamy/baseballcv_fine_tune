import optuna
import yaml
from ultralytics import YOLO
from utils import get_torch_device

DEVICE = get_torch_device()

#Defining objective function for optuma for hyperparameter optimization 
#It takes in trial object, dataset path, and model path as parameters
#It returns the validation accuracy of the model trained with the hyperparameters suggested by the trial object

def objective(trial, dataset_path='baseball_data.yaml', model_path='glove_tracking_v4_YOLOv11.pt'):
    # Define the hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    optimizer = trial.suggest_categorical('optimizer', ['AdamW', 'SGD', 'Adam'])

    # Load pretrained YOLO model
    model = YOLO(model_path).to(device=DEVICE)
    print('Pretrained model loaded')

    # Train YOLO model with the trial's hyperparameters for 5 epochs
    model.train(data=dataset_path,
                epochs=1,              # Only 5 epochs for the trial
                batch=batch_size,
                imgsz=640,
                optimizer=optimizer,
                lr0=lr,
                plots=True,
                val=True, 
                device=DEVICE,
                augment = True,
                
                degrees = 10.0,
                scale=0.5,
                fliplr=0.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                shear=0.1,
                perspective=0.1
                )

    # Evaluate the model and return validation accuracy
    metrics = model.val()
    return metrics.box.map  # Maximize accuracy (mean average precision)


def optimize(n_trials):
    # Run Optuna study with n_trials
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Get the best trial's hyperparameters
    best_params = study.best_trial.params
    print('Best hyperparams found:', best_params)
    #save params as yaml file
    with open('best_hyperparams.yaml', 'w') as f:
        yaml.dump(best_params, f)
    return best_params


if __name__ == "__main__":
    n_trials = 10
    best_params = optimize(n_trials)
    print(f'Best hyperparameters after {n_trials} trials: {best_params}')