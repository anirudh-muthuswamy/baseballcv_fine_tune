import optuna
import yaml
from ultralytics import YOLO
from utils import get_torch_device

DEVICE = get_torch_device()

#Defining objective function for optuma for hyperparameter optimization 
#It takes in trial object, dataset path, and model path as parameters
#It returns the validation accuracy of the model trained with the hyperparameters suggested by the trial object

DEFAULT_DATA_ARGS = {
    'degrees': 10.0,
    'scale': 0.5,
    'fliplr': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'shear': 5,
    'perspective': 0.0002,
    'mosaic': 0.4
}

def objective(trial, dataset_path='baseball_data.yaml', model_path='glove_tracking_v4_YOLOv11.pt', data_args=DEFAULT_DATA_ARGS,
              epochs=5):
    
    # Define the hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    optimizer = trial.suggest_categorical('optimizer', ['AdamW', 'SGD', 'Adam'])

    # Load pretrained YOLO model
    model = YOLO(model_path).to(device=DEVICE)
    print('Pretrained model loaded')

    # Train YOLO model with the trial's hyperparameters for 5 epochs
    model.train(data=dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=800,
                optimizer=optimizer,
                lr0=lr,
                plots=True,
                val=True, 
                device=DEVICE,
                augment = True,
                # Data augmentation parameters for increasing robustness and generalization
                degrees = data_args['degrees'],
                scale= data_args['scale'],
                fliplr= data_args['fliplr'],
                hsv_h=data_args['hsv_h'],
                hsv_s= data_args['hsv_s'],
                hsv_v= data_args['hsv_v'],
                shear= data_args['shear'],
                perspective= data_args['perspective'],
                mosaic = data_args['mosaic'],
                name = f'trial_{trial.number}'  # Name the run for clarity
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

def fine_tune_model(model_path='glove_tracking_v4_YOLOv11.pt', dataset_path='baseball_data.yaml', fine_tune_epochs=25, 
                    data_args = DEFAULT_DATA_ARGS, best_params=None, run_name='fine_tuned_run'):
    
    model = YOLO(model_path).to(device=DEVICE)
    print('Pretrained model loaded')

    # Train the model with the best hyperparameters or default parameters
    model.train(data=dataset_path,
                epochs=fine_tune_epochs,
                batch=best_params['batch_size'] if best_params else 8,  # Use best batch size or default
                imgsz=800,
                optimizer=best_params['optimizer'] if best_params else 'auto',
                lr0=best_params['lr'] if best_params else 0.001,
                plots=True,
                val=True,
                device=DEVICE,
                augment=True,
                # Data augmentation parameters for increasing robustness and generalization
                degrees = data_args['degrees'],
                scale= data_args['scale'],
                fliplr= data_args['fliplr'],
                hsv_h=data_args['hsv_h'],
                hsv_s= data_args['hsv_s'],
                hsv_v= data_args['hsv_v'],
                shear= data_args['shear'],
                perspective= data_args['perspective'],
                mosaic = data_args['mosaic'],
                name=run_name  # Name the run for clarity
                )
    print(f'Model trained for {fine_tune_epochs} epochs with hyperparameters: {best_params if best_params else "default"}')
    
    # Save the fine-tuned model
    model.save('fine_tuned_glove_tracking_YOLOv11.pt')
    print('Fine-tuned model saved as fine_tuned_glove_tracking_YOLOv11.pt')


if __name__ == "__main__":
    n_trials = 10
    optimize_hyperpaterameters = True  # Set to True to optimize hyperparameters
    model_path = 'glove_tracking_v4_YOLOv11.pt'  # Path to the pretrained model
    dataset_path = 'baseball_data.yaml'  # Path to the dataset YAML file
    fine_tune_epochs = 25

    if optimize_hyperpaterameters:
        print(f'Starting hyperparameter optimization with {n_trials} trials...')
        best_params = optimize(n_trials)
        print(f'Best hyperparameters after {n_trials} trials: {best_params}')
    else:
        print('Skipping hyperparameter optimization, using default parameters.')
        best_params = None # use default parameters of the YOLO model

    # fine tune model
    fine_tune_model(model_path=model_path, dataset_path=dataset_path, fine_tune_epochs=fine_tune_epochs, 
                    data_args=DEFAULT_DATA_ARGS, best_params=best_params, run_name='fine_tuned_run')