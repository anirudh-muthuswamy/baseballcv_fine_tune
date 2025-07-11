# This step involves creating the data.yaml file and organizing the dataset folders.
# The file system should look like this:

# /path/to/dataset/
#   ├── train/
#   │   ├── images/
#   │   ├── labels/
#   ├── valid/
#   │   ├── images/
#   │   ├── labels/
#   ├── test/
#   │   ├── images/
#   │   ├── labels/
#   └── baseball_data.yaml

import yaml 
import os
from IPython.display import display

def create_data_yaml(dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    data = {'train' :  f'{dataset_path}/train/images',
            'val' :  f'{dataset_path}/valid/images',
            'test' :  f'{dataset_path}/test/images',
            'nc': 4,
            'names': ['glove','homeplate','baseball','rubber'],
            }

    # overwrite the data to the .yaml file
    with open(f'baseball_data.yaml', 'w') as f:
        yaml.dump(data, f)

    # read the content in .yaml file
    with open(f'baseball_data.yaml', 'r') as f:
        hamster_yaml = yaml.safe_load(f)
        print("Data YAML content:")
        display(hamster_yaml)


# There is majorly only 4 classes (class ids: 0, 1, 2, 3) in this dataset (glove, homeplate, baseball, rubber)
# remove class id: 4 that is not used in the dataset

# Function to remove lines with class 4
def filter_labels(label_dir):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as file:
            lines = file.readlines()

        # Filter out annotations for class 4
        filtered_lines = [line for line in lines if not line.startswith('4')]

        # Rewrite the label file
        with open(label_path, 'w') as file:
            file.writelines(filtered_lines)

if __name__ == "__main__":
    dataset_path = "baseball_rubber_home_glove"

    create_data_yaml(dataset_path)
    
    label_dirs = [
    f'{dataset_path}/train/labels',
    f'{dataset_path}/valid/labels',
    f'{dataset_path}/test/labels'
    ]

    # Process all label directories
    for label_dir in label_dirs:
        filter_labels(label_dir)

    print("Filtered labels to remove class 4 annotations.")