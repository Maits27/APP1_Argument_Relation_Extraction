import numpy as np
from pathlib import Path

def get_label_list(p_labels):
    unique_labels = set()
    for label in p_labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def crear_directorio_dataset():
    directorio_a_crear = Path("../Datasets")
    if not directorio_a_crear.exists():
        directorio_a_crear.mkdir(parents=True)
    return directorio_a_crear

def generar_dataset(train_file):
    init_path = train_file.split('/')[-2]
    output_path = crear_directorio_dataset()

    for grupo in ['train', 'dev', 'test']:
        path = Path(f'{output_path}/{grupo}_{init_path}.json')
        if not path.exists():
            
