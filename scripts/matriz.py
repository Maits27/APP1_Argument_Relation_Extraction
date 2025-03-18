import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

init = "/gaueko1/users/murruela002/APP1/NLIsrc/output"
initPath = Path(f'{init}/todasLasMetricas.json')
with open(initPath, 'r', encoding='utf-8') as file:
    data_all = json.load(file)
for data in data_all:
    plt.figure()
    nombre = data['Name_of_experiment']
    nombre_barra = nombre.replace(' ', '_')
    matriz_confusion = data['eval_confusion_matrix']
    # Convertir la lista de listas en un array de numpy
    matriz_confusion_np = np.array(matriz_confusion)

    # Visualizar la matriz de confusión
    plt.imshow(matriz_confusion_np, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    # Añadir etiquetas a los ejes
    tick_marks = np.arange(len(matriz_confusion))
    plt.xticks(tick_marks, range(len(matriz_confusion)))
    plt.yticks(tick_marks, range(len(matriz_confusion)))

    # Añadir los valores de la matriz en cada celda
    for i in range(len(matriz_confusion)):
        for j in range(len(matriz_confusion)):
            plt.text(j, i, str(matriz_confusion[i][j]), horizontalalignment='center', verticalalignment='center', color='black')

    # Añadir etiquetas a los ejes x e y
    plt.xlabel('Clase predicha')
    plt.ylabel('Clase verdadera')
    plt.title(nombre)

    # Mostrar la figura
    plt.savefig(f'{init}/{nombre_barra}.png')
