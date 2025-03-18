import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que el archivo JSON se llama 'data.json'
with open('/home/shared/esperimentuak/maitaneCasimedicos/Resultados_buenos/output_RS/todasLasMetricas.json', 'r') as file:
    data = json.load(file)

# Convertir la lista de diccionarios a un DataFrame de pandas
df = pd.DataFrame(data)

# Definir los colores para los boxplots
boxplot_colors = ['#e4d4e4', '#dcecfc', '#d4ecd7', '#faf3cc', '#e4d4e4', '#dcecfc', '#d4ecd7', '#faf3cc']
model_name_mapping = {
    'train.json': 'Neutral guztiekin',
    'train_evitando_concatenaciones.json': 'Grafoen arteko neutralak',
    'train_evitando_concatenaciones_viejo.json': 'Grafo erpinekin',
    'train_sueltas.json': 'Entitate solteekin bakarrik'
}

df['train_files'] = df['train_files'].map(model_name_mapping)

# Crear el boxplot
plt.figure(figsize=(12, 8))
box = sns.boxplot(x='train_files', y='eval_f1_ent', data=df, orient='v', palette=boxplot_colors[:len(df['train_files'].unique())])

# Añadir título y etiquetas
plt.title('Modelo bakoitzeko entailment klasearen F-Score-a random seed ezberdinekin')
plt.xlabel('Modeloa')
plt.ylabel('F1-Entailment')


# Mostrar el gráfico
plt.savefig("/home/shared/esperimentuak/maitaneCasimedicos/src/scripts/graficas/graficos/boxplot.png")
plt.show()
