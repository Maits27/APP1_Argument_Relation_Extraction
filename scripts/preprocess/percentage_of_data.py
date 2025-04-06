import json
import random
import os

random.seed(42)

def create_a_percentage(infile, outfile, p=0.05, seq=False):
    # Cargar el archivo JSON original
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Crear carpeta para guardar el archivo
    output_folder = f'{outfile}/data_{p}'
    os.makedirs(output_folder, exist_ok=True)

    # Separar los datos por clase
    classes = {}
    for item in data:
        label = item['label']
        if label not in classes:
            classes[label] = []
        classes[label].append(item)

    sampled_data = []
    for label, items in classes.items():
        sample_size = max(1, int(p * len(items)))  # Asegura al menos 1 elemento
        sampled_data.extend(random.sample(items, sample_size))

    # Guardar el nuevo archivo JSON con la muestra estratificada
    output_path = os.path.join(output_folder, 'sequence_train_sueltas.json' if seq else 'train_sueltas.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=4, ensure_ascii=False)

    with open(f'{output_folder}/Informe.txt', 'w', encoding='utf-8') as f:
        f.write(f"Datos originales: {len(data)} instancias\n")
        f.write(f"Datos en el {p*100}%: {len(sampled_data)} instancias\n")

    # for seq, i in [(True, "datos_secuencia/"), (False, "")]:
for seq, i in [(False, "")]:
    for p in [0.05, 0.1, 0.2]:
        inicio = "sequence_" if seq else ""
        infile = f"/gaueko1/users/murruela002/APP1/NLIsrc/Datasets/{i}{inicio}train_sueltas.json"
        outfile = "/gaueko1/users/murruela002/APP1/NLIsrc/Datasets/"
        create_a_percentage(infile, outfile, p, seq)
