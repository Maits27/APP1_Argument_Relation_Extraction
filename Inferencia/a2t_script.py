from a2t.data import Dataset
from a2t.tasks import RelationClassificationTask, RelationClassificationFeatures
from a2t.base import EntailmentClassifier
import numpy as np
import os
import json


params_file_path = "/gaueko1/users/murruela002/APP1/NLIsrc/params/inferencia_params.json"
with open(params_file_path, 'r', encoding='utf-8') as f:
    script_params = json.load(f)

if script_params['output_dir'] is not None and not os.path.exists(script_params['output_dir']) and not script_params['overwrite_output_dir']:
    os.makedirs(script_params['output_dir'])
else:
    raise ValueError(
        f"Output directory ({script_params['output_dir']}) already exists and is not empty."
        "Use --overwrite_output_dir to overcome."
    )

labels = script_params['label_names']

def generar_instancias():
    instancias = []
    real_labels = []
    with open(script_params['data_path'], 'r', encoding='utf-8') as file:
        data = json.load(file)
    for doc in data:
        contexto = doc['context']
        for relacion in doc['relations']:
            instancias.append(RelationClassificationFeatures(
                X=relacion['X'],
                Y=relacion['Y'],
                context=contexto,
                label=relacion['label']))
            real_labels = real_labels + [relacion['label']]
    return (instancias, real_labels)

templates = script_params['templates']


task = RelationClassificationTask(
    name="Relation Classification task",
    required_variables=["X", "Y"],
    labels=labels,
    templates=templates,
    negative_label_id=script_params['negative_label_id'], # indice en labels de la label "no-relation"
    multi_label=True,
    features_class=RelationClassificationFeatures,
)


nlp = EntailmentClassifier(
    script_params['model_path'],
    use_tqdm=True,
    # use_cuda=False,
    half=True
)


test_examples, real_labels = generar_instancias()

predictions, p_predictions = nlp(
    task=task,
    features=test_examples,
    return_labels=True,
    return_raw_output=True,
    return_confidences=True,
    negative_threshold=script_params['negative_threshold']
)


predicciones = np.argmax(p_predictions, axis=1)

# Invertir el mapeo de etiquetas a identificadores y cambiar los valores de las etiquetas
label2id = {label: idx for idx, label in enumerate(labels)}
mapeo_resultante = [label2id[label] for label in real_labels]
r = np.array(mapeo_resultante,dtype = np.int_)
metrics = task.compute_metrics(r, p_predictions, "optimize")

print(metrics)

with open(f'{script_params["output_dir"]}/results.json', 'w', encoding='utf8') as archivo_json:
    json.dump(metrics, archivo_json, ensure_ascii=False, indent=4)