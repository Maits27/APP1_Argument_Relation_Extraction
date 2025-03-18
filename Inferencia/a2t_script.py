from a2t.data import Dataset
from a2t.tasks import RelationClassificationTask, RelationClassificationFeatures
from a2t.base import EntailmentClassifier
import numpy as np
import os
import json


params_file_path = "/home/shared/esperimentuak/maitaneCasimedicos/src/params/inferencia_params.json"
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
    # additional_variables=["inst_type"],
    labels=labels,
    templates=templates,
    # valid_conditions=valid_conditions,
    negative_label_id=script_params['negative_label_id'], # indice en labels de la label "no-relation"
    multi_label=True,
    features_class=RelationClassificationFeatures,
)


nlp = EntailmentClassifier(
    # "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    # "/home/shared/esperimentuak/maitaneCasimedicos/modelos/14",
    # "/home/shared/esperimentuak/maitaneCasimedicos/src/output/0",
    script_params['model_path'],
    use_tqdm=True,
    # use_cuda=False, # CUANDO HAGAS TEST SIN ESTO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    half=True
)

# test_examples = [
#     RelationClassificationFeatures(X='Treatment is with thyroxine', Y='Treat with L-Tyroxine and determine antithyroid antibodies', context='QUESTION TYPE: ENDOCRINOLOGY\nCLINICAL CASE: \nA 24-year-old man reports progressive asthenia for at least 6 months, hoarse voice, slow speech, drowsiness and swelling of the hands, feet and face. Examination: pulse at 52 beats per minute, abotarada face and dry and pale skin. CBC: mild anemia, cholesterol 385 mg/dL (normal <220), creatinine 1.3 mg/dL (normal: 0.5-1.1), negative proteinuria, TSH 187μIU/mL (normal 0.35-5.5) and Free T4 0.2 ng/dL (normal 0.85-1.86). What strategy do you think is most appropriate?\nCORRECT ANSWER: 4\nThis is severe primary hypothyroidism. Treatment is with thyroxine. If there is no nodule, neither echo nor FNA is indicated.\n', label='support'),
#     RelationClassificationFeatures(X='Perform a thyroid ultrasound before starting treatment', Y='If there is no nodule, neither echo nor FNA is indicated', context='QUESTION TYPE: ENDOCRINOLOGY\nCLINICAL CASE: \nA 24-year-old man reports progressive asthenia for at least 6 months, hoarse voice, slow speech, drowsiness and swelling of the hands, feet and face. Examination: pulse at 52 beats per minute, abotarada face and dry and pale skin. CBC: mild anemia, cholesterol 385 mg/dL (normal <220), creatinine 1.3 mg/dL (normal: 0.5-1.1), negative proteinuria, TSH 187μIU/mL (normal 0.35-5.5) and Free T4 0.2 ng/dL (normal 0.85-1.86). What strategy do you think is most appropriate?\nCORRECT ANSWER: 4\nThis is severe primary hypothyroidism. Treatment is with thyroxine. If there is no nodule, neither echo nor FNA is indicated.\n', label='no_relation')
# ]
# real_labels = ['support', 'no_relation']
test_examples, real_labels = generar_instancias()

predictions, p_predictions = nlp(
    task=task,
    features=test_examples,
    return_labels=True,
    return_raw_output=True,
    return_confidences=True,
    negative_threshold=script_params['negative_threshold']
) # negative_threshold=0.8


predicciones = np.argmax(p_predictions, axis=1)

# Invertir el mapeo de etiquetas a identificadores y cambiar los valores de las etiquetas
label2id = {label: idx for idx, label in enumerate(labels)}
mapeo_resultante = [label2id[label] for label in real_labels]
r = np.array(mapeo_resultante,dtype = np.int_)
metrics = task.compute_metrics(r, p_predictions, "optimize")

print(metrics)

with open(f'{script_params["output_dir"]}/results.json', 'w', encoding='utf8') as archivo_json:
    json.dump(metrics, archivo_json, ensure_ascii=False, indent=4)