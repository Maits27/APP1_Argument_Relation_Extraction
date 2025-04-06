from pathlib import Path
import json

import sys
ruta_abs = ('/gaueko1/users/murruela002/APP1/NLIsrc/')
sys.path.append(ruta_abs)
# from src.Preproceso.metodos_relaciones import sacar_relaciones, no_relacionados_indirectamente, \
#     sacar_nodos_sin_ninguna_relacion, por_relacionar
from metodos_tipos_de_neutrales import generar_todas_las_neutrales, generar_neutrales_evitando_entidades_concatenadas_viejo, generar_neutrales_evitando_entidades_concatenadas, generar_neutrales_evitando_entidades_concatenadas_solo_sueltas
from metodos_auxiliares_limpieza import conseguir_opciones, opciones_de_documento_jsonl, quitar_opciones_contexto, quitar_opciones_contexto2, conseguir_id_entidades_en_relaciones, quitar_opciones_relaciones, quitar_opciones_entidades
"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STEP 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
It takes the .ann (English) y los transforma en JSON con el siguiente formato:
{
    "group": "dev",
    "files": 55,
    "data": {
        "107_206.ann": {
            "context": "QUESTION TYPE: GENETICS AND IMMUNOLOGY\nCLINICAL CASE: \nAn 8-year-old girl (index case) is clinically diagnosed as having neurofibromatosis type 1 (NF1) or Von Recklinghausen's disease with multiple neurofibromas, café-au-lait spots, and Lisch nodules. Her father (not diagnosed with NF1) died in a traffic accident at the age of 38 years. On examination, the mother presents two café-au-lait spots and attends the genetic counseling consultation with her new partner where a preimplantation genetic diagnosis (PGD) is proposed. Is PGD indicated in this case?\n1- Yes, as the mother has 2 café-au-lait spots, she is a carrier and PGD is indicated with these data.\n2- It is indicated after detection of the causative mutation in the index case and eventually in his mother.\n3- It is not indicated because NF1 responds to mutations in the neurofibromin gene (17q11.2), with recessive inheritance.\n4- No, two café-au-lait spots are not diagnostic and your new partner is very unlikely to be a carrier (NF1 is a rare disease).\n5- With these data, PGD is indicated, consisting of selecting embryos in vitro, to implant in the maternal uterus those without the mutation.\nCORRECT ANSWER: 2\nThe first step is to detect the mutation and once detected, if the mother is a carrier, to proceed with PGD.",
            "supports": {
                "count": X,
                "data": [
                    {
                        "tag_number": "R1",
                        "entity1": "Ent1:T18",
                        "entity2": "Ent2:T6"
                    }, ...
                ]
            },
            "attacks": {...},
            "claims": {...},
            "premises": {...},
            "other_tags": {...}
        }, ...
    }
}

ESTRUCTURA EN BASE AL .ANN
"""

def conseguir_datos_al_completo_paso1(ficheros, initial_data_path, split_char = '\t', split_info = ' '):
    train, dev, test = {}, {}, {}
    for f in ficheros:
        print(f)
        datos_json = {
            'group': f,
            'files': 0,
            'data': {}
        }
        initPath = Path(f'{initial_data_path}/{f}') # TODO: Path de inicio. Cambiar en caso de cambio de idioma
        for anotation_path in list(initPath.glob('*.ann')):
            txtPath = str(anotation_path).replace('.ann', '.txt')
            with open(txtPath, 'r', encoding='utf-8') as file:
                context = file.read()

            supports = []
            attacks = []
            other = []
            claims = []
            premises = []

            entidades_validas = []
            path_malo = "/proiektuak/edhia/Casimedicos_BRAT_neutralized/Annotation_and_format_conversion/dataset/neutralized_BRAT_fully_annotated/train/148_158.ann"
            with open(anotation_path, 'r', encoding='utf-8') as anotation:
                for line in anotation:
                    if line != '\n' and line != '':
                        tag_number = line.split(split_char)[0]
                        if tag_number[0] == 'R':
                            tag_number, info = line.split(split_char)[0:2]
                            tag, e1, e2 = info.split(split_info)
                            data = {'tag_number': tag_number, 'entity1': e1, 'entity2': e2}
                            if e1.split(":")[-1] in entidades_validas and e2.split(":")[-1] in entidades_validas:
                                if tag == 'Support':
                                    supports.append(data)
                                else:
                                    if anotation_path == path_malo:
                                        print(data)
                                        print(e1.split(":")[-1])
                                        print(e2.split(":")[-1])
                                        print(entidades_validas)
                                    attacks.append(data)
                        else:
                            tag_number, info, text = line.split(split_char)
                            tag, s0, s1 = info.split(split_info)
                            data = {'tag_number': tag_number, 'spans': f'{s0}-{s1}', 'text': text.replace('\n', '')}
                            if tag == 'Claim':
                                claims.append(data)
                                entidades_validas.append(tag_number)
                            elif tag == 'Premise':
                                premises.append(data)
                                entidades_validas.append(tag_number)
                            else:
                                data['tag'] = tag
                                other.append(data)

            datos_json['data'][anotation_path.name] = {
                'context': context,
                'supports': {'count': len(supports), 'data': supports},
                'attacks': {'count': len(attacks), 'data': attacks},
                'claims': {'count': len(claims), 'data': claims},
                'premises': {'count': len(premises), 'data': premises},
                'other_tags': {'count': len(other), 'data': other}
            }

        datos_json['files'] = len(datos_json['data'])
        if f == 'dev':
            dev = datos_json
        elif f == 'test':
            test = datos_json
        else:
            train = datos_json

        # with open(f'/gaueko1/users/murruela002/APP1/NLIsrc/tmp/{f}_data.json', "w", encoding='utf-8') as archivo:
        #     json.dump(datos_json, archivo, ensure_ascii=False, indent=4)
    return train, dev, test


"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STEP 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Método que recoge los .json de el paso anterior y los transforma en JSON con el siguiente formato:
{
    "group": "dev",
    "files": 55,
    "data": {
        "107_206.ann": {
            "context": "QUESTION TYPE: GENETICS AND IMMUNOLOGY\nCLINICAL CASE: \nAn 8-year-old girl (index case) is clinically diagnosed as having neurofibromatosis type 1 (NF1) or Von Recklinghausen's disease with multiple neurofibromas, café-au-lait spots, and Lisch nodules. Her father (not diagnosed with NF1) died in a traffic accident at the age of 38 years. On examination, the mother presents two café-au-lait spots and attends the genetic counseling consultation with her new partner where a preimplantation genetic diagnosis (PGD) is proposed. Is PGD indicated in this case?\n1- Yes, as the mother has 2 café-au-lait spots, she is a carrier and PGD is indicated with these data.\n2- It is indicated after detection of the causative mutation in the index case and eventually in his mother.\n3- It is not indicated because NF1 responds to mutations in the neurofibromin gene (17q11.2), with recessive inheritance.\n4- No, two café-au-lait spots are not diagnostic and your new partner is very unlikely to be a carrier (NF1 is a rare disease).\n5- With these data, PGD is indicated, consisting of selecting embryos in vitro, to implant in the maternal uterus those without the mutation.\nCORRECT ANSWER: 2\nThe first step is to detect the mutation and once detected, if the mother is a carrier, to proceed with PGD.",
            "entities": {
                "T6": {
                    "text": "t is indicated after detection of the causative mutation in the index case and eventually in his mother",
                    "spans": "666-769",
                    "type": "claim"
                },...
            },
            "relations": [
                {
                    "hypothesis": "T18",
                    "premise": "T6",
                    "labels": {
                        "support": "entailment"
                    }
                },...
            ]
        }, ...
    }
}

ESTRUCTURA EN BASE A LAS RELACIONES (Solo aparecen los ID de las entidades relacionadas)
Contexto: Texto al completo
"""
def conseguir_relaciones_paso2(ficheros, a_train, a_dev, a_test, split_char = '\t', split_info = ' '):
    train, dev, test = {}, {}, {}
    for f in ficheros:
        if f == 'train':
            datos_archivo = a_train
        elif f =='dev':
            datos_archivo = a_dev
        else:
            datos_archivo = a_test
        # Formato inicial
        datos_json = {
            'group': f,
            'files': 0,
            'data': {}
        }

        # Toma los ficheros del método de arriba como referencia
        # path = Path(f'./Anotation_Data/{f}_data.json')
        # with open(path, "r", encoding="utf-8") as archivo:
        #     fichero = json.load(archivo)
        fichero = datos_archivo
        anotation = fichero['data']
        for file_name, data in anotation.items():
            relations = []
            entities = {}
            for relation in ['supports', 'attacks']:
                for r in data[relation]['data']:
                    e1_num = r['entity1'].split(':')[1]
                    e2_num = r['entity2'].split(':')[1]
                    for entity in ['claims', 'premises']:
                        for e in data[entity]['data']:
                            if e1_num not in entities or e2_num not in entities:
                                if e['tag_number'] == e1_num:
                                    if e1_num not in entities: entities[e1_num] = {'text': e['text'], 'spans': e['spans'], 'type': entity[:-1]}
                                elif e['tag_number'] == e2_num:
                                    if e2_num not in entities: entities[e2_num] = {'text': e['text'], 'spans': e['spans'], 'type': entity[:-1]}
                    relations.append({
                        'hypothesis': e1_num,
                        'premise': e2_num,
                        'labels': {relation[:-1]: 'entailment'}
                    })
            # Guarda una única vez el contexto (es el mismo en todas las relaciones del documento)
            datos_json['data'][file_name] = {
                'context': data['context'],
                'entities': entities,
                'relations': relations
            }
        datos_json['files'] = len(datos_json['data'])
        # with open(f'/gaueko1/users/murruela002/APP1/NLIsrc/tmp/{f}_relations.json', "w", encoding='utf-8') as archivo:
        #     json.dump(datos_json, archivo, ensure_ascii=False, indent=4)
        if f == 'train':
            train = datos_json
        elif f == 'dev':
            dev = datos_json
        else:
            test = datos_json
    return train, dev, test

"""
Estos métodos generan las instancias para NLI tomando el json de relaciones del paso 2:

1. Coge las relacicones existentes que han surgido de las anotaciones (del .ann) y las marca como ENTAILMENT
2. Crea las relaciones contrarias a las existentes y las señala como CONTRADICTION
3. Crea las neutrales en base a los criterios establecidos dependiendo del método auxiliar que se quiera utilizar
"""
def generate_contradictions(instances):
    for _, doc in instances['data'].items():
        for relation in doc['relations']:
            relation['labels']['attack' if 'support' in relation['labels'] else 'support'] = 'contradiction'
    return instances

def generar_instancias_paso3(files, anotations_data_all, relations_data_all, neutrales = 3):
    train, dev, test = {}, {}, {}
    for f in files:
        # path = Path(f'/gaueko1/users/murruela002/APP1/NLIsrc/Preproceso/Anotation/Relation_Data/{f}_relations.json')
        # output_path = Path(f'/gaueko1/users/murruela002/APP1/NLIsrc/Datasets/Todas_las_combinaciones')
        relations_data = relations_data_all[f]
        anotations_data = anotations_data_all[f]

        # with open(path, "r", encoding="utf-8") as archivo:
        #     data = json.load(archivo)
        instances = generate_contradictions(relations_data)

        # path = Path(
        #     f'/gaueko1/users/murruela002/APP1/NLIsrc/Preproceso/Anotation/Anotation_Data/{f}_data.json')
        # with open(path, "r", encoding="utf-8") as archivo:
        #     data = json.load(archivo)
        # data = anotations_data
        if f in ['dev', 'test'] or neutrales == 0:
            if f == 'dev':
                dev = generar_todas_las_neutrales(anotations_data, instances, f)
            elif f == 'test':
                test = generar_todas_las_neutrales(anotations_data, instances, f)
            else:
                train = generar_todas_las_neutrales(anotations_data, instances, f)
        elif neutrales == 1:
            train = generar_neutrales_evitando_entidades_concatenadas_viejo(anotations_data, instances, f)
        elif neutrales == 2:
            train = generar_neutrales_evitando_entidades_concatenadas(anotations_data, instances, f)
        else:
            train = generar_neutrales_evitando_entidades_concatenadas_solo_sueltas(anotations_data, instances, f)
    return train, dev, test


"""
Este método limpia las instancias del paso 3:

1. Consigue las opciones del examen
2. Limpia el contexto de estas (borra las opciones del examen)
3. Limpia las combinaciones posibles en las que no esté una de las opciones del examen incluida
"""
def entrenamiento_a_base_de_opciones_paso4(train_dev_test, neutrals=3, lang='en'):
    train, dev, test = {}, {}, {}
    # init_path = Path('/gaueko1/users/murruela002/APP1/NLIsrc/Datasets/Todas_las_combinaciones/')
    names = {"1":"_evitando_concatenaciones_viejo", "2": "_evitando_concatenaciones", "3": "_sueltas"}

    # for dataset_path in list(init_path.glob('*.json')):
    for f_type, data in train_dev_test:
        all_file_name = f"{f_type}{names[f'{neutrals}']}" if f'{neutrals}' in names else f_type
        if f_type in ['dev', 'test']:
            all_file_name = f_type
        aceptados = set()

        # Del dataset completo (carpeta actual) va fichero por fichero
        # cogiendo los datos y sus opciones en el dataset JSONL
        # with open(dataset_path, 'r', encoding='utf-8') as f:
        #     data = json.load(f)
        # all_file_name = dataset_path.stem


        opciones_grupo = conseguir_opciones(f_type, lang)

        for documento_separado in opciones_grupo:
            # El identificador del documento .ann se forma con la siguiente estructura:
            doc_name = f'{documento_separado["id"]}_{documento_separado["question_id_specific"]}.ann'
            aceptados.add(doc_name)

            # Hay más ficheros en el JSONL que en el dataset
            if doc_name in data['data']:
                # Conseguir opciones
                opciones = opciones_de_documento_jsonl(documento_separado)

                # Limpiar contexto
                nuevo_contexto = quitar_opciones_contexto(data['data'][doc_name]['context'], opciones)
                data['data'][doc_name]['context'] = nuevo_contexto
                grupo = all_file_name.split('_')[0]

                # Limpiar relaciones
                data['data'][doc_name]['relations'] = quitar_opciones_relaciones(data['data'][doc_name], opciones, grupo)
                entidades_mantenidas = conseguir_id_entidades_en_relaciones(data['data'][doc_name]['relations'])

                # Limpiar entidades
                data['data'][doc_name]['entities'] = quitar_opciones_entidades(data['data'][doc_name]['entities'],
                                                                               entidades_mantenidas)
        for doc_name, doc in data['data'].items():
            if doc_name not in aceptados:
                grupo = all_file_name.split('_')[0]

                nuevo_contexto, opciones = quitar_opciones_contexto2(data['data'][doc_name]['context'])
                data['data'][doc_name]['context'] = nuevo_contexto

                # Limpiar relaciones
                nuevas_relaciones = quitar_opciones_relaciones(data['data'][doc_name], opciones, grupo)
                data['data'][doc_name]['relations'] = nuevas_relaciones

                nuevas_entidades = conseguir_id_entidades_en_relaciones(data['data'][doc_name]['relations'])
                entidades_mantenidas = nuevas_entidades

                entidades_finales = quitar_opciones_entidades(data['data'][doc_name]['entities'],
                                                                               entidades_mantenidas)
                data['data'][doc_name]['entities'] = entidades_finales

        # with open(f'/gaueko1/users/murruela002/APP1/NLIsrc/tmp/{all_file_name}_sin_preguntas.json',
        #         'w') as archivo_json:
        #     json.dump(data, archivo_json, ensure_ascii=False, indent=4)
        if f_type == 'dev':
            dev = data
        elif f_type == 'test':
            test = data
        else:
            train = data
    return train, dev, test


"""
Los ficheros que se crean son en la carpeta de Datasets. Mantiene nivel de neutrales igual a los otros.
Estos son los que hay que pasarle al trainer. Cuentan con la siguiente estructura:
    [{
    "premise": "contexto",
    "hypothesis": "verbalización",
    "label": <"entailment", "neutral" o "contradiction">
    }, ...]
"""
mapeo_clases = {
  "attack": [
      "attack"
  ],
  "support": [
      "support"
  ]
}

def generar_dataset_neutrals_paso_final(train_dev_test, neutrals=3):
    output_path = "/gaueko1/users/murruela002/APP1/NLIsrc/Datasets"

    with open(f"{output_path}/Informe_ultimo_dataset.txt", 'w', encoding='utf-8') as file:
        file.write("NUEVO DATASET:\n")
    names = {"1": "_evitando_concatenaciones_viejo", "2": "_evitando_concatenaciones", "3": "_sueltas"}

    # for dataset_path in list(init_path.glob('*.json')):
    for f_type, data in train_dev_test:
        neutral, entail, contra, todas_las_neutrales = 0, 0, 0, 0
        datos_finales = []
        lista_neutrales = []

        all_file_name = f"{f_type}{names[f'{neutrals}']}" if f'{neutrals}' in names else f_type
        if f_type in ['dev', 'test']:
            all_file_name = f_type


        # with open(dataset_path, 'r',encoding='utf-8') as file:
        #     data = json.load(file)

        for nombre, documento in data['data'].items():
            n = 0
            premise = documento['context']

            for relation in documento['relations']:
                h = documento['entities'][relation['hypothesis']]['text']
                p = documento['entities'][relation['premise']]['text']
                for v, l in relation['labels'].items():

                    if l == 'entailment':
                        if (all_file_name == 'train' or all_file_name == 'train_evitando_concatenaciones' or
                all_file_name == 'train_sueltas' or all_file_name == 'train_evitando_concatenaciones_viejo'): # and entail<12222222
                            for cambio in mapeo_clases[v]:
                                datos_finales.append({
                                    'premise': premise,  # Context
                                    'hypothesis': f'"{h}" {cambio} "{p}"',
                                    # Poner comillas separando las premisas y/o claims de la relación
                                    'label': l
                                })
                                entail += 1
                        elif (all_file_name != 'train'
                          and all_file_name != 'train_evitando_concatenaciones'
                          and all_file_name != 'train_sueltas'
                          and all_file_name != 'train_evitando_concatenaciones_viejo'):
                            for cambio in mapeo_clases[v]:
                                datos_finales.append({
                                    'premise': premise,  # Context
                                    'hypothesis': f'"{h}" {cambio} "{p}"',
                                    # Poner comillas separando las premisas y/o claims de la relación
                                    'label': l
                                })
                                entail += 1
                    elif l == 'contradiction':
                        if (all_file_name == 'train' or all_file_name == 'train_evitando_concatenaciones' or
                all_file_name == 'train_sueltas' or all_file_name == 'train_evitando_concatenaciones_viejo'): #and contra<1222222222:
                            for cambio in mapeo_clases[v]:
                                datos_finales.append({
                                    'premise': premise,  # Context
                                    'hypothesis': f'"{h}" {cambio} "{p}"',
                                    # Poner comillas separando las premisas y/o claims de la relación
                                    'label': l
                                })
                                contra += 1
                        elif (all_file_name != 'train'
                          and all_file_name != 'train_evitando_concatenaciones'
                          and all_file_name != 'train_sueltas'
                          and all_file_name != 'train_evitando_concatenaciones_viejo'):
                            for cambio in mapeo_clases[v]:
                                datos_finales.append({
                                    'premise': premise,  # Context
                                    'hypothesis': f'"{h}" {cambio} "{p}"',
                                    # Poner comillas separando las premisas y/o claims de la relación
                                    'label': l
                                })
                                contra += 1

                    elif (all_file_name != 'train'
                          and all_file_name != 'train_evitando_concatenaciones'
                          and all_file_name != 'train_sueltas'
                          and all_file_name != 'train_evitando_concatenaciones_viejo'):
                        for cambio in mapeo_clases[v]:
                            datos_finales.append({
                                'premise': premise,  # Context
                                'hypothesis': f'"{h}" {cambio} "{p}"',
                                # Poner comillas separando las premisas y/o claims de la relación
                                'label': l
                            })
                            neutral += 1

                    elif n < (len(mapeo_clases['support'])) and neutral<107:
                        for cambio in mapeo_clases[v]:
                            datos_finales.append({
                                'premise': premise,  # Context
                                'hypothesis': f'"{h}" {cambio} "{p}"',
                                # Poner comillas separando las premisas y/o claims de la relación
                                'label': l
                            })
                            n += 1
                            todas_las_neutrales += 1
                            neutral += 1
                    else:
                        for cambio in mapeo_clases[v]:
                            todas_las_neutrales += 1
                            lista_neutrales.append({
                                'id': f'{todas_las_neutrales}${nombre}',
                                'id_h': relation['hypothesis'],
                                'premise': premise, # Context
                                'hypothesis': f'"{h}" {cambio} "{p}"', # Poner comillas separando las premisas y/o claims de la relación
                                'label': l})

        if (all_file_name == 'train' or all_file_name == 'train_evitando_concatenaciones' or
                all_file_name == 'train_sueltas' or all_file_name == 'train_evitando_concatenaciones_viejo'):
            while neutral < len(lista_neutrales)-1 and (neutral < (entail)):
                neutral += 1
                datos_finales.append({'premise': lista_neutrales[neutral]['premise'], # Context
                                      'hypothesis': lista_neutrales[neutral]['hypothesis'], # Poner comillas separando las premisas y/o claims de la relación
                                      'label': lista_neutrales[neutral]['label']})

        with open(f"{output_path}/Informe_ultimo_dataset.txt", 'a', encoding='utf-8') as file:
            file.write(f"Nombre: {all_file_name}\n"
                       f"\tNúmero de datos total: {len(datos_finales)}\n"
                       f"\t\tEntailment: {entail}\n"
                       f"\t\tContra: {contra}\n"
                       f"\t\tNeutral: {neutral}\n"
                       f"\t\tTodas las neutrales que había: {todas_las_neutrales}\n")

        with open(f'{output_path}/{all_file_name}.json', 'w', encoding='utf-8') as file:
            json.dump(datos_finales, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    initial_data_path = "/proiektuak/edhia/Casimedicos_BRAT_neutralized/Annotation_and_format_conversion/dataset/neutralized_BRAT_fully_annotated"
    paths = ['train', 'dev', 'test']
    neutrals = 3 # Tipo de neutrales que se quieren utilizar

    """!!!!!!!!!!!!!!!!!!!!!!! DO NOT EDIT PREPROCESS !!!!!!!!!!!!!!!!!!!!!!!"""
    json_anotaciones_train, json_anotaciones_dev, json_anotaciones_test = conseguir_datos_al_completo_paso1(paths, initial_data_path) # Paso 1
    json_anotaciones = {"train": json_anotaciones_train, "dev": json_anotaciones_dev, "test": json_anotaciones_test}

    json_relaciones_train, json_relaciones_dev, json_relaciones_test = conseguir_relaciones_paso2(paths, json_anotaciones_train, json_anotaciones_dev, json_anotaciones_test) # Paso 2
    json_relaciones = {"train": json_relaciones_train, "dev": json_relaciones_dev, "test": json_relaciones_test}

    """------------------------- EDIT IF NEEDED OTHER NEUTRALS -------------------------"""
    json_relaciones_train, json_relaciones_dev, json_relaciones_test = generar_instancias_paso3(paths, json_anotaciones, json_relaciones, neutrals) # Paso 3
    conjunto = [('train', json_relaciones_train), ('dev', json_relaciones_dev), ('test', json_relaciones_test)]
    train_limpio, dev_limpio, test_limpio = entrenamiento_a_base_de_opciones_paso4(conjunto, neutrals)

    generar_dataset_neutrals_paso_final([('train', train_limpio), ('dev', dev_limpio), ('test', test_limpio)], neutrals)
