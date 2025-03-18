from pathlib import Path
import json
import sys
# ruta_abs = ('/gaueko1/users/murruela002/APP1/NLIsrc/')
# sys.path.append(ruta_abs)
# from src.Preproceso.metodos_relaciones import sacar_relaciones, no_relacionados_indirectamente, \
#     sacar_nodos_sin_ninguna_relacion, por_relacionar
from metodos_relaciones import sacar_relaciones_documento, por_relacionar, no_relacionados_indirectamente, grupos_del_grafo, sacar_nodos_sin_ninguna_relacion, por_relacionar



def sumar_relaciones_neutrales(doc, entity, entity2, entity_name, entity_name2, entidades_a_sumar = 2):
    if entity['tag_number'] not in doc['entities']:
        doc['entities'][entity['tag_number']] = {'text': entity['text'], 'spans': entity['spans'], 'type': entity_name}
    if entity2['tag_number'] not in doc['entities']:
        doc['entities'][entity2['tag_number']] = {'text': entity2['text'], 'spans': entity2['spans'], 'type': entity_name2}
    doc['relations'].append({
        "hypothesis": entity['tag_number'],
        "premise": entity2['tag_number'],
        "labels": {
            "support": "neutral",
            "attack": "neutral"
        }
    })
    if entidades_a_sumar == 2:
        doc['relations'].append({
            "hypothesis": entity2['tag_number'],
            "premise": entity['tag_number'],
            "labels": {
                "support": "neutral",
                "attack": "neutral"
            }
        })
    return doc
def estan_en_el_mismo_grupo(grupos, h, p):

    for grupo in grupos:
        if h in grupo:
            if p in grupo:
                return True
    return False

def generar_neutrales_evitando_entidades_concatenadas_solo_sueltas(data, instances, file_tipe):
    if file_tipe == 'train':
        for name, doc in instances['data'].items():
            sin_relacionar = sacar_nodos_sin_ninguna_relacion(data['data'][name], doc)
            documento = data['data'][name]
            for entity_name in ['claims', 'premises']:
                for entity in documento[entity_name]['data']:
                    if entity['tag_number'] in sin_relacionar:

                        for entity_name2 in ['claims', 'premises']:
                            for entity2 in documento[entity_name2]['data']:
                                if entity2['tag_number'] != entity['tag_number']:
                                    doc = sumar_relaciones_neutrales(
                                        doc,
                                        entity,
                                        entity2,
                                        entity_name,
                                        entity_name2,
                                        1)
            instances['data'][name] = doc
            with open(f'/gaueko1/users/murruela002/APP1/NLIsrc/scripts/preprocess/{file_tipe}_sueltas.json', "w", encoding='utf-8') as archivo:
                json.dump(instances, archivo, ensure_ascii=False, indent=4)
        return instances



def generar_neutrales_evitando_entidades_concatenadas_viejo(data, instances, file_tipe):
    if file_tipe == 'train':
        for name, doc in instances['data'].items():
            sin_relacionar = por_relacionar(data['data'][name], doc)
            documento = data['data'][name]
            for entity_name in ['claims', 'premises']:
                for entity in documento[entity_name]['data']:
                    if entity['tag_number'] in sin_relacionar:

                        for entity_name2 in ['claims', 'premises']:
                            for entity2 in documento[entity_name2]['data']:
                                if entity2['tag_number'] != entity['tag_number']:
                                    doc = sumar_relaciones_neutrales(
                                        doc,
                                        entity,
                                        entity2,
                                        entity_name,
                                        entity_name2,
                                        1)
            instances['data'][name] = doc
        return instances



def generar_neutrales_evitando_entidades_concatenadas(data, instances, file_tipe):
    if file_tipe == 'train':
        for name, doc in instances['data'].items():
            sin_relacionar = sacar_nodos_sin_ninguna_relacion(data['data'][name], doc)
            grupos_relacionados = grupos_del_grafo(doc)
            documento = data['data'][name]

            for entity_name in ['claims', 'premises']:
                for entity in documento[entity_name]['data']:

                        for entity_name2 in ['claims', 'premises']:
                            for entity2 in documento[entity_name2]['data']:
                                if entity2['tag_number'] != entity['tag_number']:
                                    if (entity2['tag_number'] in sin_relacionar
                                            or entity['tag_number'] in sin_relacionar):
                                        doc = sumar_relaciones_neutrales(
                                            doc,
                                            entity,
                                            entity2,
                                            entity_name,
                                            entity_name2,
                                            1)
                                    elif not estan_en_el_mismo_grupo(grupos_relacionados, entity['tag_number'], entity2['tag_number']):
                                        doc = sumar_relaciones_neutrales(
                                            doc,
                                            entity,
                                            entity2,
                                            entity_name,
                                            entity_name2,
                                            1)
            instances['data'][name] = doc
        return instances


def generar_todas_las_neutrales(data, instances,file_tipe ):

    for name, doc in instances['data'].items():
        relaciones = sacar_relaciones_documento(doc)
        documento = data['data'][name]

        for entity_name in ['claims', 'premises']:
            for entity in documento[entity_name]['data']:
                    for entity_name2 in ['claims', 'premises']:
                        for entity2 in documento[entity_name2]['data']:
                            if entity2['tag_number'] != entity['tag_number']:
                                if (entity['tag_number'], entity2['tag_number']) not in relaciones:
                                    doc = sumar_relaciones_neutrales(
                                        doc,
                                        entity,
                                        entity2,
                                        entity_name,
                                        entity_name2,
                                        1)

        instances['data'][name] = doc
        with open(f'/gaueko1/users/murruela002/APP1/NLIsrc/scripts/preprocess/{file_tipe}.json', "w", encoding='utf-8') as archivo:
            json.dump(instances, archivo, ensure_ascii=False, indent=4)
    return instances