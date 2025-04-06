import json
from pathlib import Path


"""
Consigue los ID de las relaciones de un caso completo. Es decir:
set("T1", "T3", "T8")
"""
def add_extra_options(opciones):
    # opciones.add("The correct answer is 1")
    # opciones.add("The correct answer is 2")
    # opciones.add("The correct answer is 3")
    # opciones.add("The correct answer is 4")
    # opciones.add("The correct answer is 5")
    # opciones.add("the correct option is 1")
    # opciones.add("the correct answer is 1")
    # opciones.add("the correct answer is 2")
    # opciones.add("the correct answer is 3")
    # opciones.add("the correct answer is 4")
    # opciones.add("the correct answer is 5")
    # opciones.add("The correct option is 1")
    # opciones.add("incorrect option 4")
    # opciones.add("incorrect option 2")
    # opciones.add("incorrect option 1")
    # opciones.add("incorrect option 5")
    # opciones.add("incorrect option 3")
    # opciones.add("I think the correct answer is 2")
    # opciones.add("answer 1 is correct")
    # opciones.add("answer 2 is correct")
    # opciones.add("answer 3 is correct")
    # opciones.add("answer 4 is correct")
    # opciones.add("answer 5 is correct")
    return opciones

def conseguir_id_entidades_en_relaciones(relaciones):
    ids = set()
    for relacion in relaciones:
        ids.add(relacion['hypothesis'])
        ids.add(relacion['premise'])
    return ids


def quitar_opciones_contexto2(contexto):
    lineas = contexto.split('\n')
    contexto_final = ''
    opciones = set()

    for linea_init in lineas:
        if linea_init[0].isdigit():
            if linea_init[1] == '-':
                if linea_init[2] == ' ':
                    opciones.add(linea_init[3:].replace('\n', '').replace('\r', '').replace('.', ''))
                    opciones.add(linea_init[4:].replace('\n', '').replace('\r', '').replace('.', ''))
                    opciones.add(linea_init[3:-1].replace('\n', '').replace('\r', '').replace('.', ''))
                    opciones.add(linea_init[3:].replace('\n', '').replace('\r', ''))
                    opciones = add_extra_options(opciones)

                else:
                    contexto_final += linea_init + '\n'
            else:
                contexto_final += linea_init + '\n'
        else:
            contexto_final += linea_init + '\n'

    return contexto_final, opciones
"""
Quita las posibles respuestas del contexto
"""


def quitar_opciones_contexto(contexto, opciones):
    lineas = contexto.split('\n')
    contexto_final = ''

    for linea_init in lineas:
        linea = ' '.join(linea_init.split('- ')[1:])
        linea = linea.replace('.', '').replace('\n', '')
        if linea not in opciones:
            contexto_final += linea_init + '\n'

    return contexto_final


"""
Por cada documento, mantiene solo las relaciones en las que haya una MC
ya sea la premisa o la claim y devuelve las relaciones restantes por casa documento.
"""


def quitar_opciones_relaciones(documento, opciones, group = 'dev'):
    nuevas_relaciones = []
    relaciones = documento['relations']
    entidades = documento['entities']

    for relacion in relaciones:
        # print(documento)
        # print(f'\n\nENTIDADES: {entidades}\n')
        # print(f'RELACION: {relacion}')
        hypothesis = entidades[relacion['hypothesis']]['text']
        premise = entidades[relacion['premise']]['text']
        if hypothesis in opciones or premise in opciones:
            if not (hypothesis in opciones):  # SIEMPRE QUITAMOS LAS RELACIONES ENTRE MAJOR CLAIMS
                # if hypothesis == 'It is a typical picture of infectious mononucleosis':
                #     print(relacion)
                #     print(f'{relacion["hypothesis"]}: {hypothesis}')
                #     print(f'{relacion["premise"]}: {premise}')
                #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                nuevas_relaciones.append(relacion)
            # if not (hypothesis in opciones and group != 'train'): # SIEMPRE QUITAMOS LAS RELACIONES ENTRE MAJOR CLAIMS
            #     nuevas_relaciones.append(relacion)
            # elif not (hypothesis in opciones and premise in opciones):
            #     nuevas_relaciones.append(relacion)

    return nuevas_relaciones


"""
Mantiene en las entidades de un documento solo los ID que se le han pasado. Es decir:
entidades = {"T6": {...}, "T18": {...}, "T21": {...}, "T20": {...}}
ids_a_mantener = set("T6", "T18")

entidades_mantenidas = entidades = {"T6": {...}, "T18": {...}}
"""


def quitar_opciones_entidades(entidades, ids_a_mantener):
    entidades_mantenidas = {}
    for id, entidad in entidades.items():
        if id in ids_a_mantener:
            entidades_mantenidas[id] = entidad
    return entidades_mantenidas


"""
Devuelve el contenido de las respuestas posibles del examen, dado un JSON 
del dataset de los JSONL. Es decir:
documento_separado = {
                        "id": 274, 
                        "year": 2016, 
                        "question_id_specific": 73, 
                        "full_question": <Pregunta parte superior>, 
                        "full_answer": <Justificacion>, 
                        "type": "DIGESTIVE SYSTEM", 
                        "options": {
                            "1": "Endoscopic therapy and subsequently to establish endovenous treatment with proton pump inhibitor.", 
                            "2": "Endoscopic therapy is not indicated.", 
                            "3": "Endoscopic therapy.", 
                            "4": "Complicated ulcer.", 
                            "5": NaN
                        }, 
                        "correct_option": 1, 
                        "explanations": {...}
                        ...
}}

opciones = set("Endoscopic therapy and subsequently to establish endovenous treatment with proton pump inhibitor", 
               "Endoscopic therapy is not indicated", 
               "Endoscopic therapy", 
               "Complicated ulcer", 
               NaN)
"""
def opciones_de_documento_jsonl(documento_separado):
    opciones = set()
    puntuacion = ",.?!;:"
    for opcion in documento_separado['options'].values():
        if isinstance(opcion, str):
            if opcion[-1] in puntuacion:
                # Se le quita el punto porque no aparecen en el otro dataset
                opciones.add(opcion[:-1])
                opciones.add(opcion[:-2])
                opciones.add(opcion[1:])
                opciones.add(opcion)
                opciones = add_extra_options(opciones)
            else:
                opciones.add(opcion)
                opciones.add(opcion[:-1])
                opciones.add(opcion[1:])
                opciones.add(f'{opcion}.')
                opciones = add_extra_options(opciones)
    return opciones

def conseguir_opciones(all_file_name, lang):
    # Coge el JSONL en el idioma establecido y del grupo que corresponda
    opciones_path = Path(
        '/gaueko1/users/murruela002/evaluador/data/casimedicos/original')

    file_name = all_file_name.split('_')[0]
    opciones_path = opciones_path.joinpath(f'{lang}_{file_name}_casimedicos.jsonl')

    with open(opciones_path, "r", encoding="utf-8") as f2:
        opciones_grupo = [json.loads(row) for row in f2]  # Carga todos los documentos (1 array de JSON)
    return opciones_grupo