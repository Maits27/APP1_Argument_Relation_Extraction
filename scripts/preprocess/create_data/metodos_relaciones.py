
# Formato:
# relaciones = set(
#     ('T2', 'T3')
#     ('T1', 'T10')
# )
def sacar_relaciones_documento(instances):
    relaciones_doc = set()
    for relation in instances['relations']:
        relaciones_doc.add((relation['hypothesis'], relation['premise']))
    return relaciones_doc


# Formato:
# relaciones = {
#     ('T2', 'T3'): {'tipo': 'support'},
#     ('T1', 'T10'): {'tipo': 'attack'}
# }
def sacar_dic_relaciones_relacion(doc, value):
    relaciones = {}
    for relation in doc['relations']:
        if value == 'num':
            tupla = (relation[f'e1_{value}'], relation[f'e2_{value}'])
        else:
            entity_name = doc['entities'][relation["hypothesis"]]['type']
            entity_name2 = doc['entities'][relation["premise"]]['type']
            tupla = (f'{relation["hypothesis"]}_{entity_name}', f'{relation["premise"]}_{entity_name2}')
        keys = relation['labels'].keys()
        relaciones[tupla] = {'tipo': next(iter(keys))}
    return relaciones


# Diccionario es de tipo: {'T1': 'T2', 'T2': 'T5', 'T20': 'T7'}
def sacar_dic_relaciones(documento):
    relaciones = {}
    for r in documento['relations']:
        relaciones[r['hypothesis']] = r['premise']
    return relaciones


# Devuelve el set con todos: {'T1': 'T2', 'T2': 'T5', 'T20': 'T7'} ->
# set(T1, T2, T5, T7, T20)
def sacar_nodos_en_relacion(documento):
    relaciones = sacar_dic_relaciones(documento)
    return set(relaciones.keys()) | set(relaciones.values())


# Devuelve el set con todos los nodos fuera de relacion:
# Relaciones -> {'T1': 'T2', 'T2': 'T5', 'T20': 'T7'}
# Mis valores: T1, T2, T3, T4, T5, T7, T20
# Devuelve: set(T3, T4)
def sacar_nodos_sin_ninguna_relacion(doc_completo, documento_relaciones):
    todas = set()
    relacionados = sacar_nodos_en_relacion(documento_relaciones)
    for entity_name in ['claims', 'premises']:
        for entity in doc_completo[entity_name]['data']:
            todas.add(entity['tag_number'])
    return todas - relacionados


# Devuelve el set con todos los nodos que no tienen relacion indirecta con mas:
# Relaciones -> {'T1': 'T2', 'T2': 'T5', 'T20': 'T7'}
# Devuelve: set(T7, T5)
def no_relacionados_indirectamente(documento):
    relaciones = sacar_dic_relaciones(documento)
    keys = set(relaciones.keys())
    values = set(relaciones.values())
    nodos_sin_relaciones = values - keys
    return nodos_sin_relaciones


# Devuelve los grupos creados por los grafos. Por ejemplo en este tipo de grafo:
# T1 -> T2 -> T3
# T5 -> T6
# T4
# Devuelve: [{T1, T2, T3}, {T5, T6}]
def grupos_del_grafo(documento):
    grupos = []
    for relacion in documento['relations']:
        h = relacion['hypothesis']
        p = relacion['premise']
        if len(grupos) == 0:
            grupos.append({h, p})
        else:
            sumado = False
            for i, grupo in enumerate(grupos):
                if h in grupo or p in grupo:
                    grupos[i] = grupo | {h, p}
                    sumado = True
                    break
            if not sumado:
                grupos.append({h, p})
    return grupos


def por_relacionar(doc_completo, documento_relaciones):
    sin_relacionar = sacar_nodos_sin_ninguna_relacion(doc_completo, documento_relaciones)
    sin_relacionar = sin_relacionar | set(no_relacionados_indirectamente(documento_relaciones))
    return sin_relacionar
