from re import finditer


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def split_entity(id):
    if '_' in id:
        parts = id.split('_')
    else:
        parts = camel_case_split(id)
    parts = [p for p in parts if p.strip() != '']
    return parts


idx = 0
with open('../data-py.csv', 'r') as f:
    for line in f:
        idx += 1
        if idx > 100:
            break

        id = line.split(',')[0]
        parts = split_entity(id)
        print(id, parts)
