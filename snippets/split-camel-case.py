from re import finditer


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


with open('../data-py.csv', 'r') as f:
    for line in f:
        id = line.split(',')[0]
        print(id, camel_case_split(id))
