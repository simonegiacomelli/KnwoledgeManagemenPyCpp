import ast
import os
from glob import glob


class Csv:
    def __init__(self, filename):
        self.f = open(filename, 'w')

    def append(self, line):
        self.f.write(line)
        self.f.write('\n')
        self.f.flush()

    def __del__(self):
        self.f.close()


def main():
    csv = Csv('data-py.csv')

    source_folder_path = "./tensorflow"
    # if len(sys.argv) > 1:
    #     source_folder_path = sys.argv[1]

    if not os.path.isdir(source_folder_path):
        print('Folder ', source_folder_path, 'not found')
        exit(0)

    source_files_path = [y for x in os.walk(source_folder_path) for y in glob(os.path.join(x[0], '*.py'))]

    for filename in source_files_path:

        # if not filename.endswith('configure.py'):
        #     continue

        with open(filename, "r") as source:

            print(f'VISITING {filename}')
            tree = ast.parse(source.read())

            analyzer = Analyzer(filename, csv)
            analyzer.visit(tree)


class Analyzer(ast.NodeVisitor):
    def __init__(self, filename, csv):
        self.filename = filename
        self.csv = csv

    def visit_ClassDef(self, node: ast.ClassDef):
        print(f'  visit_ClassDef {node.name} {[x.id for x in node.bases if hasattr(x, "id")]}')
        self.csv.append(f'{node.name},{self.filename},{node.lineno}')

    # todo distinguish method from functions
    # it can be done reading the first parameter, if it is self, its a method
    def visit_FunctionDef(self, node: ast.ClassDef):
        print(f'  visit_FunctionDef {node.name}')  # {[x.id for x in node.bases if hasattr(x, "id")]}')
        self.csv.append(f'{node.name},{self.filename},{node.lineno}')


if __name__ == "__main__":
    main()
