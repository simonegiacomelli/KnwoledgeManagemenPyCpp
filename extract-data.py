import ast
import os
from glob import glob
import ast

import pandas as pd
import sys


def main():
    source_folder_path = "./tensorflow"
    # if len(sys.argv) > 1:
    #     source_folder_path = sys.argv[1]

    if not os.path.isdir(source_folder_path):
        print('Folder ', source_folder_path, 'not found')
        exit(0)

    source_files_path = [y for x in os.walk(source_folder_path) for y in glob(os.path.join(x[0], '*.py'))]

    for source_path in source_files_path:
        with open(source_path, "r") as source:
            print(f'VISITING {source_path}')
            tree = ast.parse(source.read())

            analyzer = Analyzer()
            analyzer.visit(tree)


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        pass

    def visit_ClassDef(self, node: ast.ClassDef):
        print(f'  visit_ClassDef {node.name} {[x.id for x in node.bases if hasattr(x, "id")]}')


if __name__ == "__main__":
    main()
