import ast
import os
from glob import glob

import clang.cindex

clang.cindex.Config.set_library_path('./clang+llvm-9.0.0/lib')


class Csv:
    def __init__(self, filename):
        self.f = open(filename, 'w')
        self.idx = 0

    def append(self, line):
        self.idx += 1
        self.f.write(f'{self.idx},' + line)
        self.f.write('\n')
        self.f.flush()

    def __del__(self):
        self.f.close()


csv = Csv('data.csv')
source_folder_path = "tensorflow/tensorflow"
if not os.path.isdir(source_folder_path):
    print('Folder ', source_folder_path, 'not found')
    exit(0)


def extract_cc():
    source_files_path = [y for x in os.walk(source_folder_path) for y in glob(os.path.join(x[0], '*.cc'))]

    for file in source_files_path:

        filestr = file[len(source_folder_path) + 1:]

        def find_typerefs(node):
            try:
                kind = node.kind
                if kind == clang.cindex.CursorKind.FUNCTION_DECL:
                    csv.append(f'{node.spelling},{filestr},{node.extent.start.line},function')
                elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
                    csv.append(f'{node.spelling},{filestr},{node.extent.start.line},class')
                elif node.kind == clang.cindex.CursorKind.CXX_METHOD:
                    csv.append(f'{node.spelling},{filestr},{node.extent.start.line},method')
            except:
                pass

            for c in node.get_children():
                find_typerefs(c)

        index = clang.cindex.Index.create()
        tree = index.parse(file)
        print(f'VISITING {filestr}')

        find_typerefs(tree.cursor)


def extract_py():
    source_files_path = [y for x in os.walk(source_folder_path) for y in glob(os.path.join(x[0], '*.py'))]

    for file in source_files_path:
        filestr = file[len(source_folder_path) + 1:]

        with open(file, "r") as source:
            print(f'VISITING {filestr}')
            tree = ast.parse(source.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    csv.append(f'{node.name},{filestr},{node.lineno},class')
                elif isinstance(node, ast.FunctionDef):
                    csv.append(f'{node.name},{filestr},{node.lineno},function')


if __name__ == '__main__':
    extract_cc()
    extract_py()
