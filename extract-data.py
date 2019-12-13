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


def extract_cc():
    source_folder_path = "tensorflow"
    # if len(sys.argv) > 1:
    #     source_folder_path = sys.argv[1]

    if not os.path.isdir(source_folder_path):
        print('Folder ', source_folder_path, 'not found')
        exit(0)

    source_files_path = [y for x in os.walk(source_folder_path) for y in glob(os.path.join(x[0], '*.cc'))]

    for file in source_files_path:
        # if not file.endswith('scratch_buffer.cc'):
        #     continue
        filestr = file[len(source_folder_path) + 1:]

        def find_typerefs(node, indent):
            """ Find all references to the type named 'typename'
            """
            try:
                kind = node.kind
                if kind == clang.cindex.CursorKind.FUNCTION_DECL:
                    # node.location.line
                    csv.append(f'{node.spelling},{filestr},{node.extent.start.line}')

                elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
                    csv.append(f'{node.spelling},{filestr},{node.extent.start.line}')
                elif node.kind == clang.cindex.CursorKind.CXX_METHOD:
                    csv.append(f'{node.spelling},{filestr},{node.extent.start.line}')
                # elif node.kind == clang.cindex.CursorKind.FIELD_DECL:
                #     csv.append(f'{node.spelling},{filestr},{node.extent.start.line}')
                # elif node.kind == clang.cindex.CursorKind.OBJC_INSTANCE_METHOD_DECL:
                #     csv.append(f'{node.spelling},{filestr},{node.extent.start.line}')
                # elif node.kind == clang.cindex.CursorKind.OBJC_CLASS_METHOD_DECL:
                #     csv.append(f'{node.spelling},{filestr},{node.extent.start.line}')
                else:
                    pass
            except:
                pass

                # print(f'{indent}node kind {node.kind}')
            # if node.kind.is_reference():
            #     ref_node = clang.cindex.Cursor_ref(node)
            # if ref_node.spelling == typename:
            # print('Found %s [line=%s, col=%s]' % (
            #     typename, node.location.line, node.location.column))
            # Recurse for children of this node
            for c in node.get_children():
                find_typerefs(c, indent + '  ')

        # with open(filename, "r") as source:
        index = clang.cindex.Index.create()
        tree = index.parse(file)
        print(f'VISITING {filestr}')

        find_typerefs(tree.cursor, '  ')


def extract_py():
    source_folder_path = "tensorflow"
    # if len(sys.argv) > 1:
    #     source_folder_path = sys.argv[1]

    if not os.path.isdir(source_folder_path):
        print('Folder ', source_folder_path, 'not found')
        exit(0)

    source_files_path = [y for x in os.walk(source_folder_path) for y in glob(os.path.join(x[0], '*.py'))]

    for file in source_files_path:

        # if not file.endswith('configure.py'):
        #     continue
        filestr = file[len(source_folder_path) + 1:]

        with open(file, "r") as source:

            print(f'VISITING {filestr}')
            tree = ast.parse(source.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    csv.append(f'{node.name},{filestr},{node.lineno}')
                elif isinstance(node, ast.FunctionDef):
                    csv.append(f'{node.name},{filestr},{node.lineno}')
            # analyzer = Analyzer(file, csv)
            # analyzer.visit(tree)


class Analyzer(ast.NodeVisitor):
    def __init__(self, filename, csv):
        self.filename = filename
        self.csv = csv

    def visit_ClassDef(self, node: ast.ClassDef):
        print(f'  visit_ClassDef {node.name} {[x.id for x in node.bases if hasattr(x, "id")]}')
        self.csv.append(f'{node.name},{self.filename},{node.lineno}')

    def visit_FunctionDef(self, node: ast.ClassDef):
        print(f'  visit_FunctionDef {node.name}')  # {[x.id for x in node.bases if hasattr(x, "id")]}')
        self.csv.append(f'{node.name},{self.filename},{node.lineno}')


if __name__ == '__main__':
    extract_cc()
    extract_py()
