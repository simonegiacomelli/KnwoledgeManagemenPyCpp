import ast
import os
from glob import glob
import clang.cindex

clang.cindex.Config.set_library_path('./clang+llvm-9.0.0-x86_64-darwin-apple/lib')

class Csv:
    def __init__(self, filename):
        self.f = open(filename, 'w')

    def append(self, line):
        self.f.write(line)
        self.f.write('\n')
        self.f.flush()

    def __del__(self):
        self.f.close()

def find_typerefs(node, typename):
    """ Find all references to the type named 'typename'
    """
    if node.kind.is_reference():
        ref_node = clang.cindex.Cursor_ref(node)
        #if ref_node.spelling == typename:
        print( 'Found %s [line=%s, col=%s]' % (
            typename, node.location.line, node.location.column))
    # Recurse for children of this node
    for c in node.get_children():
        print('recursing')
        find_typerefs(c, typename)


def main():

    csv = Csv('data-cc.csv')

    source_folder_path = "./tensorflow"
    # if len(sys.argv) > 1:
    #     source_folder_path = sys.argv[1]

    if not os.path.isdir(source_folder_path):
        print('Folder ', source_folder_path, 'not found')
        exit(0)

    source_files_path = [y for x in os.walk(source_folder_path) for y in glob(os.path.join(x[0], '*.cc'))]

    for file in source_files_path:
        # if not filename.endswith('configure.py'):
        #     continue

        # with open(filename, "r") as source:
        index = clang.cindex.Index.create()
        tree = index.parse(file)
        print(f'VISITING {file}')

        find_typerefs(tree.cursor,'')


if __name__ == "__main__":
    main()
