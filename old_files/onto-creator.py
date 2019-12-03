import ast

from owlready2 import *


def main():
    with open("tree.py", "r") as source:
        tree = ast.parse(source.read())

    onto = get_ontology("http://usi.ch/giacomelli/Knowledge_Analysis_and_Management.owl")

    analyzer = Analyzer(onto)
    analyzer.visit(tree)

    onto.save('tree.owl')


class Analyzer(ast.NodeVisitor):
    def __init__(self, onto):
        self.onto = onto

    def visit_ClassDef(self, node: ast.ClassDef):
        x: ast.Name
        print(f'visit_ClassDef {node.name} {[x.id for x in node.bases]}')

        bases = tuple([self.decodeBase(b) for b in node.bases])
        with self.onto:
            cl = types.new_class(node.name, bases)

        # for all statements that are assignments
        for a in (stmt for stmt in node.body if type(stmt) == ast.Assign):
            # read the tuple values and create the according property
            for val in a.value.elts:
                self.create_property(val)

    def create_property(self, val):
        id = val.s
        if id == 'name':
            id = 'jname'
        bases = (ObjectProperty,) if id == 'body' or id == 'parameters' else (DataProperty,)
        with self.onto:
            types.new_class(id, bases)

    def decodeBase(self, base):
        return Thing if base.id == 'Node' else self.onto[base.id]


if __name__ == "__main__":
    main()
