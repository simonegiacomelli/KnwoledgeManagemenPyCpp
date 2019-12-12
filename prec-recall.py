import query

docmap = dict()
for idx, doc in enumerate(query.documents):
    docmap[(doc[1], doc[2])] = idx


class Measure:
    def __init__(self, query, results, correct_idx):

        pos = 0
        for i, tup in enumerate(results):
            if correct_idx == tup[0]:
                pos = i + 1
                break
        if pos == 0:
            st = 'not found'
        else:
            st = f'in position {pos}'
        print('  doc', st)


class GroundTruthRow:
    def __init__(self, line):
        p = line.split(',')
        self.query = query.Query(p[0])
        self.entity = p[1]
        self.file = p[2]
        self.correct_idx = docmap.get((self.entity, self.file), -1)
        if self.correct_idx == -1:
            raise Exception('Ground truth not valid ' + p)
        print(query.documents[self.correct_idx])
        # self.query.print()
        print("query: " + p[0])
        results = [Measure(p[0], r, self.correct_idx) for r in self.query.execute()]

    def documents_index(self):
        return


def main():
    with open('ground-truth.txt', 'r') as f:
        gt_lines = f.read().split('\n')

    gts = [GroundTruthRow(line) for line in gt_lines]
    # for gt in gts:
    #     # print(gt.query.execute())
    #     print(query.documents[gt.documents_index()])
    #     # gt.query.print()


if __name__ == '__main__':
    main()
