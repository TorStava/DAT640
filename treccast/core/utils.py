import re
from collections import defaultdict
from typing import TypedDict, Dict, List, Tuple


class Query(TypedDict):
    query: str
    topic_number: int
    turn_number: int


class Qrel(TypedDict):
    iter: int
    docno: int
    label: int


def load_queries(filepath: str, skip_header: bool = True) -> Dict[str, Query]:
    queries = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i == 0 and skip_header:
                continue
            splitline = line.split(',')
            qid = splitline[0]
            query = splitline[1]
            queries[qid] = clean_query(query)
    return queries


def clean_query(query: str) -> str:
    query = re.sub(r'\W', ' ', query).lower().strip()
    return query


def load_qrels(filepath: str) -> Dict:
    qrels = defaultdict(dict)
    with open(filepath, 'r') as f:
        for line in f:
            splitline = line.split()
            qid = splitline[0]
            docno = splitline[2]
            label = splitline[3]
            qrels[qid][docno] = int(label)
    return qrels


def write_to_trec(
    filepath: str,
    rankings: Dict[str, List[Tuple[str, float]]],
    run_id: str = 'BM25',
    placeholder: str = 'Q0',
    train: bool = False
) -> None:
    with open(filepath, 'w') as f:
        if not train:
            f.write('qid,docid\n')
        for qid, values in rankings.items():
            rank = 1
            for doc_id, score in values.items():
                if train:
                    f.write(
                        f'{qid} {placeholder} {doc_id} {rank} {score} {run_id}\n')
                    rank += 1
                else:
                    f.write(f'{qid},{doc_id}\n')
