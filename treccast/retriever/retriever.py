from collections import defaultdict
from elasticsearch import Elasticsearch
from typing import Dict, List, Tuple

def get_passages(
    es: Elasticsearch,
    queries: Dict,
    index: str,
    k: int = 100,
) -> Tuple[
    Dict[str, List[Tuple[str, float]]],
    Dict[str, str]
]:
    results = defaultdict(dict)
    contexts = defaultdict(str)
    for qid, query in queries.items():
        try:
            hits = es.search(index=index, q=query, _source=True, size=k)[
                'hits']['hits']
            for hit in hits:
                results[qid][hit['_id']] = hit['_score']
                contexts[hit['_id']] = hit['_source']['body']
        except Exception as e:
            print(f'Error querying Elasticsearch: {e}')
    return results, contexts

