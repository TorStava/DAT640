import argparse
from typing import Any, Dict, Iterator

from elasticsearch.helpers import parallel_bulk

from treccast.core.collection import ElasticSearchIndex
from treccast.core.util.data_generator import DataGeneratorMixin

DEFAULT_MS_MARCO_DATASET = (
    "../../data/collections/collection.tsv"
)
DEFAULT_INDEX_NAME = 'ms_marco'
DEFAULT_ES_HOST = 'localhost:9200'

_DataIterator = Iterator[dict]


class Indexer(DataGeneratorMixin, ElasticSearchIndex):
    def __init__(
            self,
            index_name: str,
            hostname: str = 'localhost:9200'
    ) -> None:
        super().__init__(
            index_name,
            hostname,
            timeout=120,
            max_retries=10,
            retry_on_timeout=True
        )

    def process_documents(self, data_generator: _DataIterator) -> _DataIterator:
        for document in data_generator:
            document['_index'] = self._index_name
            yield document

    def batch_index(self, data_generator: _DataIterator) -> None:
        for success, info in parallel_bulk(
            self._es,
            data_generator,
            thread_count=12,
            chunk_size=5000,
            max_chunk_bytes=104857600,
            queue_size=6,
        ):
            if not success:
                print("A document failed:", info)


def main(args):
    indexer = Indexer(args.index_name, args.host)
    if args.reset_index:
        indexer.delete_index()

    indexer.create_index()

    data_generator = indexer.generate_data_marco(
        action='indexing',
        filepath=args.ms_marco
    )

    documents = indexer.process_documents(data_generator)
    indexer.batch_index(documents)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="indexer.py")
    parser.add_argument(
        "-i", "--index-name", type=str, default=DEFAULT_INDEX_NAME,
        help="Specifies the index name"
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_ES_HOST,
        help="Specifies the hostname and the port"
    )
    parser.add_argument(
        "-r", "--reset-index", action="store_true", help="Reset index"
    )
    parser.add_argument(
        "-m", "--ms-marco", type=str, nargs="?",
        const=DEFAULT_MS_MARCO_DATASET,
        help="Specifies the path to MS MARCO dataset",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
