from typing import Any, Dict

from elasticsearch.client import Elasticsearch

ES_DEFAULT_SETTINGS = {
    'similarity': {
        'default': {
            'type': 'BM25',
            'b': 0.75,
            'k1': 1.2
        }
    },
    'max_result_window': 100000
}

ES_DEFAULT_MAPPINGS = {
    '_source': {
        'includes': ['body']
    }
}


class ElasticSearchIndex:
    def __init__(
        self,
        index_name: str,
        hostname: str = 'localhost:9200',
        **kwargs
    ) -> None:
        """Interface to Elasticsearch index.

        Parameters
        ----------
        index_name : str
            Name of the index
        """
        super().__init__()
        self._index_name = index_name
        self._es = Elasticsearch(hostname, **kwargs)

    def create_index(self) -> None:
        if not self._es.indices.exists(self._index_name):
            self._es.indices.create(
                self._index_name,
                {
                    'settings': ES_DEFAULT_SETTINGS,
                    'mappings': ES_DEFAULT_MAPPINGS
                }
            )
            print('New Index: ', self._index_name, '\n',
                  self._es.indices.get_settings()[self._index_name])

    def delete_index(self) -> None:
        if self._es.indices.exists(self._index_name):
            self._es.indices.delete(self._index_name)

    def index(self, collection: Dict[str, Any], settings: Dict[str, Any]):
        if self._es.indices.exists(index=self._index_name):
            self._es.indices.delete(index=self._index_name)

        self._es.indices.create(index=self._index_name, mappings=settings)

        for doc_id, doc in collection.items():
            self._es.index(document=doc, id=doc_id, index=self._index_name)
