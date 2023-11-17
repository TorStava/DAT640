import logging
from typing import Any, Iterator, List

from treccast.core.util.file_parser import FileParser

_DataIterator = Iterator[Any]


class DataGeneratorMixin:
    def generate_data_marco(self, action: str, filepath: str) -> _DataIterator:
        logging.info("Starting to generate data for MS MARCO passage dataset.")
        for i, line in enumerate(FileParser.parse(filepath)):
            pid, content = line.split("\t")
            if action == "encoding":
                yield (f"MARCO_{pid}", content)
            elif action == "indexing":
                yield {
                    "_id": pid,
                    "body": content,
                }
            else:
                raise RuntimeError(
                    "Cannot generate data. Supported actions: indexing and "
                    "encoding."
                )
            if i % 1000000 == 0:
                logging.info("Generated %s paragraphs", i)
        logging.info("Generation finished. Generated total %s paragraphs.", i)
