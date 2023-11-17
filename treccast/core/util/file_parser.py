from typing import Iterator


class FileParser:
    @staticmethod
    def parse(filepath: str) -> Iterator[str]:
        return FileParser._parse_text_file(filepath)

    @staticmethod
    def _parse_text_file(filepath):
        with open(filepath, mode='r') as f:
            for line in f:
                yield line.strip()
