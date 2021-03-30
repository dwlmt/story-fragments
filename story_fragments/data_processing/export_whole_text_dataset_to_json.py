import copy
import glob
import os
import pathlib
import re
from random import Random

import fire
from jsonlines import jsonlines

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

class ExportWholeTextDatasetToJson(object):


    def export(self,
               dataset_directory: str,
               glob_path: str,
               output_file: str):

        from pathlib import Path
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        examples = list(self._generate_examples(dataset_directory, glob_path))

        with jsonlines.open(f'{output_file}', mode='w') as writer:
            for output in examples:
                writer.write(output)

    ''' Outputs datasets to a nested format.
    '''
    def _cleanup_text(self, text):

        text = text.replace("\t", " ")
        text = text.replace("\n", " ")

        if text.startswith('"'):
            text = text[1:]
        if text.endswith('"'):
            text = text[:-1]

        text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()

        return text

    def _generate_examples(self, directory, glob_path):
        glob_target = os.path.join(directory, glob_path)
        book_files = glob.glob(glob_target, recursive=True)

        print(f"Text paths: {book_files}")

        # This uses the Allennlp default random seed to keep validation and testsets the same across runs.
        Random(13370).shuffle(book_files)

        def _reader(book_files):
            _id = 0
            for book_file_path in book_files:
                path = pathlib.PurePath(book_file_path)
                try:
                    with open(book_file_path, mode="r", encoding="utf-8") as f:
                        text = self._cleanup_text(f.read())

                        print(f"Book text: {book_file_path} - {text[: 100]}")
                        glob_dict = {"title": str(path.name), "text": text, "id": _id}
                        yield glob_dict
                        _id += 1
                except Exception as e:
                    print(f"{e}")

        for example in _reader(book_files=book_files):
            yield example

if __name__ == '__main__':
    fire.Fire(ExportWholeTextDatasetToJson)
