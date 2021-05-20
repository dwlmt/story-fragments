import collections
import os
from pathlib import Path

import fire
import spacy
from jsonlines import jsonlines


class AlignSparknotes(object):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def align(self,
              src_json: str,
              output_file: str):

        output_dir = os.path.dirname(output_file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        chapter_summary_dict = collections.OrderedDict()
        text_dict = collections.OrderedDict()
        book_dict = collections.OrderedDict()
        character_dict = collections.OrderedDict()

        with jsonlines.open(src_json) as reader:
            for obj in reader:

                json_type = obj["type"]

                title = obj["title"]

                if title not in book_dict:
                    book_dict["title"] = collections.OrderedDict()

                    text_dict["title"] = collections.OrderedDict()
                    text_dict["chapters"] = collections.OrderedDict()

                    chapter_summary_dict["title"] = collections.OrderedDict()
                    chapter_summary_dict["title"]["chapters"] = collections.OrderedDict()

                    character_dict["title"] = collections.OrderedDict()
                    character_dict["title"]["characters"] = collections.OrderedDict()

                if json_type == "chapter_full_text" or json_type == "chapter_parallel":
                    text_title = ""
                    if "chapter_title" in obj:
                        text_title = obj["chapter_title"]

                    if "text" in obj:
                        text = obj["text"]

                        if isinstance(text, str):
                            text_dict["chapters"][text_title] = text
                        else:
                            for t_k, t_v in text.items():
                                text_dict["chapters"][t_k] = t_v

                elif json_type == "chapter_summary":

                    text_title = ""
                    if "chapter_title" in obj:
                        text_title = obj["chapter_title"]

                    if "text" in obj:
                        text = obj["text"]

                        if isinstance(text, str):
                            text_dict["chapters"][text_title] = text
                        else:
                            for t_k, t_v in text.items():
                                text_dict["chapters"][t_k] = t_v

                elif json_type == "book":
                    book_dict["title"]["url"] = obj["url"]
                elif json_type == "plot_summary":
                    book_dict["text"] = obj["text"]

        print(chapter_summary_dict)


if __name__ == '__main__':
    fire.Fire(AlignSparknotes)
