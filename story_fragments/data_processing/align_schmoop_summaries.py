import collections
import glob
import os
from pathlib import Path

import fire
import spacy
from blingfire import text_to_sentences
from jsonlines import jsonlines


class AlignSummariesAndText(object):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def align(self,
               summaries_path: str,
               full_text_path: str,
               output_file: str,
               glob_pattern: str= "**/*.txt.utf8"):

        output_dir = os.path.dirname(output_file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        summaries_list = sorted(list(glob.glob(f"{summaries_path}/{glob_pattern}")))
        summaries_dict = self.read_into_structure(summaries_list)

        full_text_list = sorted(list(glob.glob(f"{full_text_path}/{glob_pattern}")))
        full_text_dict = self.read_into_structure(full_text_list)

        books = []
        for k, v in summaries_dict.items():
            if k in full_text_dict:

                summaries = summaries_dict[k]
                texts = full_text_dict[k]

                book = {}
                chapters = []

                print(f"Book: {k}")
                for s_k, s_v in summaries.items():
                    if s_k in texts:

                        print(f"Chapter: {s_k}")

                        summary = summaries[s_k]
                        text = texts[s_k]

                        chapter = {}
                        chapter["seq_num"] = summary["chapter"]
                        #"text" : summary["text"],
                        chapter["summary"] = {"sentences" : summary["sentences"]}
                        #"text": text["text"],
                        chapter["full_text"] = {"sentences": text["sentences"]}
                        book["title"] = summary["title"]
                        chapters.append(chapter)


                book["chapters"] = chapters
                books.append(book)

        with jsonlines.open(f'{output_file}', mode='w') as writer:
            for b in books:
                writer.write(b)


    def read_into_structure(self, summaries_list):

        data_dict = {}

        for sum_file in summaries_list:
            dir_name = os.path.dirname(sum_file)
            dir_name = dir_name.split("/")[-1]
            base_filename = os.path.basename(sum_file)
            base_filename = base_filename.replace(".txt.utf8", "")
            #print(sum_file, dir_name, base_filename)

            with open(sum_file, mode="r", encoding="utf-8") as f:
                text = f.read()
                text = text.replace("\n"," ").replace("\t"," ")

            key = f"{dir_name}"
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append({"title": dir_name, "chapter": int(base_filename), "text": text})

        def get_chapter(e):
            return e['chapter']

        # Sort the chapters
        for k, v in data_dict.items():
            data_dict[k] = sorted(v, key=get_chapter)

        for k, v in data_dict.items():
            for chapter in v:
                text = chapter["text"]
                #doc = self.nlp(text)
                sentences_split = text_to_sentences(text).split('\n')

                sentences = []

                i = 0
                for sent in sentences_split:
                    text = sent.strip()

                    if len(text) > 0:

                        sentence_dict = {}
                        sentence_dict["seq_num"] = i
                        sentence_dict["text"] = text
                        print(sentence_dict)
                        sentences.append(sentence_dict)

                        i += 1

                chapter["sentences"] = sentences


        # Convert list to dict for fast access.
        for k, v in data_dict.items():
            chapter_dict = collections.OrderedDict()
            for chapter in v:
                chapter_dict[chapter["chapter"]] = chapter
            data_dict[k] = chapter_dict


        return data_dict


if __name__ == '__main__':
    fire.Fire(AlignSummariesAndText)