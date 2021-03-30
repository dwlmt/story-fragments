''' Script for cluster analysis for story vectors.
'''
import collections
import glob
import os
import textwrap
from pathlib import Path
from typing import List, OrderedDict

import fire
import jsonlines
import xml


class ConvertPropLearner(object):
    ''' Convert the ProppLearner XML file to json that can be evaluated against using the RAG model.
    '''

    def convert(self,
             src_path: str,
             output_file: str,
             glob_pattern: str = '*.sty'
             ):
        #print(f"Params: {src_json}", {output_dir}, {plot_fields}, {plot_field_names})

        #Path(os.path.basename(output_file)).mkdir(parents=True, exist_ok=True)

        with jsonlines.open(f"{output_file}", mode='w') as writer:

            for i, file in enumerate(list(glob.glob(f"{src_path}/{glob_pattern}"))):
                print(f"Process {file}")

                story_json = {}

                story_json["title"] = os.path.basename(file).replace(".sty","")
                story_json["id"] = f"{i}"

                import xml.etree.ElementTree as ET
                tree = ET.parse(file)
                root = tree.getroot()

                for child in root:
                    print(child.tag, child.attrib)

                text_id = "edu.mit.parsing.token"
                story_tokens = tree.find(f"//rep[@id = '{text_id}']")

                token_dict = {}
                offset_to_token_dict = {}
                sentences_dict = collections.OrderedDict()
                offset_to_sentence_dict = {}
                token_to_sentence_dict = {}

                for k, elem in enumerate(story_tokens.findall(".//desc")):

                    token_dict[int(elem.attrib.get("id"))] = {"id": int(elem.attrib.get("id")),
                                                              "len": int(elem.attrib.get("len")),
                                                        "off": int(elem.attrib.get("off")),
                                                              "text": elem.text}

                    for j in range(int(elem.attrib.get("off")),int(elem.attrib.get("off")) + int(elem.attrib.get("len"))):
                        offset_to_token_dict[j] = int(elem.attrib.get("id"))

                text_id = "edu.mit.parsing.sentence"
                story_sentences = tree.find(f"//rep[@id = '{text_id}']")
                for k, elem in enumerate(story_sentences.findall(".//desc")):
                    sentence_dict = {"seq_num": k,"id": int(elem.attrib.get("id")), "len": int(elem.attrib.get("len")),
                                                        "off": int(elem.attrib.get("off")),
                                     "salient": False, "salience_score": 0.0}

                    for j in range(int(elem.attrib.get("off")),
                                   int(elem.attrib.get("off")) + int(elem.attrib.get("len"))):
                        offset_to_sentence_dict[j] = k

                    sentence_token_list = []
                    sentence_token_ids = []
                    text_indices = elem.text.split("~")
                 
                    for ind in text_indices:

                        sentence_token_list.append(token_dict[int(ind)]["text"])
                        sentence_token_ids.append(token_dict[int(ind)]["id"])

                    for id in sentence_token_ids:
                        token_to_sentence_dict[id] = k

                    sentence_dict["text"] = " ".join(sentence_token_list).strip()
                    sentence_dict["tokens"] = sentence_token_list

                    sentences_dict[k] = sentence_dict

                print(sentences_dict)

                text_id = "edu.mit.semantics.rep.event"
                events = tree.find(f"//rep[@id = '{text_id}']")
                for elem in events:
                    print("Events: ", elem.tag, elem.attrib, elem.text)

                text_id = "edu.mit.semantics.rep.function"
                events = tree.find(f"//rep[@id = '{text_id}']")
                for elem in events:

                    print("Functions: ", elem.tag, elem.attrib, elem.text)

                    offset = elem.attrib.get("off")

                    if offset is not None:

                        for j in range(int(elem.attrib.get("off")),
                                       int(elem.attrib.get("off")) + int(elem.attrib.get("len"))):
                            print(f"Offset: {j}")
                            if j in offset_to_sentence_dict:
                                sentence = int(offset_to_sentence_dict[j])
                                print(f"Offset: {j}, Sentence: {sentence}")
                                sentences_dict[sentence]["salient"] = True
                                sentences_dict[sentence]["salience_score"] = 1.0

                            break # Always break after the first offset.

                text_id = "edu.mit.semantics.semroles"
                events = tree.find(f"//rep[@id = '{text_id}']")
                for elem in events:
                    print("Semantic Roles: ", elem.tag, elem.attrib, elem.text)

                #print(offset_to_sentence_dict)


                sentences = []
                for k, v in sentences_dict.items():
                    sentences.append({"seq_num": v["seq_num"], "text": v["text"], "tokens": v["tokens"],
                                      "salient": v["salient"], "salience_score": v["salience_score"]})

                story_json["sentences"] = sentences

                writer.write(story_json)





if __name__ == '__main__':
    fire.Fire(ConvertPropLearner)
