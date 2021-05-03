import collections
import itertools
import os
from pathlib import Path

import fire
import more_itertools
import numpy
from allennlp.predictors import Predictor
from jsonlines import jsonlines
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AlignedEventProcessing(object):

    def __init__(self):
        self.srl_predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
            cuda_device=0)

        '''
        self.coref_predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",
            cuda_device=0)
        '''

        self.sentence_transformer = SentenceTransformer('stsb-roberta-large').cuda()

    def events(self,
               src_json: str,
               output_dir: str,
               srl_batch_size: int = 20,
               tag_batch_size: int = 20,
               vector_batch_size: int = 5,
               match_type: str = "whole",
               min_threshold: float = 0.3,
               more_k_diff_similarity: float = 0.05,
               nearest_k: int = 5,
               earliest_k: int = 3,
               min_sentence_len_chars: int = 20,
               plus_minus_percentile: float = 7.5):

        #output_dir = os.path.dirname(output_file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/books/").mkdir(parents=True, exist_ok=True)

        #import neuralcoref
        #coref = neuralcoref.NeuralCoref(nlp.vocab)
        #nlp.add_pipe(coref, name='neuralcoref')

        objects_to_write = []

        with jsonlines.open(f'{output_dir}/alignment.jsonl', mode='w') as writer:

            with jsonlines.open(src_json) as reader:
                for obj in reader:

                    with jsonlines.open(f"{output_dir}/books/{obj['title'].replace(' ','_')}.jsonl", mode="w") as book_writer:

                        # Dict to map text to vectors.
                        vectors_dict = {}

                        for c, chapter in enumerate(obj["chapters"]):

                            summary_sentences = chapter["summary"]["sentences"]

                            summary_sentences = self.sentence_to_dict(summary_sentences)


                            '''
                            coreference_summary_text = nlp(" ".join(s["text"] for s in summary_sentences))
                            print(coreference_summary_text._.coref_resolved)

                            for sent_dict, sent in zip(summary_sentences, nlp(coreference_summary_text._.coref_resolved).sents):
                                sent_text = sent.text.strip()
                                sent_dict["coref_text"] = sent_text
                                print(sent_dict)
                            '''
                         

                            if match_type != "whole":
                                summary_sentences = self.extract_srl(summary_sentences, srl_batch_size)
                                summary_sentences = self.extract_tags_text(summary_sentences, tag_batch_size)
                            self.fill_embeddings_dict(summary_sentences, vectors_dict, vector_batch_size, match_type)
                            chapter["summary"]["sentences"] = summary_sentences

                            full_text_sentences = chapter["full_text"]["sentences"]

                            full_text_sentences = self.sentence_to_dict(full_text_sentences)

                            '''
                            coreference_full_text = self.coref_predictor.coref_resolved(
                                " ".join(s["text"] for s in full_text_sentences))
                            print(coreference_full_text)

                            coreference_full_text = nlp(" ".join(s["text"] for s in full_text_sentences))
                            print(coreference_full_text._.coref_resolved)
                            

                            for sent_dict, sent in zip(coreference_full_text ,
                                                       nlp(coreference_full_text._.coref_resolved).sents):
                                sent_text = sent.text.strip()
                                sent_dict["coref_text"] = sent_text
                            '''

                            if match_type != "whole":
                                full_text_sentences = self.extract_srl(full_text_sentences, srl_batch_size)
                                full_text_sentences = self.extract_tags_text(full_text_sentences, tag_batch_size)
                            self.fill_embeddings_dict(full_text_sentences, vectors_dict, vector_batch_size, match_type)
                            chapter["full_text"]["sentences"] = full_text_sentences

                            

                            avg_matrix, max_matrix, whole_sentence_matrix = \
                                self.pairwise_similarity_matrix(summary_sentences, full_text_sentences, vectors_dict,
                                                                match_type)


                            if match_type == "avg_srl":
                                matrix_list = avg_matrix
                            elif match_type == "max_srl":
                                matrix_list = max_matrix
                            else:
                                matrix_list = whole_sentence_matrix

                            similarity_tensor = numpy.array([list(val.values()) for val in matrix_list.values()])
                            #print(f"Embedding Tensor Shape: {similarity_tensor.shape}")

                            summary_len, full_text_len = similarity_tensor.shape

                            # Default the salience attributes.
                            for f in full_text_sentences:
                                f["salient"] = False
                                f["salience_score"] = 0.0

                            for sum_index in range(summary_len):

                                summary_percentile = (float(sum_index) / float(summary_len)) * 100.0
                                full_text_min_percentile = max(summary_percentile - plus_minus_percentile, 0.0)
                                full_text_max_percentile = min(summary_percentile + plus_minus_percentile, 100.0)

                                full_text_min_index = int(max(0,(full_text_min_percentile / 100.0) * full_text_len))
                                full_text_max_index = int(min(full_text_len - 1, (full_text_max_percentile / 100.0) * full_text_len))

                                sim_tensor_slice = numpy.copy(similarity_tensor[sum_index,full_text_min_index:full_text_max_index])


                                index_range = full_text_max_index - full_text_min_index

                                if index_range == 0:
                                    continue
                                    
                                top_k_indices = numpy.argpartition(-sim_tensor_slice, range(min(nearest_k, index_range )))[:nearest_k]

                                if len(top_k_indices) > earliest_k:
                                    top_k_indices = top_k_indices[numpy.argpartition(top_k_indices, earliest_k)[:earliest_k]]

                                #num_aligned = 0

                                first_similarity = None
                                for j, k in enumerate(top_k_indices):

                                    full_text_index = min(k + full_text_min_index, full_text_len - 1)

                                    if len(summary_sentences[sum_index]['text']) >= min_sentence_len_chars and len(
                                            full_text_sentences[full_text_index]['text']) >= min_sentence_len_chars:


                                        similarity = sim_tensor_slice[k]

                                        if similarity > min_threshold and (first_similarity is None or similarity > (first_similarity - more_k_diff_similarity)):

                                            if not first_similarity:
                                                first_similarity = similarity

                                            #num_aligned += 1

                                            print(
                                                f"ALIGN {sum_index}: {summary_sentences[sum_index]['text']} WITH {full_text_index}: {full_text_sentences[full_text_index]['text']}, SIMILARITY: {similarity}")

                                            summary_sentence = summary_sentences[sum_index]
                                            if "alignments" not in summary_sentence:
                                                summary_sentence["alignments"] = []
                                            summary_sentence["alignments"].append({"rank": int(j), "similarity": float(similarity), "index": int(full_text_index),
                                                                                   "text": full_text_sentences[full_text_index]['text']})

                                            full_text_sentence = full_text_sentences[full_text_index]
                                            full_text_sentence["salient"] = True

                                            if "salience_score" in full_text_sentence:
                                                full_text_sentence["salience_score"] = float(max(full_text_sentence["salience_score"],similarity))
                                            else:
                                                full_text_sentence["salience_score"] = float(similarity)

                                for f in full_text_sentences:
                                    if "verbs" in f:
                                        del f["verbs"]
                                    if "alignments" in f:
                                        del f["alignments"]

                            book_writer.write({"title": obj["title"], "chapter": c, "sentences": full_text_sentences })

                    objects_to_write.append(obj)

                    writer.write(obj)

    def sentence_to_dict(self, sentences):
        if not isinstance(sentences[0], dict):
            full_text_sentences_dict = []
            for i, s in enumerate(sentences):
                full_text_sentences_dict.append({"seq_num": i, "text": s})
            sentences = full_text_sentences_dict
        return sentences

    def fill_embeddings_dict(self, sentences, vectors_dict, vector_batch_size, match_type):
        for sent_batch in more_itertools.chunked(sentences, n=vector_batch_size):
            embeddings_text = []
            for sent in sent_batch:

                embeddings_text.append(sent["text"])

                if match_type != "whole":

                    for verb in sent["verbs"]:
                        for tag, tag_text in verb["tag_text"].items():
                            if tag_text not in vectors_dict:
                                embeddings_text.append(tag_text)

            embeddings = self.sentence_transformer.encode(embeddings_text)

            for t, e in zip(embeddings_text, embeddings):
                vectors_dict[t] = numpy.expand_dims(e, axis=0)

    def extract_tags_text(self, sentences, vector_batch_size):

        # #print(f"Extract tags: {sentences}")

        returned_sentences = []

        for sent_batch in more_itertools.chunked(sentences, n=vector_batch_size):

            for sent in sent_batch:
                # #print(f"Sent: {sent}")
                if "verbs" in sent:
                    # #print(f"Verbs: {sent['verbs']}")
                    for verb in sent["verbs"]:
                        tags = verb["tags"]
                        distinct_tags = set()
                        for t in tags:
                            if t != "O":
                                # This just remove the the initial I O boundary marker from the type.
                                split_tag = t.split("-")
                                processed_tag = "-".join(split_tag[1:])
                                distinct_tags.add(processed_tag)

                        tag_words_dict = collections.OrderedDict()
                        for t in distinct_tags:
                            # #print(f"Distinct tags{t}")
                            tag_indices = [i for i, tag in enumerate(tags) if t in tag]
                            tag_words_dict[t] = " ".join(list(itertools.compress(sent["words"], tag_indices)))

                        # #print(f"{tag_words_dict}")
                        verb["tag_text"] = tag_words_dict

                returned_sentences.append(sent)

        return returned_sentences

    def extract_srl(self, summary_sentences, srl_batch_size):
        returned_sentences = []
        for sent_batch in more_itertools.chunked(summary_sentences, n=srl_batch_size):
            batch_json = [{"sentence": " ".join(s["text"].split()[:100])} for s in sent_batch]
            #print(batch_json)
            sents_srl = self.srl_predictor.predict_batch_json(batch_json)

            for sent, srl in zip(sent_batch, sents_srl):
                sent = {**sent, **srl}
                #print(f"Updated sentence: {sent}")
                returned_sentences.append(sent)

        return returned_sentences

    def pairwise_similarity_matrix(self, summary_sentences, full_text_sentences, vectors_dict, match_type):

        max_matrix = collections.OrderedDict()
        avg_matrix = collections.OrderedDict()
        whole_sentence_matrix = collections.OrderedDict()

        for summary_sent in summary_sentences:

            avg_matrix[summary_sent["seq_num"]] = {}
            max_matrix[summary_sent["seq_num"]] = {}
            whole_sentence_matrix[summary_sent["seq_num"]] = {}

            for full_sent in full_text_sentences:

                whole_sentence_matrix[summary_sent["seq_num"]][full_sent["seq_num"]] = cosine_similarity(
                    vectors_dict[summary_sent["text"]], vectors_dict[full_sent["text"]]).item()


                if match_type != "whole":
                    sentence_similarity_list = []

                    for summary_verb in summary_sent["verbs"]:
                        for full_verb in full_sent["verbs"]:

                            summary_verb_tags = set(summary_verb["tag_text"].keys())
                            full_verb_tags = set(full_verb["tag_text"].keys())

                            shared_tags = summary_verb_tags.intersection(full_verb_tags)
                            #print(f"Shared tags: {shared_tags}")

                            if len(shared_tags) > 0:

                                tag_similarity_list = []

                                for tag in shared_tags:
                                    summary_text = summary_verb["tag_text"][tag]
                                    summary_embedding = vectors_dict[summary_text]

                                    full_text = full_verb["tag_text"][tag]
                                    full_text_embedding = vectors_dict[full_text]

                                    cosine_sim = cosine_similarity(summary_embedding, full_text_embedding).item()
                                    #print(f"{tag}, {summary_text}, {full_text}, {cosine_sim}")
                                    tag_similarity_list.append(cosine_sim)

                                tag_similarity = sum(tag_similarity_list) / len(tag_similarity_list)
                                sentence_similarity_list.append(tag_similarity)

                            else:
                                sentence_similarity_list.append(0.0)

                    if len(sentence_similarity_list) > 0:
                        avg_matrix[summary_sent["seq_num"]][full_sent["seq_num"]] = sum(
                            sentence_similarity_list) / len(sentence_similarity_list)
                        max_matrix[summary_sent["seq_num"]][full_sent["seq_num"]] = max(sentence_similarity_list)
                    else:
                        avg_matrix[summary_sent["seq_num"]][full_sent["seq_num"]] = 0.0
                        max_matrix[summary_sent["seq_num"]][full_sent["seq_num"]] = 0.0

        return avg_matrix, max_matrix, whole_sentence_matrix


if __name__ == '__main__':
    fire.Fire(AlignedEventProcessing)
