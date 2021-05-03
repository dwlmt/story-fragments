''' Script for cluster analysis for story vectors.
'''
import gzip
from pathlib import Path
from random import shuffle
from typing import List

import fire
import jsonlines as jsonlines
import pandas
import sklearn
from sklearn.metrics.pairwise import pairwise_distances

from story_fragments.data_processing.utils import build_salience_override_dict


class EvaluateSalience(object):
    ''' Evaluate salience scores.
    '''

    def evaluate(self, src_json: List[str], output_dir: str, top_k_list=[1, 5, 10, 20, "n"],
                 salience_score_filter: float = 0.325,
                 salience_override_json: str = None):

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        salience_dict = build_salience_override_dict(salience_override_json)

        print(f"Salience dict: {salience_dict}")

        from datasets import load_metric
        rouge = load_metric("rouge")

        rank_fields = ["random", "first", "last"]

        if isinstance(src_json, str):
            src_json = [src_json]

        all_evaluation = []

        i = 0

        for json_file in src_json:
            # print(f"Process: {json_file}")

            if "gz" in json_file:

                def read_jsonl_gz(filename):
                    data = []
                    with gzip.open(filename, 'rb') as fp:
                        j_reader = jsonlines.Reader(fp)

                        for obj in j_reader:
                            data.append(obj)

                    return data

            with jsonlines.open(json_file) as reader:
                for obj in reader:

                    if "title" in obj:
                        title = obj["title"].replace(".txt", "")
                    else:
                        title = f"{i}"

                    if "passages" in obj:

                        story_evaluation = []

                        peaks_data = obj["passages"][0]["peaks"]
                        for k in peaks_data.keys():
                            if "rank" in k and "likelihood_salience" in k:
                                rank_fields.append(k)

                        ##print(f"RANK FIELDS: {rank_fields}")

                        if "input_sentences" in obj:
                            # print(f"Input sentences: {obj['input_sentences']}")

                            salience_binary_list = []
                            if len(salience_dict) == 0:
                                for s in obj['input_sentences']:
                                    if "salient" in s:
                                        if bool(s["salient"]) == True and float(
                                                s["salience_score"]) > salience_score_filter:
                                            salience_binary_list.append(True)
                                        else:
                                            salience_binary_list.append(False)
                            else:
                                title = obj["title"]
                                for sentence in obj["input_sentences"]:
                                    if sentence["text"] in salience_dict[title]:
                                        salience_sentence_dict =  salience_dict[title][sentence["text"]]

                                        if float(salience_sentence_dict["similarity"]) > salience_score_filter:
                                            salience_binary_list.append(True)
                                        else:
                                            salience_binary_list.append(False)

                                    else:
                                        salience_binary_list.append(False)

                            total_salient_sentences = len([s for s in salience_binary_list if s == True])

                            gold_salient_text = " ".join([s["text"] for s in obj['input_sentences'] if
                                                          "salient" in s and s["salient"] == True])

                            if len(salience_binary_list) > 0:

                                for field in rank_fields:

                                    field_cleaned = field.replace("_rank", "")

                                    num_salient_examples = total_salient_sentences

                                    # ROUGE and other text based evaluations

                                    if field == "random":

                                        # Randomly set number of examples.
                                        salient_predictions = [False] * len(salience_binary_list)

                                        indices = [i[0] for i in enumerate(salience_binary_list)]
                                        shuffle(indices)

                                        for i in indices[:num_salient_examples]:
                                            salient_predictions[i] = True

                                        salient_text = " ".join(
                                            [s["text"] for s, pred in zip(obj["passages"], salient_predictions) if
                                             pred == True])

                                    elif field == "first":

                                        # Randomly set number of examples.
                                        salient_predictions = ([True] * num_salient_examples) + (
                                                [False] * (len(salience_binary_list) - num_salient_examples))

                                        salient_text = " ".join(
                                            [s["text"] for s, pred in zip(obj["passages"], salient_predictions) if
                                             pred == True])


                                    elif field == "last":

                                        salient_predictions = (
                                                                      [False] * (len(
                                                                  salience_binary_list) - num_salient_examples)) + (
                                                                      [True] * num_salient_examples)

                                        salient_text = " ".join(
                                            [s["text"] for s, pred in zip(obj["passages"], salient_predictions) if
                                             pred == True])


                                    else:

                                        salient_predictions = []

                                        # print(f"Counts: {len(salience_binary_list)}, {len(obj['passages'])}")
                                        for p in obj["passages"]:

                                            peak_data = p["peaks"]

                                            if field in peak_data:

                                                rank = peak_data[field]

                                                if rank < num_salient_examples:
                                                    salient_predictions.append(True)
                                                else:
                                                    salient_predictions.append(False)

                                        salient_text = " ".join(
                                            [s["text"] for s, pred in zip(obj["passages"], salient_predictions) if
                                             pred == True])

                                    def compute_rouge_metrics(label_str, pred_str):

                                        # Compute the metric.
                                        rouge_results = rouge.compute(
                                            predictions=[pred_str],
                                            references=[label_str],
                                            rouge_types=["rouge2", "rougeL"],
                                            use_agregator=True,
                                            use_stemmer=False,
                                        )
                                        return rouge_results['rouge2'].mid.fmeasure, rouge_results[
                                            'rougeL'].mid.fmeasure

                                    rouge2, rougeL = compute_rouge_metrics(gold_salient_text, salient_text)

                                    story_evaluation.append(
                                        {"field": field_cleaned, "title": title, "metric": "rouge2",
                                         "value": rouge2})
                                    story_evaluation.append(
                                        {"field": field_cleaned, "title": title, "metric": "rougeL",
                                         "value": rougeL})

                                    # print(f"Salient sentences {salience_binary_list}")

                                    # Top K Evaluation
                                    for top_k in top_k_list:

                                        if top_k != "n":
                                            num_salient_examples = top_k  #
                                        else:
                                            num_salient_examples = total_salient_sentences

                                        if field == "random":

                                            # Randomly set number of examples.
                                            salient_predictions = [False] * len(salience_binary_list)

                                            indices = [i[0] for i in enumerate(salience_binary_list)]
                                            shuffle(indices)
                                            print(indices)
                                            for i in indices[:num_salient_examples]:
                                                salient_predictions[i] = True

                                        elif field == "first":

                                            # Randomly set number of examples.
                                            salient_predictions = ([True] * num_salient_examples) + (
                                                    [False] * (len(salience_binary_list) - num_salient_examples))

                                        elif field == "last":

                                            salient_predictions = (
                                                                          [False] * (len(
                                                                      salience_binary_list) - num_salient_examples)) + (
                                                                          [True] * num_salient_examples)

                                        else:

                                            salient_predictions = []

                                            # print(f"Counts: {len(salience_binary_list)}, {len(obj['passages'])}")
                                            for p in obj["passages"]:

                                                peak_data = p["peaks"]

                                                if field in peak_data:

                                                    rank = peak_data[field]

                                                    if rank < num_salient_examples:
                                                        salient_predictions.append(True)
                                                    else:
                                                        salient_predictions.append(False)

                                        # Some metrics will have no predictions for the final sentence and so just set to false.
                                        if len(salient_predictions) < len(salience_binary_list):
                                            salience_binary_list = salience_binary_list[: len(salient_predictions)]
                                        elif len(salient_predictions) > len(salience_binary_list):
                                            salient_predictions = salient_predictions[:len(salience_binary_list)]

                                        print(f"{field}, {salience_binary_list}, {salient_predictions}")
                                        prec, recall, f_score, support = sklearn.metrics.precision_recall_fscore_support(
                                            salience_binary_list, salient_predictions, average="binary")

                                        story_evaluation.append(
                                            {"field": field_cleaned, "title": title, "metric": f"precision_{top_k}",
                                             "value": prec})
                                        story_evaluation.append(
                                            {"field": field_cleaned, "title": title, "metric": f"recall_{top_k}",
                                             "value": recall})

                                        story_evaluation.append(
                                            {"field": field_cleaned, "title": title, "metric": f"f_score_{top_k}",
                                             "value": f_score})

                                    # Map ranking evaluation.
                                    if field == "random":

                                        indices = [i[0] for i in enumerate(salience_binary_list)]
                                        shuffle(indices)

                                        salient_predictions_scores = indices
                                    elif field == "first":

                                        salient_predictions_scores = [r for r in range(len(salience_binary_list))]

                                    elif field == "last":

                                        salient_predictions_scores = list(
                                            reversed([r for r in range(len(salience_binary_list))]))

                                    else:

                                        salient_predictions_scores = []

                                        # print(f"Counts: {len(salience_binary_list)}, {len(obj['passages'])}")
                                        for p in obj["passages"]:

                                            metrics_data = p["metrics"]

                                            if field.replace("_rank", "") in metrics_data:
                                                salient_predictions_scores.append(
                                                    metrics_data[field.replace("_rank", "")])

                                    if len(salient_predictions_scores) < len(salience_binary_list):
                                        salience_binary_list = salience_binary_list[: len(salient_predictions_scores)]
                                    elif len(salient_predictions_scores) > len(salience_binary_list):
                                        salient_predictions_scores = salient_predictions_scores[
                                                                     :len(salience_binary_list)]

                                    map = sklearn.metrics.average_precision_score(salience_binary_list,
                                                                                  salient_predictions_scores)
                                    print(f"MAP: {map}")

                                    story_evaluation.append(
                                        {"field": field_cleaned, "title": title, "metric": "map", "value": map})

                                    # Calculate ROUGE score.

                        story_evaluation_df = pandas.DataFrame(story_evaluation)
                        story_evaluation_df.to_csv(f"{output_dir}/{title}_salience_eval.csv")

                        all_evaluation.append(story_evaluation_df)

        all_evaluation_df = pandas.concat(all_evaluation)
        all_stats_df = all_evaluation_df.groupby(["field", "metric"]).describe()
        all_stats_df.reset_index()

        # all_stats_df = pandas.DataFrame(group.describe().rename(columns={'name': name}).squeeze()
        #                   for name, group in all_evaluation_df.groupby(['field','metric']))
        all_stats_df.to_csv(f"{output_dir}/salience_eval.csv")

if __name__ == '__main__':
    fire.Fire(EvaluateSalience)
