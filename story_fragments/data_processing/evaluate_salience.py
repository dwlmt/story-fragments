''' Script for cluster analysis for story vectors.
'''
import collections
import glob
from pathlib import Path
from random import shuffle

import fire
import jsonlines as jsonlines
import numpy
import pandas
import plotly
import sklearn
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics.pairwise import pairwise_distances

from story_fragments.data_processing.utils import build_salience_override_dict


class EvaluateSalience(object):
    ''' Evaluate salience scores.
    '''

    def evaluate(
            self,
            src_json_glob: str,
            output_dir: str,
            salience_score_filter: float = 0.35,
            salience_override_json: str = None):

        top_k_list: list = [1, 5, 10, 20, "n"]
        plot_together_mapping = {"avg_log_likelihood_salience": "Like-Sal",
                                 "avg_log_likelihood_salience_impact_adj": "Like-Imp-Sal",
                                 "cluster_score": "Clus-Sal",
                                 "avg_log_likelihood_salience_cluster": "Like-Clus-Sal",
                                 "avg_log_likelihood_salience_cluster_imp_adj": "Like-Clus-Imp-Sal",
                                 "avg_log_likelihood_no_ret_diff": "Know-Diff-Sal",
                                 "avg_log_likelihood_no_ret_salience": "No-Know-Sal",
                                 "avg_log_likelihood_swapped_salience": "Swap-Sal",
                                 "sentiment_abs": "Imp",
                                 "generator_enc_embedding_cosine_sim": "Emb-Surp",
                                 "generator_enc_embedding_cosine_sim_salience": "Emb-Sal"}

        src_json = sorted(list(glob.glob(f"{src_json_glob}")))
        print(src_json)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        salience_dict = build_salience_override_dict(salience_override_json, salience_score_filter)

        # print(f"Salience dict: {salience_dict}")

        from datasets import load_metric
        rouge = load_metric("rouge")

        eval_field = ["random", "ascending", "descending"]

        if isinstance(src_json, str):
            src_json = [src_json]

        all_correlation = []
        all_evaluation = []

        i = 0

        for json_file in src_json:
            # #print(f"Process: {json_file}")

            metrics_list = []
            with jsonlines.open(json_file) as reader:
                for chapter_counter, obj in enumerate(reader):

                    try:

                        if "title" in obj:
                            title = obj["title"].replace(".txt", "")
                        else:
                            title = f"{i}"

                        print(f"Evaluate Salience for: {title} - {chapter_counter}")
                        title = f"{title.replace(' ', '_')}_{chapter_counter}"

                        if "passages" in obj:

                            story_evaluation = []
                            correlation_values_dict = collections.OrderedDict()

                            metrics_data = obj["passages"][0]["metrics"]

                            # This is a fix to recreate some impact combinations metrics that were incorrectly run in the model.
                            # There is not impact to results since the changed metrics are just the weighted combination of two others.

                            for p in obj["passages"]:
                                metrics = p["metrics"]

                                if "sentiment" in metrics and "avg_log_likelihood_salience" in metrics and "avg_log_likelihood_salience_cluster" in metrics \
                                        and "avg_log_likelihood_salience_impact_adj" in metrics and "avg_log_likelihood_salience_cluster_imp_adj" in metrics:

                                    s = metrics["sentiment"]

                                    if s < 0:
                                        s *= 2
                                    s = (abs(s) + 1.0)

                                    metrics["avg_log_likelihood_salience_impact_adj"] = metrics[
                                                                                            "avg_log_likelihood_salience"] * s
                                    metrics["avg_log_likelihood_salience_cluster_imp_adj"] = metrics[
                                                                                                 "avg_log_likelihood_salience_cluster"] * s

                            if len(eval_field) == 3:
                                for k in metrics_data.keys():
                                    if k in plot_together_mapping.keys() or ("embedding" in k and (
                                            "cosine" in k or "l2" in k) and "no_ret" not in k and "diff" not in k and "swap" not in k):
                                        eval_field.append(k)

                            eval_field = ["avg_log_likelihood_salience"]
                            # eval_field = ["random", "ascending", "descending"]

                            if "input_sentences" in obj:
                                # #print(f"Input sentences: {obj['input_sentences']}")

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
                                    orig_title = obj["title"]
                                    for sentence in obj["input_sentences"]:
                                        if sentence["text"] in salience_dict[orig_title]:
                                            salience_sentence_dict = salience_dict[orig_title][sentence["text"]]

                                            # print(salience_sentence_dict["similarity"], salience_score_filter)
                                            if float(salience_sentence_dict["similarity"]) > salience_score_filter:
                                                salience_binary_list.append(True)
                                            else:
                                                salience_binary_list.append(False)

                                        else:
                                            salience_binary_list.append(False)

                                total_salient_sentences = len([s for s in salience_binary_list if s == True])
                                # print(f"{title} - num salient sentences: {total_salient_sentences}")
                                # print(f"{salience_binary_list}")

                                gold_salient_text = " ".join([s["text"] for s in obj['input_sentences'] if
                                                              "salient" in s and s["salient"] == True])

                                self.metrics_stats(title, metrics_list, obj, output_dir)

                                if len(salience_binary_list) > 0:

                                    for field in eval_field:

                                        print(f"Evaluate salience for field: {field}")

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

                                        elif field == "ascending":

                                            # Randomly set number of examples.
                                            salient_predictions = ([True] * num_salient_examples) + (
                                                    [False] * (len(salience_binary_list) - num_salient_examples))

                                            salient_text = " ".join(
                                                [s["text"] for s, pred in zip(obj["passages"], salient_predictions) if
                                                 pred == True])


                                        elif field == "descending":

                                            salient_predictions = (
                                                                          [False] * (len(
                                                                      salience_binary_list) - num_salient_examples)) + (
                                                                          [True] * num_salient_examples)

                                            salient_text = " ".join(
                                                [s["text"] for s, pred in zip(obj["passages"], salient_predictions) if
                                                 pred == True])


                                        else:

                                            salient_predictions = []

                                            # #print(f"Counts: {len(salience_binary_list)}, {len(obj['passages'])}")
                                            metrics_values = []
                                            for p in obj["passages"]:
                                                # Calculate rank from expectation.
                                                if field in p["metrics"]:
                                                    metric_value = p["metrics"][field]
                                                else:
                                                    metric_value = 0.0

                                                metrics_values.append(float(metric_value))
                                            # print(f"VALUES {field}:{metrics_values}")

                                            sorted_metric_idx = numpy.argsort(metrics_values)
                                            # print(f"SORTED INDEX: {sorted_metric_idx}")
                                            ranks = [0] * len(sorted_metric_idx)
                                            for i, sor in enumerate(reversed(sorted_metric_idx)):
                                                ranks[sor] = i

                                            for rank in ranks:

                                                if rank < num_salient_examples:
                                                    salient_predictions.append(True)
                                                else:
                                                    salient_predictions.append(False)

                                            # print(f"SALIENT PREDICTIONS {field}: {salient_predictions}")

                                            salient_text = " ".join(
                                                [s["text"] for s, pred in zip(obj["passages"], salient_predictions) if
                                                 pred == True])

                                        def compute_rouge_metrics(label_str, pred_str):

                                            # Compute the metric.
                                            rouge_results = rouge.compute(
                                                predictions=[pred_str],
                                                references=[label_str],
                                                rouge_types=["rouge1", "rouge2", "rougeL"],
                                                use_agregator=True,
                                                use_stemmer=False,
                                            )
                                            return rouge_results['rouge1'].mid.fmeasure, rouge_results[
                                                'rouge2'].mid.fmeasure, rouge_results[
                                                       'rougeL'].mid.fmeasure

                                        rouge1, rouge2, rougeL = compute_rouge_metrics(gold_salient_text, salient_text)

                                        story_evaluation.append(
                                            {"field": field, "title": title, "metric": "rouge1",
                                             "value": rouge1})
                                        story_evaluation.append(
                                            {"field": field, "title": title, "metric": "rouge2",
                                             "value": rouge2})
                                        story_evaluation.append(
                                            {"field": field, "title": title, "metric": "rougeL",
                                             "value": rougeL})

                                        # #print(f"Salient sentences {salience_binary_list}")

                                        # Top K Evaluation.
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
                                                # print(indices, num_salient_examples)
                                                for i in indices[:num_salient_examples]:
                                                    salient_predictions[i] = True

                                            elif field == "ascending":

                                                # Randomly set number of examples.
                                                salient_predictions = ([True] * num_salient_examples) + (
                                                        [False] * (len(salience_binary_list) - num_salient_examples))

                                            elif field == "descending":

                                                salient_predictions = (
                                                                              [False] * (len(
                                                                          salience_binary_list) - num_salient_examples)) + (
                                                                              [True] * num_salient_examples)

                                            else:

                                                salient_predictions = []

                                                # #print(f"Counts: {len(salience_binary_list)}, {len(obj['passages'])}")
                                                metrics_values = []
                                                for p in obj["passages"]:
                                                    # Calculate rank from expectation.

                                                    if field in p["metrics"]:
                                                        metric_value = p["metrics"][field]
                                                    else:
                                                        metric_value = 0.0

                                                    metrics_values.append(float(metric_value))
                                                # print(f"VALUES {field}: {metrics_values}")

                                                sorted_metric_idx = numpy.argsort(metrics_values)
                                                # print(f"SORTED INDEX: {sorted_metric_idx}")
                                                ranks = [0] * len(sorted_metric_idx)
                                                for i, sor in enumerate(reversed(sorted_metric_idx)):
                                                    ranks[sor] = i

                                                for rank in ranks:

                                                    if rank < num_salient_examples:
                                                        salient_predictions.append(True)
                                                    else:
                                                        salient_predictions.append(False)

                                                # print(f"SALIENT PREDICTIONS {field}: {salient_predictions}")

                                            correlation_values_dict[field] = salient_predictions

                                            # Some metrics will have no predictions for the final sentence and so just set to false.
                                            if len(salient_predictions) < len(salience_binary_list):
                                                salience_binary_list = salience_binary_list[: len(salient_predictions)]
                                            elif len(salient_predictions) > len(salience_binary_list):
                                                salient_predictions = salient_predictions[:len(salience_binary_list)]

                                            # print(f"{field}, {salience_binary_list}, {salient_predictions}")
                                            prec, recall, f_score, support = sklearn.metrics.precision_recall_fscore_support(
                                                salience_binary_list, salient_predictions, average="binary")

                                            story_evaluation.append(
                                                {"field": field, "title": title, "metric": f"precision_{top_k}",
                                                 "value": prec})
                                            story_evaluation.append(
                                                {"field": field, "title": title, "metric": f"recall_{top_k}",
                                                 "value": recall})

                                            story_evaluation.append(
                                                {"field": field, "title": title, "metric": f"f_score_{top_k}",
                                                 "value": f_score})

                                        # Map ranking evaluation.
                                        if field == "random":

                                            indices = [i[0] for i in enumerate(salience_binary_list)]
                                            shuffle(indices)

                                            salient_predictions_scores = indices
                                        elif field == "ascending":

                                            salient_predictions_scores = [r for r in range(len(salience_binary_list))]

                                        elif field == "descending":

                                            salient_predictions_scores = list(
                                                reversed([r for r in range(len(salience_binary_list))]))

                                        else:

                                            salient_predictions_scores = []

                                            # #print(f"Counts: {len(salience_binary_list)}, {len(obj['passages'])}")
                                            for p in obj["passages"]:

                                                metrics_data = p["metrics"]

                                                if field in metrics_data:
                                                    salient_predictions_scores.append(
                                                        metrics_data[field])

                                        if len(salient_predictions_scores) < len(salience_binary_list):
                                            salience_binary_list = salience_binary_list[
                                                                   : len(salient_predictions_scores)]
                                        elif len(salient_predictions_scores) > len(salience_binary_list):
                                            salient_predictions_scores = salient_predictions_scores[
                                                                         :len(salience_binary_list)]

                                        map = sklearn.metrics.average_precision_score(salience_binary_list,
                                                                                      salient_predictions_scores)
                                        # print(f"MAP: {map}")

                                        story_evaluation.append(
                                            {"field": field, "title": title, "metric": "map", "value": map})

                                        # Calculate ROUGE score.

                            correlation_list = []
                            keys = correlation_values_dict.keys()
                            for k1 in keys:
                                for k2 in keys:
                                    v1 = correlation_values_dict[k1]
                                    v2 = correlation_values_dict[k2]

                                    if len(v1) > len(v2):
                                        v1 = v1[:len(v2)]
                                    elif len(v2) > len(v1):
                                        v2 = v2[:len(v1)]

                                    kendall, kendall_p = kendalltau(v1, v2)
                                    spearman, spearman_p = spearmanr(v1, v2)
                                    correlation_list.append({"metric_1": k1, "metric_2": k2, "kendall": kendall,
                                                             "kendall_p": kendall,
                                                             "spearman": spearman, "spearman_p": spearman_p})

                            story_correlation_df = pandas.DataFrame(correlation_list)
                            story_correlation_df.to_csv(f"{output_dir}/{title}_correlation.csv")
                            all_correlation.append(story_correlation_df)

                            story_evaluation_df = pandas.DataFrame(story_evaluation)
                            story_evaluation_df.to_csv(f"{output_dir}/{title}_salience_eval.csv")

                            all_evaluation.append(story_evaluation_df)

                    except:
                        pass

        metrics_df = pandas.DataFrame(metrics_list)
        metrics_df = metrics_df.groupby(["metric"]).describe().reset_index()
        metrics_df.to_csv(f"{output_dir}/all_metrics.csv")

        all_evaluation_df = pandas.concat(all_evaluation)
        all_stats_df = all_evaluation_df.groupby(["field", "metric"]).describe().reset_index()
        all_stats_df.reset_index()
        all_stats_df.to_csv(f"{output_dir}/salience_eval.csv")

        all_correlation_df = pandas.concat(all_correlation)
        all_corr_stats_df = all_correlation_df.groupby(["metric_1", "metric_2"]).describe().reset_index()
        all_corr_stats_df.to_csv(f"{output_dir}/correlation.csv")

        all_corr_avg_df = all_correlation_df.groupby(["metric_1", "metric_2"]).mean().reset_index()
        # print(all_corr_avg_df)

        corr_metrics_list = all_correlation_df["metric_1"].unique().tolist()
        z = []
        x = []
        for m1 in corr_metrics_list:
            if m1 in plot_together_mapping.keys():

                x.append(plot_together_mapping[m1])
                z_list_nested = []
                for m2 in corr_metrics_list:

                    if m2 in plot_together_mapping.keys():
                        z_value = all_corr_avg_df.iloc[
                            ((all_corr_stats_df["metric_1"] == m1) & (all_corr_stats_df["metric_2"] == m2)).values][
                            "spearman"].values[0]
                        z_list_nested.append(z_value)

                z.append(z_list_nested)

        y = x

        z_text = numpy.around(z, decimals=2)

        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text)
        plotly.io.write_html(fig=fig, file=f"{output_dir}/correlation_heatmap.html",
                             include_plotlyjs='cdn', include_mathjax='cdn', auto_open=False)

        '''
        z = []
        x = plot_together_mapping.values()
        y = plot_together_mapping.values()
        for m1 in plot_together_mapping.keys():
            z_list_nested = []
            for m2 in  plot_together_mapping.keys():
                z_value = all_corr_avg_df.iloc[((all_corr_stats_df["metric_1"] == m1) & (all_corr_stats_df["metric_2"] == m2)).values]["spearman"].values[0]
                z_list_nested.append(z_value)

            z.append(z_list_nested)

        z_text = numpy.around(z, decimals=2)

        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text)
        plotly.io.write_html(fig=fig, file=f"{output_dir}/selected_fields_correlation_heatmap.html",
                             include_plotlyjs='cdn', include_mathjax='cdn', auto_open=False)
        '''

        # Create a single aggregated heatmap for correlation.

    def metrics_stats(self, title, metrics_list, obj, output_dir):
        # Collect metrics for for statistics.
        chapter_metrics_list = []
        for p in obj["passages"]:
            for k, v in p["metrics"].items():
                metrics_list.append({"metric": k, "value": v})
                chapter_metrics_list.append({"metric": k, "value": v})
        chapter_metrics_df = pandas.DataFrame(chapter_metrics_list)
        chapter_metrics_df = chapter_metrics_df.groupby(["metric"]).describe().reset_index()
        chapter_metrics_df.to_csv(f"{output_dir}/{title}_metrics_stats.csv")


if __name__ == '__main__':
    fire.Fire(EvaluateSalience)
