import collections
import glob
import os
import textwrap
from pathlib import Path
from typing import List

import fire
import jsonlines as jsonlines
import numpy
import pandas
import plotly
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import RobustScaler

from story_fragments.data_processing.plotly_utils import text_table, create_peak_text_and_metadata
from story_fragments.data_processing.utils import build_salience_override_dict

colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
          'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
          'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

''' Script and function for plotting story predictions output. 
'''


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class PlotStories(object):
    ''' Outputs datasets to a nested format.
    '''

    def plot(self,
             src_json_glob: str,
             output_dir: str,
             salience_override_json: str = None,
             salience_score_filter: float = 0.35,
             plot_peaks: bool = False,
             plot_fields: List[str] = [
                 "perplexity",
                 "avg_log_likelihood",
                 "avg_log_likelihood_salience",
                 "avg_log_likelihood_salience_impact_adj",
                 "cluster_score",
                 "avg_log_likelihood_salience_cluster",
                 "avg_log_likelihood_salience_cluster_imp_adj",
                 "avg_log_likelihood_no_ret_diff",
                 "avg_log_likelihood_no_ret_salience",
                 "avg_log_likelihood_swapped_salience",
                 "sentiment_abs",
                 "generator_enc_embedding_cosine_sim",
                 "generator_enc_embedding_cosine_sim_salience"
             ],
             full_text_fields=["avg_log_likelihood_salience",
                               "avg_log_likelihood_salience_impact_adj",
                               "cluster_score",
                               "avg_log_likelihood_salience_cluster",
                               "avg_log_likelihood_salience_cluster_imp_adj",
                               "avg_log_likelihood_no_ret_diff",
                               "avg_log_likelihood_no_ret_salience",
                               "avg_log_likelihood_swapped_salience",
                               "sentiment_abs",
                               "generator_enc_embedding_cosine_sim",
                               "generator_enc_embedding_cosine_sim_salience"
                               ],
             plot_together_mapping={"avg_log_likelihood_salience": "Like-Sal",
                                    "avg_log_likelihood_salience_impact_adj": "Like-Imp-Sal",
                                    "cluster_score": "Clus-Sal",
                                    "avg_log_likelihood_salience_cluster": "Like-Clus-Sal",
                                    "avg_log_likelihood_salience_cluster_imp_adj": "Like-Clus-Imp-Sal",
                                    "avg_log_likelihood_no_ret_diff": "Know-Diff-Sal",
                                    "avg_log_likelihood_no_ret_salience": "No-Know-Sal",
                                    "avg_log_likelihood_swapped_salience": "Swap-Sal",
                                    "generator_enc_embedding_cosine_sim": "Emb-Surp",
                                    "generator_enc_embedding_cosine_sim_salience": "Emb-Sal"}

             ):

        '''
        ["perplexity",
                                       "avg_log_likelihood",
                                       "avg_log_likelihood_salience_impact_adj",
                                       "cluster_score",
                                       "avg_log_likelihood_salience_cluster",
                                       "avg_log_likelihood_salience_cluster_imp_adj",
                                       "sentiment_abs",
                                       "sentiment",
                                       "generator_enc_embedding_l1_dist",
                                       "generator_enc_embedding_cosine_dist",
                                       "generator_enc_embedding_wasserstein_dist",
                                       "generator_dec_embedding_l1_dist",
                                       "generator_dec_embedding_cosine_dist",
                                       "generator_dec_embedding_wasserstein_dist"
                                       "retrieved_embedding_l1_dist",
                                       "retrieved_doc_embedding_cosine_dist",
                                       "retrieved_doc_embedding_wasserstein_dist",
                                       "question_embedding_cosine_sim",
                                       "question_embedding_l1_dist",
                                       "question_embedding_wasserstein_dist"
                                       ],


        '''

        src_json = sorted(list(glob.glob(f"{src_json_glob}")))
        print(src_json)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        salience_dict = build_salience_override_dict(salience_override_json, salience_score_filter)

        all_stats = []

        for json_file in src_json:
            print(f"Process: {json_file}")

            with jsonlines.open(json_file) as reader:

                chapter_counter = 0

                for obj in reader:

                    chapter_counter += 1

                    if "title" in obj:
                        title = f'{obj["title"].replace(".txt", "")}_{chapter_counter}'
                    else:
                        title = f"{chapter_counter}"

                    print(f"Plot: {title} - {chapter_counter}")

                    Path(f"{output_dir}/{title}").mkdir(parents=True, exist_ok=True)

                    if "passages" in obj:

                        try:

                            passages = obj["passages"]

                            passages_flattened = [flatten(p) for p in passages]

                            passages_df = pandas.DataFrame(passages_flattened)

                            passages_df["hover_text"] = passages_df["seq_num"].astype(
                                str)  # + ": " + passages_df["text"]

                            passages_df.fillna(0)

                            # This is a fix to recreate some impact combinations metrics that were incorrectly run in the model.
                            # There is not impact to results since the changed metrics are just the weighted combination of two others.

                            self._peak_distance = int(os.getenv("PEAK_DISTANCE", default=5))
                            self._peak_prominence = float(os.getenv("PEAK_PROMINENCE", default=0.10))
                            self._peak_threshold = float(os.getenv("PEAK_THRESHOLD", default=0.01))
                            self._peak_height = float(os.getenv("PEAK_HEIGHT", default=0.01))
                            for field in ["avg_log_likelihood_salience_impact_adj",
                                          "avg_log_likelihood_salience_cluster_imp_adj"]:
                                if "avg_log_likelihood_salience_impact_adj" in field and "metrics.avg_log_likelihood_salience" in passages_df.columns:
                                    values = passages_df[f"metrics.avg_log_likelihood_salience"].to_numpy().tolist()
                                elif "avg_log_likelihood_salience_cluster_imp_adj" in field and "metrics.avg_log_likelihood_salience_cluster" in passages_df.columns:
                                    values = passages_df[
                                        f"metrics.avg_log_likelihood_salience_cluster"].to_numpy().tolist()
                                else:
                                    values = None

                                if values is not None:

                                    sentiment = passages_df[f"metrics.sentiment"].to_numpy().tolist()
                                    sentiment_adj = []
                                    for s in sentiment:
                                        if s < 0:
                                            s *= 2
                                        s = (abs(s) + 1.0)
                                        sentiment_adj.append(s)

                                    field_adj = [y * s for (y, s) in zip(values, sentiment_adj)]

                                    passages_df[f"metrics.{field}"] = field_adj

                                    scaler = RobustScaler()
                                    metric_scaled = numpy.squeeze(
                                        scaler.fit_transform(numpy.expand_dims(values, axis=1)),
                                        axis=1)
                                    ##print(f"Metric scales: {metric_scaled}, {metric}")
                                    peaks, properties = find_peaks(metric_scaled, prominence=self._peak_prominence,
                                                                   distance=self._peak_distance,
                                                                   threshold=self._peak_threshold,
                                                                   height=self._peak_height)
                                    peak_series = [False] * len(values)
                                    left_base = [None] * len(values)
                                    right_base = [None] * len(values)
                                    height = [None] * len(values)
                                    prominence = [None] * len(values)

                                    for i, p in enumerate(peaks):
                                        peak_series[p] = True
                                        height[p] = float(properties["peak_heights"][i])
                                        prominence[p] = float(properties["prominences"][i])
                                        left_base[p] = float(properties["left_bases"][i])
                                        right_base[p] = float(properties["right_bases"][i])

                                    passages_df[f"peaks.{field}_peak"] = peak_series
                                    passages_df[f"peaks.{field}_peak_properties.left_base"] = left_base
                                    passages_df[f"peaks.{field}_peak_properties.right_base"] = right_base
                                    passages_df[f"peaks.{field}_peak_properties.prominence"] = prominence
                                    passages_df[f"peaks.{field}_peak_properties.height"] = height

                            fig = text_table(passages_df)
                            plotly.io.write_html(fig=fig, file=f"{output_dir}/{title}/text.html",
                                                 include_plotlyjs='cdn',
                                                 include_mathjax='cdn', auto_open=False)

                            # print(f"{passages_df.seq_num}, {passages_df.text}")

                            if len(plot_together_mapping) > 0:

                                y_values_dict = collections.OrderedDict()

                                fig = go.Figure()

                                for j, (field, name) in enumerate(plot_together_mapping.items()):

                                    x = passages_df["seq_num"]

                                    from sklearn.preprocessing import StandardScaler
                                    if f"metrics.{field}" in passages_df.columns:

                                        y_values_dict[field] = passages_df[f"metrics.{field}"].to_numpy().tolist()

                                        y = passages_df[f"metrics.{field}"].tolist()

                                        # print(field,y)
                                        # Fix for display as embedding fields are degenerate in first/final prediction which skews the plots.
                                        if field in ["generator_enc_embedding_cosine_sim"]:
                                            y[-1] = 1.0
                                            y[0] = 1.0

                                        scaler = RobustScaler()
                                        y = numpy.squeeze(
                                            scaler.fit_transform(numpy.asarray(y).reshape(-1, 1)),
                                            axis=1).tolist()
                                        # y = numpy.squeeze(
                                        #    normalize(numpy.array(passages_df[f"metrics.{field}"]).reshape(1, -1)))

                                        # print(f"Field: {field}, {x}, {y}")

                                        if field in full_text_fields:

                                            text_values = ["<br>".join(textwrap.wrap(t)) for t in
                                                           passages_df['text'].tolist()]

                                            hover_text = [f"<b>{id}</b> <br><br>{t}" for id, t in
                                                          zip(passages_df['seq_num'], text_values)]

                                        else:
                                            hover_text = passages_df["hover_text"]

                                        fig.add_trace(go.Scatter(x=x,
                                                                 y=y,
                                                                 mode='lines+markers',
                                                                 line=dict(color=colors[(j - 1) % len(colors)]),
                                                                 name=f'{name}',
                                                                 line_shape='spline',
                                                                 hovertext=hover_text))

                                        if plot_peaks and f"peaks.{field}_peak" in passages_df.columns:
                                            peak_indices = [p["peaks"][f"{field}_peak"] for p in passages]

                                            if any(peak_indices):
                                                peak_x = [x_item for x_item, peak in zip(x, peak_indices) if
                                                          peak == True]
                                                peak_y = [y_item for y_item, peak in zip(y, peak_indices) if
                                                          peak == True]

                                                peak_properties = [p["peaks"][f"{field}_peak_properties"] for p, peak in
                                                                   zip(passages, peak_indices) if peak == True]

                                                peak_metadata = create_peak_text_and_metadata(peak_properties)

                                                fig.add_trace(go.Scatter(
                                                    x=peak_x,
                                                    y=peak_y,
                                                    mode='markers',
                                                    marker=dict(
                                                        symbol='star-triangle-up',
                                                        size=12,
                                                        color=colors[(j - 1) % len(colors)],
                                                    ),
                                                    name=f'{name} - Peak',
                                                    text=peak_metadata,
                                                ))

                                if len(salience_dict) > 0 and field:
                                    sentences = passages_df['text'].tolist()

                                    salient_points = []
                                    salient_text_labels = []
                                    for sal_idx, s in enumerate(sentences):
                                        orig_title = f'{obj["title"].replace(".txt", "")}'
                                        if s in salience_dict[orig_title]:
                                            salient_points.append(sal_idx)
                                            salient_text_labels.append(
                                                f"{'<br>'.join(textwrap.wrap(salience_dict[orig_title][s]['summary_text']))}"
                                                f"<br><br><b>Alignment Similarity: {salience_dict[orig_title][s]['similarity']}</b><br><br>"
                                                f"{'<br>'.join(textwrap.wrap(s))}")

                                    y = [0.0] * len(salient_points)

                                    trace = go.Scatter(
                                        x=salient_points,
                                        y=y,
                                        mode='markers',
                                        marker=dict(
                                            color="gold",
                                            symbol='star',
                                            size=12,
                                        ),
                                        name=f'Shmoop Label',
                                        hovertext=salient_text_labels,

                                    )
                                fig.add_trace(trace)

                                fig.update_layout(template="plotly_white")

                                fig.update_layout(
                                    xaxis_title="Sentence",
                                    yaxis_title="Salience",
                                )

                                plotly.io.write_html(fig=fig, file=f"{output_dir}/{title}/all.html",
                                                     include_plotlyjs='cdn', include_mathjax='cdn', auto_open=False)

                                # fig.update_layout(showlegend=False)

                                plotly.io.write_image(fig=fig, file=f"{output_dir}/{title}/all.svg")

                                # Extract heatmaps.
                                correlation_list = []
                                z = []
                                keys = y_values_dict.keys()
                                for k1 in keys:
                                    z_list_nested = []
                                    for k2 in keys:
                                        kendall, kendall_p = kendalltau(y_values_dict[k1], y_values_dict[k2])
                                        spearman, spearman_p = spearmanr(y_values_dict[k1], y_values_dict[k2])
                                        correlation_list.append({"metric_1": k1, "metric_2": k2, "kendall": kendall,
                                                                 "kendall_p": kendall,
                                                                 "spearman": spearman, "spearman_p": spearman_p})

                                        z_list_nested.append(spearman)

                                    z.append(z_list_nested)

                                story_correlation_df = pandas.DataFrame(correlation_list)
                                story_correlation_df.to_csv(f"{output_dir}/{title}/correlation.csv")

                                z_text = numpy.around(z, decimals=2)
                                x = list(plot_together_mapping.values())
                                y = list(plot_together_mapping.values())

                                import plotly.figure_factory as ff
                                print(len(x), len(y), len(z), len(z_text))
                                fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text)
                                plotly.io.write_html(fig=fig, file=f"{output_dir}/{title}/all_correlation_heatmap.html",
                                                     include_plotlyjs='cdn', include_mathjax='cdn', auto_open=False)

                            for field in plot_fields:

                                field_type_list = [""]  # , "_no_ret", "_no_ret_diff", "_swapped", "_swapped_diff"]
                                field_type_list += [f"{f}_salience" for f in field_type_list]
                                sub_fields = [f"{field}{ext}" for ext in field_type_list]

                                sub_fields = [sub for sub in sub_fields if f"metrics.{sub}" in passages_df.columns]

                                for j, subfield in enumerate(sub_fields):

                                    single_fig = go.Figure()

                                    x = passages_df["seq_num"]
                                    y = passages_df[f"metrics.{subfield}"].replace(numpy.nan,
                                                                                   0)  # numpy.squeeze(normalize(numpy.array(passages_df[f"metrics.{field}"]).reshape(1, -1)))

                                    print(f"Field: {subfield}, {x}, {y}")

                                    if subfield in full_text_fields:

                                        text_values = ["<br>".join(textwrap.wrap(t)) for t in
                                                       passages_df['text'].tolist()]

                                        hover_text = [f"<b>{id}</b> <br><br>{t}" for id, t in
                                                      zip(passages_df['seq_num'], text_values)]

                                    else:
                                        hover_text = passages_df["hover_text"]

                                    single_fig.add_trace(go.Scatter(x=x,
                                                                    y=y,
                                                                    mode='lines+markers',
                                                                    line=dict(color=colors[(j - 1) % len(colors)]),
                                                                    name=f'{subfield}',
                                                                    line_shape='spline',
                                                                    hovertext=hover_text))

                                    for x_item, y_item in zip(x, y):
                                        all_stats.append({"field": subfield, "value": y_item})

                                    if plot_peaks and f"peaks.{subfield}_peak" in passages_df.columns:
                                        peak_indices = [p["peaks"][f"{subfield}_peak"] for p in passages]

                                        if any(peak_indices):
                                            peak_x = [x_item for x_item, peak in zip(x, peak_indices) if peak == True]
                                            peak_y = [y_item for y_item, peak in zip(y, peak_indices) if peak == True]

                                            peak_properties = [p["peaks"][f"{subfield}_peak_properties"] for p, peak in
                                                               zip(passages, peak_indices) if peak == True]

                                            peak_metadata = create_peak_text_and_metadata(peak_properties)

                                            single_fig.add_trace(go.Scatter(
                                                x=peak_x,
                                                y=peak_y,
                                                mode='markers',
                                                marker=dict(
                                                    symbol='star-triangle-up',
                                                    size=12,
                                                    color=colors[(j - 1) % len(colors)],
                                                ),
                                                name=f'{subfield} - peak',
                                                text=peak_metadata))

                                    if len(salience_dict) > 0:
                                        sentences = passages_df['text'].tolist()

                                        salient_points = []
                                        salient_text_labels = []
                                        for sal_idx, s in enumerate(sentences):
                                            orig_title = f'{obj["title"].replace(".txt", "")}'
                                            if s in salience_dict[orig_title]:
                                                salient_points.append(sal_idx)
                                                salient_text_labels.append(
                                                    f"{'<br>'.join(textwrap.wrap(salience_dict[orig_title][s]['summary_text']))}"
                                                    f"<br><br><b>Alignment Similarity: {salience_dict[orig_title][s]['similarity']}</b><br><br>"
                                                    f"{'<br>'.join(textwrap.wrap(s))}")

                                        y = [0.0] * len(salient_points)

                                        trace = go.Scatter(
                                            x=salient_points,
                                            y=y,
                                            mode='markers',
                                            marker=dict(
                                                color="gold",
                                                symbol='star',
                                                size=11,
                                            ),
                                            name=f'Shmoop Label',
                                            hovertext=salient_text_labels,

                                        )
                                        single_fig.add_trace(trace)

                                    single_fig.update_layout(template="plotly_white")

                                    single_fig.update_layout(
                                        xaxis_title="Sentence",
                                        yaxis_title="Salience",
                                    )

                                    plotly.io.write_html(fig=single_fig, file=f"{output_dir}/{title}/{subfield}.html",
                                                         include_plotlyjs='cdn',
                                                         include_mathjax='cdn', auto_open=False)

                                    single_fig.update_layout(showlegend=False)

                                    plotly.io.write_image(fig=single_fig, file=f"{output_dir}/{title}/{subfield}.svg")
                        except:
                            pass

        all_stats_df = pandas.DataFrame(all_stats)
        all_stats_df_agg = all_stats_df.groupby(["field"]).describe().reset_index()
        all_stats_df_agg.to_csv(f"{output_dir}/metric_stats.csv")


if __name__ == '__main__':
    fire.Fire(PlotStories)
