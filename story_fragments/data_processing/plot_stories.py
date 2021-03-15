import collections
import textwrap
from pathlib import Path
from typing import List

import fire
import jsonlines
import jsonlines as jsonlines
import numpy
import pandas
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import normalize

from story_fragments.data_processing.plotly_utils import text_table, create_peak_text_and_metadata

colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
          'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
          'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

colors = list(reversed(colors))

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
             src_json: List[str],
             output_dir: str,
             plot_fields: List[str] = ["perplexity","avg_log_likelihood","sentiment"],
             full_text_fields = ["avg_log_likelihood_salience"],
             ):
        #print(f"Params: {src_json}", {output_dir}, {plot_fields}, {plot_field_names})

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if isinstance(src_json, str):
            src_json = [src_json]

        i = 0
        for json_file in src_json:
            print(f"Process: {json_file}")

            with jsonlines.open(json_file) as reader:
                for obj in reader:

                    if "title" in obj:
                        title = obj["title"]
                    else:
                        title = f"{i}"

                    Path(f"{output_dir}/{title}").mkdir(parents=True, exist_ok=True)

                    if "passages" in obj:

                        passages = obj["passages"]

                        passages_flattened = [flatten(p) for p in passages]

                        passages_df = pandas.DataFrame(passages_flattened)

                        passages_df["hover_text"] = passages_df["seq_num"].astype(str) #+ ": " + passages_df["text"]

                        fig = text_table(passages_df)
                        plotly.io.write_html(fig=fig, file=f"{output_dir}/{title}/text.html", include_plotlyjs='cdn', include_mathjax='cdn', auto_open=False)

                        #print(f"{passages_df.seq_num}, {passages_df.text}")

                        for field in plot_fields:

                            field_type_list = ["","_no_ret","_no_ret_diff","_swapped","_swapped_diff"]
                            field_type_list += [f"{f}_salience" for f in field_type_list]
                            sub_fields = [f"{field}{ext}" for ext in field_type_list]

                            sub_fields = [sub for sub in sub_fields if f"metrics.{sub}" in passages_df.columns]

                            if "sentiment" not in field:
                                fig = make_subplots(rows=3, shared_xaxes=True)
                            else:
                                fig = make_subplots(rows=1)

                            #fig = go.Figure()

                            for j, subfield in enumerate(sub_fields):

                                single_fig = go.Figure()

                                if "sentiment" in subfield:
                                    row = 1
                                elif "diff" not in subfield and "salience" not in subfield:
                                    row = 2
                                elif  "salience" not in subfield:
                                    row = 3
                                else:
                                    row = 1

                                x = passages_df["seq_num"]
                                y = passages_df[f"metrics.{subfield}"] #numpy.squeeze(normalize(numpy.array(passages_df[f"metrics.{field}"]).reshape(1, -1)))


                                if subfield in full_text_fields:

                                    text_values = ["<br>".join(textwrap.wrap(t)) for t in passages_df['text'].tolist()]

                                    hover_text = [f"<b>{id}</b> <br><br>{t}" for id, t in
                                                 zip(passages_df['seq_num'], text_values)]

                                else:
                                    hover_text = passages_df["hover_text"]


                                #print(f"XXX: {x}, YYY: {y}")
                                fig.add_trace(go.Scatter(x=x,
                                                         y=y,
                                                         mode='lines+markers',
                                                         line=dict(color=colors[(j - 1) % len(colors)]),
                                                         name=f'{subfield}',
                                                         line_shape='spline',
                                                         hovertext=hover_text),
                                                         col=1,row=row)
                                single_fig.add_trace(go.Scatter(x=x,
                                                         y=y,
                                                         mode='lines+markers',
                                                         line=dict(color=colors[(j - 1) % len(colors)]),
                                                         name=f'{subfield}',
                                                         line_shape='spline',
                                                         hovertext=hover_text))


                                if f"peaks.{subfield}_peak" in passages_df.columns:
                                    peak_indices = [p["peaks"][f"{subfield}_peak"] for p in passages]

                                    if any(peak_indices):

                                        peak_x = [x_item for x_item, peak in zip(x, peak_indices) if peak == True]
                                        peak_y = [y_item for y_item, peak in zip(y, peak_indices) if peak == True]

                                        peak_properties = [p["peaks"][f"{subfield}_peak_properties"] for p, peak in zip(passages, peak_indices) if peak == True]

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
                                            name=f'{subfield} - peak',
                                            text=peak_metadata,
                                        ),col=1,row=row)
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

                                '''
                                single_fig.update_layout(
                                    xaxis=dict(
                                        rangeslider=dict(
                                            visible=True
                                        )
                                    )
                                )'''

                                single_fig.update_layout(template="plotly_white")

                                plotly.io.write_html(fig=single_fig, file=f"{output_dir}/{title}/{subfield}.html",
                                                     include_plotlyjs='cdn',
                                                     include_mathjax='cdn', auto_open=False)


                            '''
                            fig.update_layout(
                                xaxis=dict(
                                    rangeslider=dict(
                                        visible=True
                                    )
                                )
                            )'''

                            fig.update_layout(template="plotly_white")

                            plotly.io.write_html(fig=fig, file=f"{output_dir}/{title}/{field}_all.html", include_plotlyjs='cdn',
                                                 include_mathjax='cdn', auto_open=False)

                    i += 1
if __name__ == '__main__':
    fire.Fire(PlotStories)
