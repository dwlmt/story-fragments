import pandas
import plotly
import plotly.graph_objects as go

def create_peak_text_and_metadata(peaks_meta):
    hover_text = []

    for i, ind in enumerate(peaks_meta):

        prominence = ind["prominence"]
        left_base = ind["left_base"]
        right_base = ind["right_base"]

        text = f"<br>Prominence: {prominence}, <br>Left Base: {left_base}, <br>Right Base: {right_base}"

        hover_text.append(text)
    return hover_text

def text_table(passages_df):
    fig = go.Figure(data=[go.Table(
        columnorder=[1, 2],
        columnwidth=[20, 400],
        header=dict(
            values=[['<b>Seq Num</b>'],
                    ['<b>Text</b>']],
            line_color='darkslategray',
            fill_color='maroon',
            align=['left', 'left'],
            font=dict(color='white', size=16),
            height=40
        ),
        cells=dict(
            values=[passages_df.seq_num, passages_df.text],
            line_color='darkslategray',
            fill=dict(color=['white', 'white']),
            align=['left', 'left'],
            font_size=16,
            height=40)
    )
    ])
    return fig