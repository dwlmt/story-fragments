''' Script for cluster analysis for story vectors.
'''
import collections
import textwrap
from pathlib import Path
from typing import List, OrderedDict

import fire
import jsonlines
import umap
from joblib import dump
import jsonlines as jsonlines
import numpy
import pandas
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
import multiprocessing

from faerun import Faerun
from PIL import Image

import networkx as nx


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


class ClusterStories(object):
    ''' Outputs datasets to a nested format.
    '''

    def umap(self, data, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine'):
        umap_model = umap.UMAP(
            densmap=True,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
        umap_projections = umap_model.fit_transform(data)
        return umap_projections, umap_model

    def hdbscan(self, data, metric: str ="cosine"):

        if metric == "cosine":
            data = normalize(data, norm='l2')
            metric = "l2"

        #distances_matrix = pairwise_distances(data, metric=metric, n_jobs=multiprocessing.cpu_count() - 1)
        #print(f"Distances shape: {distances_matrix.shape}")

        #print(f"Data Shape: {data.shape}")
        clusterer = hdbscan.HDBSCAN(metric=metric,
                                    core_dist_n_jobs=multiprocessing.cpu_count() - 1)

        clusterer.fit(data)

        #print(f"Cluster: {clusterer.labels_}, {clusterer.probabilities_}")

        return clusterer.labels_.tolist() , clusterer.probabilities_.tolist(), clusterer

    def tmap(self, data):

        import tmap as tm

        dims = data.shape[1]
        samples = data.shape[0]
        enc = tm.Minhash(dims, 42, dims)
        lf = tm.LSHForest(dims * 2, 128)

        lf.batch_add(enc.batch_from_weight_array(data))
        lf.index()

        x, y, source_edges, target_edges, _ = tm.layout_from_lsh_forest(lf)

        return x, y, source_edges, target_edges
        

    def cluster(self,
             src_json: List[str],
             output_dir: str,
             embedding_fields: List[str] = ["retrieved_doc_embedding","generator_enc_embedding","generator_dec_embedding","question_embedding","answer_embedding"],
             metrics: List[str] = ["l2","cosine"],
             local_cluster: bool = False
             ):
        #print(f"Params: {src_json}", {output_dir}, {plot_fields}, {plot_field_names})

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if isinstance(src_json, str):
            src_json = [src_json]

        for field in embedding_fields:

            Path(f"{output_dir}/{field}").mkdir(parents=True, exist_ok=True)

            i = 0

            all_story_dfs_list = []

            for json_file in src_json:
                #print(f"Process: {json_file}")

                with jsonlines.open(json_file) as reader:

                    for obj in reader:

                        story_field_df_list = []

                        if "title" in obj:
                            story_id = obj["title"].replace(".txt","")
                        else:
                            story_id = f"{i}"

                        if "passages" in obj:

                            for p in obj["passages"]:

                                if field in p:
                                    p_dict = {"story_id": i, "seq_num": p["seq_num"], "text": p["text"], "title": story_id}

                                    p_embedding = numpy.array(p[field])
                                    p_dict["embedding"] = p_embedding

                                    #print(p_dict)

                                    story_field_df_list.append(p_dict)

                        story_df = pandas.DataFrame(story_field_df_list)

                        all_story_dfs_list.append(story_df)

                        #print(story_df)
                                
                        i += 1

            all_story_df = pandas.concat(all_story_dfs_list)
            print(f"All stories dataframe: {all_story_df}")

            Path(f"{output_dir}/{field}/").mkdir(parents=True, exist_ok=True)

            stacked_field = numpy.stack(all_story_df["embedding"])

            # Add cluster columns.
            for metric in metrics:
                cluster_labels, cluster_probs, clusterer = self.hdbscan(stacked_field, metric)

                dump(clusterer, f"{output_dir}/{field}/hdbscan_model_{metric}.joblib")

                all_story_df[f"cluster_label_{metric}"] = cluster_labels
                all_story_df[f"cluster_prob_{metric}"] = cluster_probs


            '''
            x_nodes, y_nodes, source_edges, target_edges = self.tmap(data=stacked_field)
            x_edges, y_edges = self.graph_edges_to_lines(x_nodes, y_nodes, source_edges, target_edges)

            text_values = ["<br>".join(textwrap.wrap(t)) for t in all_story_df['text'].tolist()]

            seq_values = all_story_df['seq_num'].tolist()
            title_values = all_story_df['title'].tolist()
            story_id_values = all_story_df['story_id'].tolist()

            id_values = [f"{title} - {seq}" for title, seq in zip(title_values, seq_values)]

            for metric in metrics:

                cluster = [c for c in all_story_df[f"cluster_label_{metric}"]]
                cluster_prob = all_story_df[f"cluster_prob_{metric}"]

                #
                fig = self.create_mst_plot(x_edges, x_nodes, y_edges, y_nodes, id_values, text_values,
                                           cluster, cluster_prob, color_override=None)

                plotly.io.write_html(fig=fig, file=f"{output_dir}/{field}/tmap_cluster_{metric}.html",
                                     include_plotlyjs='cdn',
                                     include_mathjax='cdn', auto_open=False)


            # All stories tmap

            cluster = [c for c in all_story_df[f"cluster_label_cosine"]]
            cluster_prob = all_story_df[f"cluster_prob_cosine"]
            fig = self.create_mst_plot(x_edges, x_nodes, y_edges, y_nodes, id_values, text_values,
                                       cluster=cluster, cluster_prob=cluster_prob, color_override=story_id_values)

            plotly.io.write_html(fig=fig, file=f"{output_dir}/{field}/tmap_story.html",
                                 include_plotlyjs='cdn',
                                 include_mathjax='cdn', auto_open=False)
            '''

            # Per Story plots.
            story_ids = all_story_df.story_id.unique()
            for story_id in story_ids:
                print(f"Story: {story_id}")

                story_df = all_story_df[all_story_df["story_id"] == story_id]

                stacked_field = numpy.stack(story_df["embedding"])

                title = story_df["title"][0]
                Path(f"{output_dir}/{title}/{field}/").mkdir(parents=True, exist_ok=True)

                fig = text_table(story_df)
                plotly.io.write_html(fig=fig, file=f"{output_dir}/{title}/text.html", include_plotlyjs='cdn',
                                     include_mathjax='cdn', auto_open=False)

                x_nodes, y_nodes, source_edges, target_edges = self.tmap(data=stacked_field)

                x_edges, y_edges = self.graph_edges_to_lines(x_nodes, y_nodes, source_edges, target_edges)

                seq_values = story_df['seq_num'].tolist()
                text_values = ["<br>".join(textwrap.wrap(t)) for t in story_df['text'].tolist()]

                id_values = [s for s in seq_values]

                for metric in metrics:

                    if local_cluster:

                        story_df = story_df.copy(deep=True)

                        cluster_labels, cluster_probs, clusterer = self.hdbscan(stacked_field, metric)

                        dump(clusterer, f"{output_dir}/{story_id}/{field}/hdbscan_model_{metric}.joblib")

                        story_df[f"cluster_label_{metric}"] = cluster_labels
                        story_df[f"cluster_prob_{metric}"] = cluster_probs

                    cluster = story_df[f"cluster_label_{metric}"].astype(int)
                    cluster_prob = story_df[f"cluster_prob_{metric}"]

                    fig = self.create_mst_plot(x_edges, x_nodes, y_edges, y_nodes, id_values, text_values,
                                               cluster, cluster_prob)

                    plotly.io.write_html(fig=fig, file=f"{output_dir}/{title}/{field}/tmap_{metric}.html",
                                         include_plotlyjs='cdn',
                                         include_mathjax='cdn', auto_open=False)

    def graph_edges_to_lines(self, x_points, y_points, source_edges, target_edges):
        x_edges = []
        y_edges = []
        for s, t in zip(source_edges, target_edges):
            x_edges.extend([x_points[s], x_points[t], None])
            y_edges.extend([y_points[s], y_points[t], None])
        return x_edges, y_edges

    def network_memory(self,
                src_json: List[str],
                output_dir: str,
                num_central_nodes_to_plot: int = 50,
                centrality_depth_limit: int = 1
                ):
        # print(f"Params: {src_json}", {output_dir}, {plot_fields}, {plot_field_names})

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if isinstance(src_json, str):
            src_json = [src_json]

            i = 0

            for json_file in src_json:
                # print(f"Process: {json_file}")

                with jsonlines.open(json_file) as reader:

                    for obj in reader:

                        if "title" in obj:
                            story_id = obj["title"].replace(".txt","")
                        else:
                            story_id = f"{i}"

                        Path(f"{output_dir}/{story_id}/memory_network/").mkdir(parents=True, exist_ok=True)

                        if "passages" in obj:

                            G = self.construct_graph(obj, story_id)


                            self.add_centrality(G, nx.degree_centrality(G), "degree_centrality")
                            try:
                                self.add_centrality(G, nx.eigenvector_centrality(G, weight="weight"), "eigenvector_centrality")
                            except:
                                pass
                            self.add_centrality(G, nx.closeness_centrality(G), "closeness_centrality")
                            self.add_centrality(G, nx.information_centrality(G, weight="weight"),
                                                                    "information_centrality")
                            self.add_centrality(G, nx.betweenness_centrality(G, weight="weight"), "betweenness_centrality")
                            sorted_centrality = self.add_centrality(G, nx.current_flow_betweenness_centrality(G, weight="weight"),
                                                "betweenness_current_centrality")

                            pos = nx.spring_layout(G)#nx.multipartite_layout(G, align="horizontal")
                            print(pos)

                            plot_name = "memory_network"
                            fig = self._plot_memory_network(G, pos)
                            plotly.io.write_html(fig=fig, file=f"{output_dir}/{story_id}/memory_network/{plot_name}.html",
                                                 include_plotlyjs='cdn',
                                                 include_mathjax='cdn', auto_open=False)

                            nx.write_yaml(G, f"{output_dir}/{story_id}/memory_network/memory_network.yaml")

                            cent_counter = 0
                            for j, (node, centrality) in enumerate(sorted_centrality.items()):

                                if cent_counter == num_central_nodes_to_plot:
                                    continue

                                successors = nx.bfs_successors(G, source=node, depth_limit=centrality_depth_limit)
                                predecessors = nx.bfs_predecessors(G, source=node, depth_limit=centrality_depth_limit)

                                print(f"CENTRALITY NODE: {node}, {predecessors}, {successors}")

                                graph_set = set()
                                for s in successors:
                                    graph_set.add(s[0])

                                    if isinstance(s[1], int):
                                        graph_set.add(s[1])
                                    else:
                                        graph_set.update(s[1])

                                for p in predecessors:
                                    graph_set.add(p[0])

                                    if isinstance(p[1], int):
                                        graph_set.add(p[1])
                                    else:
                                        graph_set.update(p[1])

                                SG = G.subgraph(list(graph_set))
                                pos = nx.spring_layout(G)

                                fig = self._plot_memory_network(SG, pos)
                                plotly.io.write_html(fig=fig,
                                                     file=f"{output_dir}/{story_id}/memory_network/{j}_{node}_memory_network.html",
                                                     include_plotlyjs='cdn',
                                                     include_mathjax='cdn', auto_open=False)

                                cent_counter += 1



                        i += 1

    def add_centrality(self, G, centrality, centrality_field_name):
        sorted_centrality = collections.OrderedDict(
            reversed(sorted(centrality.items(), key=lambda item: item[1])))
        for k, v in sorted_centrality.items():
            print(centrality_field_name,k,v,)
            G.nodes[k][f"{centrality_field_name}"] = v

            print(f"{centrality_field_name}: ")

        return sorted_centrality

    def _plot_memory_network(self, G, pos):
        node_x = []
        node_y = []
        node_colours = []
        hover_text = []
        for (n, data) in G.nodes(data=True):
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            node_colours.append(data["subset"])

            if "text" in data:
                if "title" in data:
                    hover_text.append(
                        f"<b>{abs(n)}</b><br><b>{data['title']}</b> <br><br> {'<br>'.join(textwrap.wrap(data['text']))}")
                else:
                    hover_text.append(f"<b>{abs(n)}</b><br>{'<br>'.join(textwrap.wrap(data['text']))}")
        edge_traces = []
        for (u, v, data) in G.edges(data=True):

            if "doc_scores" in data:
                edge_hover = f"<b>{abs(u)} --> {abs(v)} : {data['weight']}, {data['doc_scores']}</b>"

            else:
                edge_hover = f"<b>{abs(u)} --> {abs(v)}"

            if "text" in data:
                if "title" in data:
                    text = f"<br><b>{data['title']}</b> <br><br> {'<br>'.join(textwrap.wrap(data['text']))}"
                else:
                    text = f"<br>{'<br>'.join(textwrap.wrap(data['text']))}"

                edge_hover += f"<br>{text}"

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            if data["weight"] < 1.0:
                edge_weighting  = data["probability"] * 10
            else:
                edge_weighting = 1.0

            edge_weighting = max(1, edge_weighting )

            edge_trace = go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=edge_weighting, color='#888'),#, color=edge_weighting),
                hoverinfo='text',
                hovertext=edge_hover,
                mode='lines')

            edge_traces.append(edge_trace)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False)
        node_trace.marker.color = node_colours
        node_trace.marker.opacity = 1.0
        node_adjacencies = []
        for n, data in G.nodes(data=True):
            #print(data)
            centrality = max(data["closeness_centrality"] * 20.0, 5.0)

            #centrality = max(min(len(adjacencies[1]) * 1.0, 100.00),10.0)

            node_adjacencies.append(centrality)

        node_trace.marker.size = node_adjacencies
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            template="plotly_white",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

    def construct_graph(self, obj, story_id):
        G = nx.OrderedGraph(story_id=story_id)
        previous_node = None
        for p in obj["passages"]:

            G.add_node(-int(p["seq_num"]), text=p["text"], subset=0)

            if previous_node is not None:
               G.add_edge(int(-previous_node["seq_num"]), -int(p["seq_num"]), weight=1.0, probability=1.0)

            if "memory_id" in p:
                G.add_node(int(p["memory_id"]), text=p["text"], subset=1)
                G.add_edge(-int(p["seq_num"]), int(p["memory_id"]), weight=1.0, probability=1.0)

            if "retrieved_docs" in p:
                retrieved_docs = p["retrieved_docs"]

                for doc in retrieved_docs:

                    if not G.has_node(int(doc["id"])):
                        text = ""
                        title = ""
                        if "text" in doc:
                            text = doc["text"]

                        if "title" in doc:
                            title = doc["title"]

                        G.add_node(int(doc["id"]), text=text, title=title, subset=2)

                    G.add_edge(int(doc["id"]), -int(p["seq_num"]),
                               weight=1.0 - doc["probability"], dot_score=doc["dot_score"], probability=doc["probability"])

            previous_node = p
        return G

    def create_mst_plot(self, x_edges, x_points, y_edges, y_points, id_values, text_values, cluster=None, cluster_prob=None, color_override=None):

        if color_override is not None:
            color = color_override
        else:
            color = cluster

        if cluster is not None:
            text_list = [f"<b>{id}</b> <br><br>Cluster: {c} - {cp}, <br><br>{t}" for id, c, cp, t in
             zip(id_values, cluster, cluster_prob, text_values)]
        else:
            text_list = [f"<b>{id}</b> <br><br>{t}" for id, t in
                         zip(id_values, text_values)]

        edge_trace = go.Scatter(x=x_edges, y=y_edges,
                            mode='lines',
                            line=go.scatter.Line(color='#888', width=2),
                            hoverinfo='none',
                            showlegend=False)
        node_trace = go.Scatter(x=list(x_points), y=list(y_points),
                                   mode='markers',
                                   marker=go.scatter.Marker(size=18,
                                                            line=dict(width=1),
                                                            colorscale="Magma"
                                                            ),
                                   # hover_name=,
                                   text=text_list,
                                   hoverinfo='text',
                                   # textposition='top center',
                                   marker_color=color,
                                   showlegend=False)
        axis_style = dict(title='',
                          titlefont=dict(size=20),
                          showgrid=False,
                          zeroline=False,
                          showline=False,
                          ticks='',
                          showticklabels=False)
        layout = dict(autosize=True,
                      showlegend=False,
                      xaxis=axis_style,
                      yaxis=axis_style,
                      hovermode='closest',
                      template="plotly_white")
        
        fig = go.Figure(dict(data=[edge_trace, node_trace], layout=layout))
        return fig


if __name__ == '__main__':
    fire.Fire(ClusterStories)
