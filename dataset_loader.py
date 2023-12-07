import utils
import csv
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

#data_h5 = utils.read_h5("frames_data_small.h5")

#print(len(data_h5))

csv_path = "metavd_v1.csv"

class GraphLoader:
    def __init__(self, verbose=False) -> None:
        #each key is a label and each value is the label's id and also an array of the video ids
        self.labels = {}
        #never used but is a set of all used node_ids
        self.all_node_ids = set()
        #next id that will get assigned
        self.cur_id = 0
        #actual graph
        self.G = nx.Graph()
        #verbose printing
        self.verbose = verbose

    #give nodes an id for graph connections
    def get_id(self):
        self.all_node_ids.add(self.cur_id)
        r = self.cur_id
        self.cur_id+=1
        if self.cur_id % 1000:
            print("loaded id", self.cur_id)
        return r

    #iterate all in h5 and create nodes and store set of all nodes of each type
    def create_labels(self, data_h5):
        for key in data_h5.keys():
            l = data_h5[key]['label'].decode('utf-8')
            
            id = self.get_id()
            if l in self.labels:
                self.labels[l]['nodes'].append(id)
            else:
                self.labels[l] = {}
                self.labels[l]['nodes'] = [id]
                label_id = self.get_id()
                self.labels[l]['label_id'] = label_id
                self.G.add_node(
                    label_id,
                    node_type='label',
                    label=l
                )
                
            """ 
            print()
            for attribute in data_h5[key]:
                print(attribute)
                print(type(data_h5[key][attribute]))
            print()
            """
            self.G.add_node(
                id,
                node_type='video', 
                dataset=data_h5[key]['dataset'].decode('utf-8'), 
                label=data_h5[key]['label'].decode('utf-8'),
                split=data_h5[key]['split'].decode('utf-8'),
                num_frames=int(data_h5[key]['num_frames']),
                frame_indices=data_h5[key]['frame_indices'].tolist(),
                frames=data_h5[key]['frames'].tolist(),
                raw_path=data_h5[key]['raw_path'].decode('utf-8'),
                embeddings=data_h5[key]['embeddings'].tolist()
            )
            """
            self.G.add_node(
                id,
                node_type='video', 
                dataset=data_h5[key]['dataset'].decode('utf-8'), 
                label=data_h5[key]['label'].decode('utf-8'),
                split=data_h5[key]['split'].decode('utf-8'),
                num_frames=int(data_h5[key]['num_frames']),
                frame_indices=data_h5[key]['frame_indices'].tolist(),
                frames=data_h5[key]['frames'].tolist(),
                raw_path=data_h5[key]['raw_path'].decode('utf-8'),
                embeddings=data_h5[key]['embeddings'].tolist()
            )
            """
    
    #iterate all sets of types of nodes and create node and interconnect all nodes within the set
    def connect_videos(self):
        for label in self.labels:
            for node_id in self.labels[label]['nodes']:
                self.G.add_edge(self.labels[label]['label_id'], node_id, relationship='video-to-label')
                for other_id in self.labels[label]['nodes']:
                    if node_id != other_id:
                        self.G.add_edge(other_id, node_id, relationship='video-to-video')

    #iterate all csv connections and create virtual node connections
    def connect_labels(self, csv_path):
        df_edges = pd.read_csv(csv_path, usecols=['from_action_name', 'to_action_name', 'relation'])
        for _, edge_data in df_edges.iterrows():
            relation = edge_data['relation']
            if relation != 'is-a':
                if edge_data['from_action_name'] in self.labels and edge_data['to_action_name'] in self.labels:
                    from_id = self.labels[edge_data['from_action_name']]['label_id']
                    to_id = self.labels[edge_data['to_action_name']]['label_id']
                    self.G.add_edge(from_id, to_id, relationship='label-to-label-'+relation)
                else:
                    if self.verbose:
                        if edge_data['from_action_name'] not in self.labels:
                            print(f'{edge_data["from_action_name"]} not in self.labels')
                        if edge_data['to_action_name'] not in self.labels:
                            print(f'{edge_data["to_action_name"]} not in self.labels')

    def create_graph(self, data_h5, csv, export_path):
        self.create_labels(data_h5)
        self.connect_videos()
        self.connect_labels(csv)
        #self.export_graph(export_path)

    #draw the graph
    def draw_graph(self):
        nx.draw(self.G, with_labels=True, font_weight='bold', node_size=300, node_color='skyblue', font_color='black', font_size=8)
        plt.show()

    def export_graph(self, export_path):
        G_export = nx.Graph()
        for node, attrs in self.G.nodes(data=True):
            converted_attrs = {key: str(value) for key, value in attrs.items()}
            G_export.add_node(node, **converted_attrs)

         # Export the new graph to GEXF
        nx.write_graphml(G_export, export_path)


    # def import_graph(self, import_path):
    #     G_loaded = nx.read_graphml(import_path)
    #     for node, attrs in G_loaded.nodes(data=True):
    #         for key, value in attrs.items():




#visualize with plt


#TODO: S

if __name__ == "__main__":
    data_h5 = utils.read_h5("frames_data.h5")
    gl = GraphLoader()
    # gl.create_labels(data_h5)
    # gl.connect_videos()
    # gl.connect_labels('metavd_v1.csv')
    gl.create_graph(data_h5=data_h5, csv='metavd_v1.csv', export_path='metavd_v1.graphml')
    gl.draw_graph()
