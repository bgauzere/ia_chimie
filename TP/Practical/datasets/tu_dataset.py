import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from utils import convert_features_names

def convert_attr_TU(l, table):
    idx = np.argmax(l)
    return table[idx]


class MutagDataset():
    @staticmethod
    def convert_node_attr_MUTAG(l):
        attr_table = {0:'C',1:'N',2:'O',3:'F',4:'I',5:'Cl',6:'Br'}
        return convert_attr_TU(l, attr_table)
    
    
    def __init__(self):
        self.dataset_MUTAG = TUDataset("./datasets","MUTAG",use_node_attr=True,use_edge_attr=True)
        graphs_MUTAG = [to_networkx(g, node_attrs=["x"],
                                    edge_attrs=["edge_attr"],
                                    to_undirected=True,
                                      remove_self_loops=True) for g in self.dataset_MUTAG]
        convert_features_names(graphs_MUTAG,cv_node_attr=self.convert_node_attr_MUTAG,
                       old_name_node="x",new_name_node="atom")
        self.graph_data = graphs_MUTAG
        self.targets = self.dataset_MUTAG.data.y.numpy()
    


