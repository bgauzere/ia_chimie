import os
import pathlib
import numpy as np 
from rdkit import Chem

from gklearn.dataset import DataLoader

from utils import convert_features_names, mol_to_fp,nx_to_rdkit


class AcyclicDataset():
    _dataset_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "Acyclic/dataset_bps.ds")
                                 
    @staticmethod
    def convert_edge_attr_acyclic(l):
        attr_table = {'0':Chem.BondType.AROMATIC,
                    '1':Chem.BondType.SINGLE,
                    '2':Chem.BondType.DOUBLE,
                    '3':Chem.BondType.TRIPLE}
        return attr_table[l] 
        
    def __init__(self):
        self.dataset_Acyclic = DataLoader(AcyclicDataset._dataset_path)
        self.graphs = self.dataset_Acyclic.graphs
        convert_features_names(self.graphs,
                                cv_node_attr=None,
                                cv_edge_attr=AcyclicDataset.convert_edge_attr_acyclic)

        self.targets = np.array(self.dataset_Acyclic.targets)
        self.graph_embeddings = None

    @property
    def graph_data(self):
        if self.graph_embeddings is None:
            self.graph_embeddings = np.array([mol_to_fp(nx_to_rdkit(g)) for g in self.graphs])
        return self.graph_embeddings
        