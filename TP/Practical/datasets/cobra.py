
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pathlib

class CobraDataset():
    # Path to the dataset relative to current file
    _DATASET_COBRA = os.path.join(pathlib.Path(__file__).parent.absolute(), "cobradat_nonnan.mat")
    #_DATASET_COBRA = "./datasets/cobradat_nonnan.mat"

    def __init__(self):
        self.data_dict = self._load_cobra_mat_file(self._DATASET_COBRA)
        self.graph_data = self._get_clean_cobra_graph_data()
        self.atom_data = self._get_clean_cobra_atom_data()
        self.targets = self.data_dict['graph_electrophilicity']

    @property
    def all_data(self):
        return np.concatenate((self.graph_data,self.atom_data),axis=1)
    
    def load_data(self, mode="graph"):
        """
        mode : graph, atom, both
        """
        if mode == "graph":
            return self.graph_data, self.targets
        elif mode == "atom":
            return self.atom_data, self.targets
        elif mode == "both":
            return self.all_data, self.targets
        else:
            raise ValueError("mode must be in ['graph','atom','both']")
        
    def _get_clean_cobra_atom_data(self):
        def transform_graph_features(atom_features):
            atom_features = np.delete(atom_features, 17, axis=1)  # delete column 17
            atom_features = atom_features[:,4:]
            d = atom_features.shape[1]
            graph_rep_atom = []
            for f in atom_features.T:
                graph_rep_atom.extend([np.min(f),
                                    np.max(f),
                                    np.mean(f),
                                    np.std(f)])
            return np.array(graph_rep_atom).reshape(d*4,)

        X_cobra_atom = np.array([transform_graph_features(x) for x in self.data_dict['atom_features']])
        scaler = StandardScaler()
        X_cobra_atom = scaler.fit_transform(X_cobra_atom)
        return X_cobra_atom

    def _get_clean_cobra_graph_data(self):
        # Retrieve paper experiments
        X_cobra_graph = np.array([x[:17] for x in self.data_dict['graph_features']])
        n=X_cobra_graph.shape[0]
        families = np.zeros((n,3)) # MV, AV, CV
        for i,name in enumerate(self.data_dict['file_names']):
            if "MV" in name:
                families[i,0] = 1
            elif "AV" in name:
                families[i,1] = 1
            elif "CV" in name:
                families[i,2] = 1
            else:
                print("Error in file name")
                break
        # concatenate with the graph features
        scaler = StandardScaler()
        X_cobra_graph = scaler.fit_transform(X_cobra_graph)
        
        X_cobra_graph = np.concatenate((X_cobra_graph,families),axis=1)
        return X_cobra_graph

    @staticmethod
    def _load_cobra_mat_file(fname='./datasets/cobradat.mat'):
        """
        @author: Muhammet Balcilar
        LITIS Lab, Rouen, France
        muhammetbalcilar@gmail.com


        This script is the demonstration of how we can export extracted Cobra Dataset.
        You can write your own code by reading data from given dataset as well but we recommended you use our provided mat file

        Dataset consist of 111 different molecules graph connections global and atomic descriptors.

        Dataset consist of A,C,F,TT,Atom,Anames,Vnames,FILE,NAME variables. Here is their explanations.

        A :     List of Agencency matrix; It consist of 111  variable size of binary valued matrix
        C :     List of Connectivity matrix; It consist of 111 variable size of double valued matrix
        F :     111x28 dimensional matrix keeps the global moleculer descriptor of each molecule
        TT:     111 element of list. Each element is also matrix by number of atom of corresponding molecule row but 54 column
        Atom:   111 element list. Each element also differnet length of list as well. Keeps the atom names. Forinstance Atom[0][0] shows theh name of the atom of 1st molecules 1st atom.
        Anames: 54 length list ; keeps the name of atomic descriptor. Since we have 54 atomic descriptor it consist of 54 string
        Vnames: 28 length list ; keeps the name of global descriptor. Since we have 28 global descriptor it consist of 28 string
        FILE:   111 element of list. Keeps the file name of corresponding molecule
        NAME:   111 element of list. Keeps the molecule name of corresponding molecule

        """

        # read mat file
        mat = loadmat(fname)
        # make Adjagency and Connectivity matrixes as list
        A=[];C=[]
        for i in range(0,mat['A'].shape[0]):
            A.append(mat['A'][i][0])
            C.append(mat['C'][i][0])
        # read global features descriptors
        F=mat['F']

        # read global descr names
        Vnames=[]
        for i in range(0,mat['Vnames'][0].shape[0]):
            Vnames.append(mat['Vnames'][0][i][0])

        # read file name and molecule names
        FILE=[];NAME=[]
        for i in range(0,mat['FILE'].shape[0]):
            FILE.append(mat['FILE'][i][0][0])
            NAME.append(mat['NAME'][i][0][0])

        # read atomic descriptor name
        Anames=[]
        for i in range(0,mat['Anames'].shape[1]):
            Anames.append(mat['Anames'][0][i][0])
        # read atomic descriptors
        TT=[];Atom=[]
        for i in range(0,mat['TT'].shape[0]):
            TT.append(mat['TT'][i][0])
            SA=[]
            for j in range(0,mat['Atom'][i][0].shape[0]):
                SA.append(mat['Atom'][i][0][j][0][0])
            Atom.append(SA)
        #TT Atom Anames 
        data_dict = {'adjacency_matrices': A,   
                    'connectivity_matrices': C,
                    'graph_features': F[:,:-1],
                    'graph_electrophilicity': F[:,-1],
                    'atom_features': TT,
                    'atom_names': Atom,
                    'atom_features_names': Anames,
                    'graph_features_names': Vnames[:-1],
                    'file_names': FILE,
                    'mol_names': NAME}
        return data_dict
