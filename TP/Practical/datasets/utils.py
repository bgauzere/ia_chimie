import networkx as nx
from rdkit import Chem
from rdkit.Chem import SanitizeMol
from rdkit.Chem import AllChem
import numpy as np

def convert_features_names(graphs,
                            cv_node_attr = None,
                            old_name_node = "node",
                            new_name_node = "atom",
                            cv_edge_attr = None,
                            old_name_edge = "bond_type",
                            new_name_edge = "bond_type",
                            ):
    ## Effet de bord !!
    for g in graphs:
        if cv_node_attr is not None:
            new_values = {n:cv_node_attr(attr[old_name_node]) for n,attr in g.nodes(data=True)}
            nx.set_node_attributes(g, name=new_name_node, values=new_values)
        if cv_edge_attr is not None:
            new_edge_values = {(u,v):cv_edge_attr(attr[old_name_edge]) for u,v,attr in g.edges(data=True)}
            nx.set_edge_attributes(g, name=new_name_edge, values=new_edge_values)



def nx_to_rdkit(graph):
    '''
    Le graph doit etre au format RDKIT'''


    # Créer une molécule vide avec RDKit
    mol = Chem.RWMol()

    # Dictionnaire pour conserver une correspondance entre les nœuds de NetworkX et les atomes de RDKit
    node_to_atom = {}

    # Ajouter les atomes
    for node, data in graph.nodes(data=True):
        atom = Chem.Atom(data['atom_symbol'])
        atom_idx = mol.AddAtom(atom)
        node_to_atom[node] = atom_idx

    # Ajouter les liaisons
    for u, v, data in graph.edges(data=True):
        start_idx = node_to_atom[u]
        end_idx = node_to_atom[v]
        bond_type = data['bond_type']
        mol.AddBond(start_idx, end_idx, bond_type)

    # Convertir en molécule finale et retourner
    return mol.GetMol()

def mol_to_fp(mol):


    mol.UpdatePropertyCache()
    SanitizeMol(mol)
    # Calculer le Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    # Convertir le fingerprint en numpy.array
    fp_array = np.array(list(fp))

    # Afficher le fingerprint  
    return fp_array

