from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOMIC_FEATURE_DICT = {}

ATOMIC_FEATURE_DICT["Symbol"] = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
    "H",
    "Other",
]
ATOMIC_FEATURE_DICT["Weave_Allowed_Symbol"] = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
    "H",
    "Metal",
]
ATOMIC_FEATURE_DICT["Weave_Metal_Symbol"] = [
    "Si",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
]

ATOMIC_FEATURE_DICT["Degree"] = list(np.arange(10))

ATOMIC_FEATURE_DICT["ImplicitValence"] = list(np.arange(-1, 7))

ATOMIC_FEATURE_DICT["AtomicNum"] = list(np.arange(1, 120))
# -6 to +6 should be enough
ATOMIC_FEATURE_DICT["FormalCharge"] = list(np.arange(-6, +7))

ATOMIC_FEATURE_DICT["ChiralTag"] = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

ATOMIC_FEATURE_DICT["Hybridization"] = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.OTHER,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
ATOMIC_FEATURE_DICT["Weave_Hybridization"] = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
ATOMIC_FEATURE_DICT["TotalNumHs"] = [0, 1, 2, 3, 4, "Other"]  # type: ignore
ATOMIC_FEATURE_DICT["CIPCode"] = ["R", "S"]

ATOMIC_FEATURE_DICT["HDonor_Smarts"] = [
    "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]"
]
ATOMIC_FEATURE_DICT["HAcceptor_Smarts"] = [
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@["
    "O,N,P,S])]),$([nH0,o,s;+0])]"
]


def one_of_k_encoding(item: Any, allowable_set: list, unk_strat="last"):
    """
    One of k one hot encoder

    Parameters
    ----------
    item: Any
        any object such as int or list
    allowable_set: list
        list of possible values that the item can take
    unk_strat: str
        Strategy for handling unknown. If "last" then the the
        item is assigned to last. If "null" returned list is all zeros

    Returns
    -------
        Array of 1 hot encoded values.
    """
    # Based off https://github.com/deepchem/torchchem/
    # blob/master/torchchem/data/mol_graph.py

    if item not in allowable_set:
        if unk_strat == "last":
            item = allowable_set[-1]
        # this will result in return of zero vector
        elif unk_strat == "null":
            pass
        else:
            raise Exception(
                "input {0} not in allowable set{1}:".format(item, allowable_set)
            )
    return list(map(lambda s: int(item == s), allowable_set))


def get_one_hot_atom_features(
    atom,
    one_hot_atom_feature_list: list = ["AtomicNum", "ChiralTag"],
    unk_strat: str = "last",
):
    """
    Utility function to get 1-hot atom features for rdkit mol given a list of
    bond features. These features have to be defined in constants.py.
    They can be mixed with other features as needed.

    Parameters
    ----------
    mol:rdkit mol
        rdkit mol object
    atom_feature_list: list
        list of atomic features to comput. We match the string to Rdkit using
        ```"Get{}".format(feature_name)```.
    unk_strat: str
        Strategy for handling unknown values in the one hot. "last" corresponds to
        using the last available one hot index.

    Returns
    -------
    np.array
        The Atomic features

    """
    return np.concatenate(
        [
            one_of_k_encoding(
                getattr(atom, "Get{}".format(af))(), ATOMIC_FEATURE_DICT[af], unk_strat
            )
            for af in one_hot_atom_feature_list
        ]
    )


def featurize_graph(df, smiles_col="smiles", label_col="y"):
    graphs = []
    for smiles, y in zip(df[smiles_col], df[label_col]):
        mol = Chem.MolFromSmiles(smiles)
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = get_one_hot_atom_features(
                atom,
                [
                    "Symbol",
                    "Degree",
                    "FormalCharge",
                    "Hybridization",
                    "TotalNumHs",
                ],
            )
            atom_features_list.append(atom_feature)

        atom_features = torch.from_numpy(np.stack(atom_features_list)).float()

        edges_list = []
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()

            edges_list.append((begin_atom, end_atom))
            edges_list.append((end_atom, begin_atom))

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        assert edge_index.shape[0] == 2

        data = Data(x=atom_features, edge_index=edge_index, y=y, smiles=smiles)
        graphs.append(data)
    return graphs
