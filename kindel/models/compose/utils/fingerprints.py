import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import DataStructs, MACCSkeys, SaltRemover


def get_fps(
    smiles_list,
    fp_type="morgan",
    fp_rad=3,
    fp_dim=2048,
    use_chirality=True,
    remove_salts=True,
    as_numpy=False,
):
    mols = []
    for smiles in smiles_list:
        if remove_salts and "." in smiles:
            remover = SaltRemover.SaltRemover()
            res = remover.StripMol(Chem.MolFromSmiles(smiles))
            mols.append(res)
        else:
            mols.append(Chem.MolFromSmiles(smiles))

    # If molecule object is invalid, add empty molecule
    clean_mols = []
    for mol in mols:
        if mol is None:
            clean_mols.append(Chem.MolFromSmiles(""))
        else:
            clean_mols.append(mol)

    assert len(clean_mols) == len(mols)

    mols = clean_mols
    if fp_type == "morgan":
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(
                x, radius=fp_rad, nBits=fp_dim, useChirality=use_chirality
            )
            for x in mols
        ]
    elif fp_type == "maccs":
        fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]
    else:
        print("Fingerprint type: %s not recognized" % fp_type)
        assert False
    if as_numpy:
        fps = [
            np.unpackbits(
                np.frombuffer(DataStructs.BitVectToBinaryText(x), dtype=np.uint8),
                bitorder="little",
            )
            for x in fps
        ]
    return fps
