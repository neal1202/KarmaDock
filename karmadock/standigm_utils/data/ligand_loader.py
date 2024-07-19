#!usr/bin/env python

# -*- coding:utf-8 -*-

# @time : 2024/07/11 15:08
# @author : Hokyun Jeon
# @contact : neal1202@gmail.com

# Copyright (C) 2024 Standigm, Inc.
# All rights reserved.

from pathlib import Path
from typing import List, Literal

import pandas as pd
from rdkit import Chem


class LigandLoader:

    def __init__(self, ligand_file_path: str):
        self.ligand_file_path = Path(ligand_file_path)
        self._check_file_exists()
        indices, ligands = self.load()
        self.data = [
            {
                "index": idx,
                "smiles": Chem.MolToSmiles(m),
                "inchikey": Chem.MolToInchiKey(m),
                "canonical_smiles": Chem.CanonSmiles(Chem.MolToSmiles(m)),
                "canonical_inchikey": Chem.MolToInchiKey(
                    Chem.MolFromSmiles(Chem.CanonSmiles(Chem.MolToSmiles(m)))
                ),
                "mol": m,
            }
            for idx, m in zip(indices, ligands)
        ]

    def _check_file_exists(self):
        if not self.ligand_file_path.exists():
            raise FileNotFoundError(f"File not found: {self.ligand_file_path}")

    def _load_ligand_from_csv(self) -> List[Chem.Mol]:
        df_ligands = pd.read_csv(self.ligand_file_path)
        smiles_col = next(
            (c for c in df_ligands.columns if c.lower() == "smiles"), None
        )
        if smiles_col is None:
            raise ValueError("No SMILES column found in the ligand file")
        smiles_list = df_ligands[smiles_col].tolist()
        ligands = [
            (idx, Chem.MolFromSmiles(smi))
            for idx, smi in enumerate(smiles_list)
            if Chem.MolFromSmiles(smi) is not None
        ]
        indices, mols = zip(*ligands)
        return indices, mols

    def _load_ligand_using_rdkit(
        self, file_type: Literal["sdf", "smi"]
    ) -> List[Chem.Mol]:
        if file_type == "sdf":
            suppl_func = Chem.ForwardSDMolSupplier
        elif file_type == "smi":
            suppl_func = Chem.SmilesMolSupplier

        with suppl_func(str(self.ligand_file_path)) as suppl:
            indices = []
            mols = []
            i = 0
            for mol in suppl:
                if mol is not None:
                    mols.append(mol)
                    indices.append(i)
                i += 1
            return indices, mols

    def load(self) -> List[Chem.Mol]:
        suffix = self.ligand_file_path.suffix.lower()
        if suffix == ".csv":
            return self._load_ligand_from_csv()
        elif suffix == ".sdf":
            return self._load_ligand_using_rdkit(file_type="sdf")
        elif suffix == ".smi":
            return self._load_ligand_using_rdkit(file_type="smi")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
