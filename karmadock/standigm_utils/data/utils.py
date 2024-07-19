#!usr/bin/env python

# -*- coding:utf-8 -*-

# @time : 2024/07/11 16:59
# @author : Hokyun Jeon
# @contact : neal1202@gmail.com

# Copyright (C) 2024 Standigm, Inc.
# All rights reserved.

from pathlib import Path

# Preprocess
from rdkit import Chem


def collect_ligand_coordinates(ligand_file):
    ligand_file = Path(ligand_file)
    if not ligand_file.exists():
        raise FileNotFoundError(f"{ligand_file} not found")
    if ligand_file.suffix == ".sdf":
        ligand_mol = Chem.MolFromMolFile(ligand_file.as_posix(), removeHs=False)
    elif ligand_file.suffix == ".mol2":
        ligand_mol = Chem.MolFromMol2File(ligand_file.as_posix(), removeHs=False)
    else:
        raise ValueError(f"Unsupported ligand file format: {ligand_file}")
    if ligand_mol is not None:
        return ligand_mol.GetConformer().GetPositions()
    else:
        raise ValueError(f"Failed to read {ligand_file}")


import torch


def get_ligand_center(ligand_file):
    coordinates = collect_ligand_coordinates(ligand_file)
    return torch.from_numpy(coordinates).to(torch.float32).mean(dim=0)


from prody import parsePDB, writePDB


def get_binding_pocket_pdb(
    pdb_file, ligand_file, output_path=None, binding_site_size=12.0
):
    if not Path(pdb_file).exists():
        raise FileNotFoundError(f"{pdb_file} not found")
    if output_path is None:
        ligand_name = Path(ligand_file).stem
        pdb_name = Path(pdb_file).stem
        output_name = "binding_pocket" + f"_{pdb_name}" + f"_{ligand_name}" + ".pdb"
        output_path = Path(ligand_file).with_name(output_name)
    if output_path.exists():
        print("Output file already exists. Skipping.")
        return output_path.resolve().as_posix()
    coordinates = collect_ligand_coordinates(ligand_file)
    pdb_parsed = parsePDB(pdb_file)
    condition = f"same residue as exwithin {binding_site_size} of somepoint"
    binding_pocket = pdb_parsed.select(condition, somepoint=coordinates)
    writePDB(output_path.as_posix(), atoms=binding_pocket)
    return output_path.resolve().as_posix()


import MDAnalysis as mda
import torch
from karmadock.dataset.protein_feature import get_protein_feature_mda
from torch_geometric.data import HeteroData


def pocket_to_graph(pocket_pdb_file):
    pocket_pdb_file = Path(pocket_pdb_file)
    if not pocket_pdb_file.exists():
        raise FileNotFoundError(f"{pocket_pdb_file} not found")

    pocket_universe_obj = mda.Universe(pocket_pdb_file.as_posix())

    # get features
    (
        p_xyz,
        p_xyz_full,
        p_seq,
        p_node_s,
        p_node_v,
        p_edge_index,
        p_edge_s,
        p_edge_v,
        p_full_edge_s,
    ) = get_protein_feature_mda(pocket_universe_obj)

    # create HeteroData object and populate it with protein features
    data = HeteroData()
    data["protein"].node_s = p_node_s.to(torch.float32)
    data["protein"].node_v = p_node_v.to(torch.float32)
    data["protein"].xyz = p_xyz.to(torch.float32)
    data["protein"].xyz_full = p_xyz_full.to(torch.float32)
    data["protein"].seq = p_seq.to(torch.int32)
    data["protein", "p2p", "protein"].edge_index = p_edge_index.to(torch.long)
    data["protein", "p2p", "protein"].edge_s = p_edge_s.to(torch.float32)
    data["protein", "p2p", "protein"].full_edge_s = p_full_edge_s.to(torch.float32)
    data["protein", "p2p", "protein"].edge_v = p_edge_v.to(torch.float32)

    return data


def ligand_to_graph(ligand_file):
    ligand_file = Path(ligand_file)
    if not ligand_file.exists():
        raise FileNotFoundError(f"{ligand_file} not found")


import copy

from rdkit import Chem
from rdkit.Chem import AllChem


def refine_mol_with_rdkit_mmff(mol, refine_max_num=1):
    mol = copy.deepcopy(mol)
    mol = Chem.AddHs(mol)
    feed_back = AllChem.EmbedMolecule(mol)
    if feed_back == -1:
        return mol
    feed_back = [[-1, 1]]
    # Refine the molecule with MMFF94s force field
    n = 0
    while feed_back[0][0] == -1 and n < refine_max_num:
        feed_back = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s")
        n += 1
    mol = Chem.RemoveAllHs(mol)
    return mol
