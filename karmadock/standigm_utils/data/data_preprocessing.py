#!usr/bin/env python

# -*- coding:utf-8 -*-

# @time : 2024/07/19 11:45
# @author : Hokyun Jeon
# @contact : neal1202@gmail.com

# Copyright (C) 2024 Standigm, Inc.
# All rights reserved.


from pathlib import Path

import parmap
import torch
from karmadock.dataset.graph_obj import generate_lig_graph, save_graph
from torch_geometric.data import HeteroData
from tqdm import tqdm

from .ligand_loader import LigandLoader
from .utils import refine_mol_with_rdkit_mmff


def generate_graph(
    ligand_data,
    pocket_center,
    output_dir_path,
    ligand_refinement=False,
    ligand_refinement_max_num=10,
):

    torch.set_num_threads(1)
    ligand_name = f"{ligand_data['index']}_{ligand_data['inchikey']}"
    output_file_path = output_dir_path / f"{ligand_name}.dgl"

    mol = ligand_data["mol"]

    if ligand_refinement:
        mol = refine_mol_with_rdkit_mmff(mol, ligand_refinement_max_num)

    try:
        data = HeteroData()
        ligand_graph = generate_lig_graph(data, mol)
        ligand_graph.pdb_id = ligand_name
        ligand_graph["ligand"].mol = mol
        ligand_graph["ligand"].pos = (
            ligand_graph["ligand"].xyz
            + pocket_center
            - ligand_graph["ligand"].xyz.mean(dim=0)
        ).to(torch.float32)
        save_graph(output_file_path.as_posix(), ligand_graph)
        ligand_data["graph_file_path"] = output_file_path.as_posix()
        ligand_data["ligand_name"] = ligand_name
        return ligand_data
    except Exception as e:
        print(f"{ligand_name} failed to get graph. Error: {e}")
        return None


def generate_graph_batch(
    ligand_file_path,
    output_dir_path,
    pocket_center,
    ligand_refinement=False,
    ligand_refinement_max_num=10,
    verbose=True,
    num_cores=1,
):

    output_dir_path = Path(output_dir_path)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    ligand_loader = LigandLoader(ligand_file_path)
    ligand_data = ligand_loader.data

    args = [
        (
            ligand,
            pocket_center,
            output_dir_path,
            ligand_refinement,
            ligand_refinement_max_num,
        )
        for ligand in ligand_data
    ]

    if verbose:
        results = list(
            tqdm(
                parmap.starmap(generate_graph, args, processes=num_cores),
                total=len(ligand_data),
            )
        )
    else:
        results = parmap.starmap(generate_graph, args, processes=num_cores)

    ligand_data = [r for r in results if r is not None]
    return ligand_data
