#!usr/bin/env python

# -*- coding:utf-8 -*-

# @time : 2024/07/19 11:59
# @author : Hokyun Jeon
# @contact : neal1202@gmail.com

# Copyright (C) 2024 Standigm, Inc.
# All rights reserved.

from pathlib import Path
from typing import Tuple

from joblib import load
from karmadock.dataset.graph_obj import (
    generate_protein_graph,
    get_protein_ligand_graph,
    load_graph,
    merge_pro_lig_graph,
    random_rotation,
    shuffle_center,
)
from torch.utils.data import Dataset

from .data_preprocessing import generate_graph_batch


class GraphDataset(Dataset):

    def __init__(
        self,
        pocket_file_path,
        ligand_file_path,
        pocket_center: Tuple[float, float, float],
        output_dir_path=None,
        ligand_refinement: bool = False,
        ligand_refinement_max_num: int = 10,
        num_cores: int = 1,
    ):
        self.ligand_smis = []

        self.pocket_file_path = Path(pocket_file_path)
        self.ligand_file_path = Path(ligand_file_path)
        self.pocket_center = pocket_center
        if output_dir_path is None:
            self.output_dir_path = (
                self.ligand_file_path.parent / f"{self.ligand_file_path.stem}_graph"
            )
        else:
            self.output_dir_path = Path(output_dir_path)
        if not self.output_dir_path.exists():
            self.output_dir_path.mkdir(parents=True)
        self.ligand_refinement = ligand_refinement
        self.ligand_refinement_max_num = ligand_refinement_max_num

        self.protein_graph = generate_protein_graph(
            pocket_pdb=self.pocket_file_path.as_posix()
        )
        self.protein_graph.pocket_center = pocket_center
        graph_files = list(self.output_dir_path.glob("*.dgl"))
        if self.output_dir_path.exists() and len(graph_files) > 0:
            print("Found existing graph files. Loading...")
            self.ligand_data = [
                self.load_graph(graph_file_path) for graph_file_path in graph_files
            ]
        else:
            self.ligand_data = generate_graph_batch(
                ligand_file_path=self.ligand_file_path.as_posix(),
                output_dir_path=self.output_dir_path,
                pocket_center=self.pocket_center,
                ligand_refinement=self.ligand_refinement,
                ligand_refinement_max_num=self.ligand_refinement_max_num,
                num_cores=num_cores,
            )

    def load_graph(self, graph_path):
        graph = load(graph_path)
        graph["graph_file_path"] = graph_path
        graph["ligand_name"] = Path(graph_path).stem
        return graph

    def merge_complex_graph(self, idx):
        data = self.ligand_data[idx]
        ligand_graph_path = data["graph_file_path"]
        ligand_data = load_graph(ligand_graph_path)
        complex_data = merge_pro_lig_graph(
            pro_data=self.protein_graph.clone(), data=ligand_data
        )
        complex_graph = get_protein_ligand_graph(
            complex_data,
            pro_node_num=complex_data["protein"].xyz.size(0),
            lig_node_num=complex_data["ligand"].xyz.size(0),
        )
        return complex_graph

    def __getitem__(self, idx):
        try:
            data = self.merge_complex_graph(idx)
            data["ligand"].pos -= data["ligand"].pos.mean(dim=0) - self.pocket_center
            data["ligand"].pos = random_rotation(shuffle_center(data["ligand"].pos))
        except:
            return None
        return data

    def __len__(self):
        return len(self.ligand_data)
