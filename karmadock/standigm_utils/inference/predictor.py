#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@time : 2024/07/19 14:43
@author : Hokyun Jeon
@contact : neal1202@gmail.com

Copyright (C) 2024 Standigm, Inc.
All rights reserved.
"""

from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from karmadock import trained_models
from karmadock.dataset.dataloader_obj import PassNoneDataLoader
from tqdm import tqdm

from .postprocessing import (
    correct_and_score_positions,  # Adjust import according to your package structure
)
from .utils import (  # Adjust import according to your package structure
    get_device,
    load_trained_model,
    save_inference_results,
)

DEFAULT_MODEL_CKPT = f"{trained_models.__path__[0]}/karmadock_screening.pkl"


class KarmaDockPredictor:
    def __init__(
        self,
        test_dataset,
        output_directory,
        save_pose=True,
        model_checkpoint_path=DEFAULT_MODEL_CKPT,
        batch_size=64,
        score_threshold=0,
        num_workers=0,
        use_gpu=True,
    ):
        self.model_checkpoint_path = model_checkpoint_path
        self.output_directory = output_directory
        if not Path(self.output_directory).exists():
            Path(self.output_directory).mkdir(parents=True)
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.num_workers = num_workers
        self.save_pose = save_pose

        self.test_dataloader = PassNoneDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            follow_batch=[],
            pin_memory=True,
        )
        self.use_gpu = use_gpu
        self.device, self.device_id = get_device(use_gpu)
        self.model = load_trained_model(
            self.model_checkpoint_path, self.device_id, self.device
        )

    def run_inference(self):
        self.model.eval()
        predicted_scores = torch.as_tensor([]).to(self.device)
        forcefield_corrected_scores = torch.as_tensor([]).to(self.device)
        aligned_corrected_scores = torch.as_tensor([]).to(self.device)
        pdb_ids = []

        with TemporaryDirectory() as temp_dir:
            with torch.no_grad():
                for batch in tqdm(self.test_dataloader, desc="prediction"):
                    batch = batch.to(self.device)
                    batch_size = batch["ligand"].batch[-1] + 1

                    protein_node_states, ligand_node_states = (
                        self.model.module.encoding(batch)
                    )
                    ligand_positions, _, _ = self.model.module.docking(
                        protein_node_states, ligand_node_states, batch, recycle_num=3
                    )

                    model_scores = self.model.module.scoring(
                        lig_s=ligand_node_states,
                        lig_pos=ligand_positions,
                        pro_s=protein_node_states,
                        data=batch,
                        dist_threhold=5.0,
                        batch_size=batch_size,
                    )
                    predicted_scores = torch.cat(
                        [predicted_scores, model_scores], dim=0
                    )
                    pdb_ids.extend(batch.pdb_id)
                    (
                        forcefield_corrected_scores_batch,
                        aligned_corrected_scores_batch,
                    ) = correct_and_score_positions(
                        batch,
                        self.model,
                        ligand_node_states,
                        protein_node_states,
                        ligand_positions,
                        model_scores,
                        self.score_threshold,
                        self.device,
                        temp_dir,
                    )
                    forcefield_corrected_scores = torch.cat(
                        [
                            forcefield_corrected_scores,
                            forcefield_corrected_scores_batch,
                        ],
                        dim=0,
                    )
                    aligned_corrected_scores = torch.cat(
                        [aligned_corrected_scores, aligned_corrected_scores_batch],
                        dim=0,
                    )
            if self.save_pose:
                self.save_poses(temp_dir)

        self.pdb_ids = pdb_ids
        self.predicted_scores = predicted_scores
        self.forcefield_corrected_scores = forcefield_corrected_scores
        self.aligned_corrected_scores = aligned_corrected_scores
        self.save_results()

    def save_results(
        self,
    ):
        if not hasattr(self, "predicted_scores"):
            raise ValueError(
                "You need to run inference first before saving the results."
            )
        save_inference_results(
            self.pdb_ids,
            self.predicted_scores,
            self.forcefield_corrected_scores,
            self.aligned_corrected_scores,
            self.output_directory,
        )

    def save_poses(
        self,
        temp_dir=None,
    ):
        align_files = glob(str(Path(temp_dir) / "*_*_pred_align_corrected.sdf"))
        merge_sdf_files(align_files, Path(self.output_directory) / "aligned.sdf")

        ff_files = glob(str(Path(temp_dir) / "*_*_pred_ff_corrected.sdf"))
        merge_sdf_files(ff_files, Path(self.output_directory) / "ff_corrected.sdf")

        uncorrected_files = glob(str(Path(temp_dir) / "*_*_pred_uncorrected.sdf"))
        merge_sdf_files(
            uncorrected_files, Path(self.output_directory) / "uncorrected.sdf"
        )

        random_files = glob(str(Path(temp_dir) / "*_*_random_pose.sdf"))
        merge_sdf_files(random_files, Path(self.output_directory) / "random.sdf")


def merge_sdf_files(sdf_files, output_file):
    with open(output_file, "w") as out:
        for sdf_file in sdf_files:
            with open(sdf_file, "r") as infile:
                content = infile.read().strip()
            idx, inchikey = str(Path(sdf_file).stem).split("_")[:2]
            out.write(content)
            out.write(f"\n> <index>\n{idx}\n")
            out.write(f"\n> <inchikey>\n{inchikey}\n")
            if not content.endswith("$$$$"):
                out.write("\n$$$$\n")
            else:
                out.write("\n")
    print(f"Merged SDF files into {output_file}")
