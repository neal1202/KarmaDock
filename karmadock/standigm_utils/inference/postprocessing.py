#!usr/bin/env python

# -*- coding:utf-8 -*-

"""
@time : 2024/07/19 14:51
@author : Hokyun Jeon
@contact : neal1202@gmail.com

Copyright (C) 2024 Standigm, Inc.
All rights reserved.
"""


import numpy as np
import torch
from karmadock.utils.post_processing import correct_pos


def correct_and_score_positions(
    batch,
    model,
    ligand_node_states,
    protein_node_states,
    ligand_positions,
    model_scores,
    score_threshold,
    device,
    output_directory,
):
    batch.pos_preds = ligand_positions
    corrected_poses, _, _ = correct_pos(
        batch,
        mask=model_scores <= score_threshold,
        out_dir=output_directory,
        out_init=True,
        out_uncoorected=True,
        out_corrected=True,
    )

    forcefield_corrected_positions = torch.from_numpy(
        np.concatenate([pose[0] for pose in corrected_poses], axis=0)
    ).to(device)
    aligned_corrected_positions = torch.from_numpy(
        np.concatenate([pose[1] for pose in corrected_poses], axis=0)
    ).to(device)

    forcefield_corrected_scores = model.module.scoring(
        lig_s=ligand_node_states,
        lig_pos=forcefield_corrected_positions,
        pro_s=protein_node_states,
        data=batch,
        dist_threhold=5.0,
        batch_size=batch["ligand"].batch[-1] + 1,
    )
    aligned_corrected_scores = model.module.scoring(
        lig_s=ligand_node_states,
        lig_pos=aligned_corrected_positions,
        pro_s=protein_node_states,
        data=batch,
        dist_threhold=5.0,
        batch_size=batch["ligand"].batch[-1] + 1,
    )

    return forcefield_corrected_scores, aligned_corrected_scores
