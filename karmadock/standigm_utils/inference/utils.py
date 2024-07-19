#!usr/bin/env python

# -*- coding:utf-8 -*-

"""
@time : 2024/07/19 14:51
@author : Hokyun Jeon
@contact : neal1202@gmail.com

Copyright (C) 2024 Standigm, Inc.
All rights reserved.
"""

import subprocess

import torch


def get_free_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
    )
    free_memory = [int(x) for x in result.stdout.decode("utf-8").strip().split("\n")]
    best_gpu = free_memory.index(max(free_memory))
    return best_gpu


def get_device(use_gpu=True):
    selected_device = get_free_gpu()
    if use_gpu:
        if torch.cuda.is_available():
            return f"cuda:{selected_device}", selected_device  # Use the best GPU
        else:
            print("GPU is not available. Using CPU instead.")
            return "cpu", None
    else:
        return "cpu", None


import torch
import torch.nn as nn
from karmadock.architecture.KarmaDock_architecture import KarmaDock
from karmadock.utils.fns import Early_stopper


def load_trained_model(checkpoint_path, device_id, device):
    model = KarmaDock()
    if device_id is not None:
        model = nn.DataParallel(model, device_ids=[device_id], output_device=device_id)
    model.to(device)
    stopper = Early_stopper(model_file=checkpoint_path, mode="lower", patience=70)
    stopper.load_model(model_obj=model, my_device=device, strict=False)
    return model


import os

import pandas as pd


def save_inference_results(
    pdb_ids,
    predicted_scores,
    forcefield_corrected_scores,
    aligned_corrected_scores,
    output_directory,
):
    ligand_ids = list(zip(*[ligand_id.split("_")[:2] for ligand_id in pdb_ids]))
    data = zip(
        ligand_ids[0],
        ligand_ids[1],
        predicted_scores.view(-1).cpu().numpy().tolist(),
        forcefield_corrected_scores.view(-1).cpu().numpy().tolist(),
        aligned_corrected_scores.view(-1).cpu().numpy().tolist(),
    )
    columns = [
        "index",
        "inchikey",
        "karma_score",
        "karma_score_ff",
        "karma_score_aligned",
    ]
    df = pd.DataFrame(data, columns=columns)

    output_csv_path = os.path.join(output_directory, "karmadock_score.csv")
    df.to_csv(output_csv_path, index=False)
