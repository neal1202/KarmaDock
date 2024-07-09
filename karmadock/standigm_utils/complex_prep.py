#!/usr/bin/env python

# -*- coding:utf-8 -*-

# @time : 2024/07/08 15:27
# @autor : Hokyun Jeon
# @contact : neal1202@gmail.com

# Copyright (C) 2024 Standigm, Inc.
# All rights reserved.

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from prody import parsePDB, writePDB
from rdkit import Chem


class ComplexPreprocessor:
    def __init__(
        self,
        pdb_path: str,
        reference_ligand_path: str,
        output_dir: Optional[str] = None,
        distance: int = 12,
        parallel: bool = False,
        cores: int = None,
        verbose: bool = False,
    ):
        """
        Initialize the ComplexPreprocessor with file paths and distance.

        Parameters:
        pdb_path (str): Path to the PDB file.
        reference_ligand_path (str): Path to the reference ligand file (.sdf or .mol2).
        output_dir (str, optional): Directory to save the resulting complex. Defaults to the directory of the reference ligand file.
        distance (int, optional): Distance for selecting the pocket. Defaults to 12.
        parallel (bool, optional): Whether to process in parallel. Defaults to False.
        cores (int, optional): Number of cores to use in parallel processing. Defaults to None, which means using all available cores.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.
        """
        self.pdb_path: Path = Path(pdb_path)
        self.reference_ligand_path: Path = Path(reference_ligand_path)
        self.output_dir: Path = (
            Path(output_dir)
            if output_dir
            else self.reference_ligand_path.parent
            / f"{self.reference_ligand_path.stem}_{self.pdb_path.stem}_complex"
        )
        self.distance: int = distance
        self.parallel: bool = parallel
        self.cores: int = cores if cores is not None else mp.cpu_count()
        self.verbose: bool = verbose

        if self.verbose:
            print(
                f"Initializing ComplexPreprocessor with PDB path: {self.pdb_path}, "
                f"reference ligand path: {self.reference_ligand_path}, output directory: {self.output_dir}, "
                f"distance: {self.distance}, parallel: {self.parallel}, cores: {self.cores}"
            )

        self.validate_paths()

    def validate_paths(self) -> None:
        """
        Validate the existence of PDB and reference ligand files.

        Raises:
        FileNotFoundError: If any of the files do not exist.
        """
        if not self.pdb_path.exists():
            raise FileNotFoundError(
                f"PDB file read error: {self.pdb_path} does not exist"
            )
        if not self.reference_ligand_path.exists():
            raise FileNotFoundError(
                f"Reference ligand file read error: {self.reference_ligand_path} does not exist"
            )
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_ligands(self) -> List[Tuple[Optional[Chem.Mol], Optional[np.ndarray]]]:
        """
        Load the reference ligands from a file.

        Returns:
        List of tuples (Chem.Mol, np.ndarray): List of RDKit molecule objects and their coordinates.
        """
        if self.verbose:
            print(f"Loading ligands from {self.reference_ligand_path}")

        ligands = []
        if self.reference_ligand_path.suffix == ".sdf":
            supplier = Chem.SDMolSupplier(
                str(self.reference_ligand_path), removeHs=False
            )
        elif self.reference_ligand_path.suffix == ".mol2":
            supplier = Chem.Mol2MolSupplier(
                str(self.reference_ligand_path), removeHs=False
            )
        else:
            raise ValueError(
                f"Unsupported file format: {self.reference_ligand_path.suffix}"
            )

        for mol in supplier:
            if mol is not None:
                ligpos = mol.GetConformer().GetPositions()
                ligands.append((mol, ligpos))
            else:
                ligands.append((None, None))
        return ligands

    def extract_pocket(self, ligand_mol, ligpos, index) -> Path:
        """
        Extract the pocket around the reference ligand and save the resulting complex.

        Parameters:
        ligand_mol (Chem.Mol): RDKit molecule object.
        ligpos (np.ndarray): Coordinates of the ligand.
        index (int): Index of the molecule in the file.

        Returns:
        Path: Path to the saved complex PDB file.
        """
        output_path = self.output_dir / f"{self.reference_ligand_path.stem}_{index}.pdb"
        if output_path.exists():
            print(f"{output_path} already exists.")
            return output_path

        if self.verbose:
            print(f"Extracting pocket for ligand at index {index}")

        protein_prody_obj = parsePDB(self.pdb_path.as_posix())
        condition = f"same residue as exwithin {self.distance} of somepoint"
        pocket_selected = protein_prody_obj.select(condition, somepoint=ligpos)
        writePDB(output_path.as_posix(), atoms=pocket_selected)

        return output_path

    def process_ligand(self, ligand_info):
        """
        Process a single ligand to extract the pocket.
        """
        index, ligand_mol, ligpos = ligand_info
        if ligand_mol is None:
            print(f"Error: Molecule at index {index} is None.")
            return
        try:
            output_path = self.extract_pocket(ligand_mol, ligpos, index)
            if self.verbose:
                print(f"Pocket {index} saved to {output_path}")
        except Exception as e:
            smiles = Chem.MolToSmiles(ligand_mol) if ligand_mol else "N/A"
            print(
                f"Error processing molecule at index {index} with SMILES {smiles}: {e}"
            )

    def process(self):
        """
        Process the ligands and extract pockets.
        """
        ligands = self.load_ligands()
        ligand_info_list = [
            (index, mol, pos) for index, (mol, pos) in enumerate(ligands)
        ]

        if self.parallel:
            if self.verbose:
                print(f"Processing ligands in parallel using {self.cores} cores")
            with mp.Pool(processes=self.cores) as pool:
                pool.map(self.process_ligand, ligand_info_list)
        else:
            if self.verbose:
                print("Processing ligands sequentially")
            for ligand_info in ligand_info_list:
                self.process_ligand(ligand_info)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the pocket around a reference ligand in a PDB file and save the resulting complex."
    )
    parser.add_argument(
        "-p", "--pdb_path", required=True, type=str, help="Path to the PDB file."
    )
    parser.add_argument(
        "-r",
        "--reference_ligand_path",
        required=True,
        type=str,
        help="Path to the reference ligand file (.sdf or .mol2).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the resulting complex files. Defaults to the directory of the reference ligand file.",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=12,
        help="Distance for selecting the pocket. Defaults to 12.",
    )
    parser.add_argument(
        "--multiprocessing",
        "-mp",
        action="store_true",
        help="Process molecules in parallel.",
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=None,
        help="Number of cores to use for parallel processing. Defaults to use all available cores.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information during processing.",
    )

    args = parser.parse_args()

    preprocessor = ComplexPreprocessor(
        args.pdb_path,
        args.reference_ligand_path,
        args.output_dir,
        args.distance,
        args.multiprocessing,
        args.cores,
        args.verbose,
    )
    preprocessor.process()


if __name__ == "__main__":
    main()
