"""
    This code is directly from:
        https://github.com/MolecularAI/reinvent-chemistry

    The code was slightly modified to fit in our project.
    Download date: 2023/06/28
"""

import random
from typing import List, Tuple

from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles, MolStandardize, MolToInchiKey
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import RenumberAtoms
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from rdkit.Chem import SDWriter
###
from rdkit.Chem.Scaffolds import MurckoScaffold


class Conversions:
    ###
    @staticmethod
    def get_scaffold(smile: str):
        mol = MolFromSmiles(smile)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return MolToSmiles(scaffold, isomericSmiles=False)
        else:
            return None

    @staticmethod
    def smiles_to_mols_and_indices(query_smiles: List[str]) -> Tuple[List[Mol], List[int]]:
        mols = [MolFromSmiles(smile) for smile in query_smiles]
        valid_mask = [mol is not None for mol in mols]
        valid_idxs = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols, valid_idxs

    @staticmethod
    def mols_to_fingerprints(molecules: List[Mol], radius=3, use_counts=True, use_features=True) \
            -> List[UIntSparseIntVect]:
        fingerprints = [AllChem.GetMorganFingerprint(mol, radius, useCounts=use_counts, useFeatures=use_features) for
                        mol in molecules]
        return fingerprints

    @staticmethod
    def smiles_to_mols(query_smiles: List[str]) -> List[Mol]:
        mols = [MolFromSmiles(smile) for smile in query_smiles]
        valid_mask = [mol is not None for mol in mols]
        valid_idxs = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols

    def smiles_to_fingerprints(self, query_smiles: List[str], radius=3, use_counts=True, use_features=True) -> List[
        UIntSparseIntVect]:
        mols = self.smiles_to_mols(query_smiles)
        fingerprints = self.mols_to_fingerprints(mols, radius=radius, use_counts=use_counts, use_features=use_features)
        return fingerprints

    def smile_to_mol(self, smile: str) -> Mol:
        """
        Creates a Mol object from a SMILES string.
        :param smile: SMILES string.
        :return: A Mol object or None if it's not valid.
        """
        if smile:
            return MolFromSmiles(smile)

    def mols_to_smiles(self, molecules: List[Mol], isomericSmiles=False, canonical=True) -> List[str]:
        """This method assumes that all molecules are valid."""
        valid_smiles = [MolToSmiles(mol, isomericSmiles=isomericSmiles, canonical=canonical) for mol in molecules]
        return valid_smiles

    def mol_to_smiles(self, molecule: Mol, isomericSmiles=False, canonical=True) -> str:
        """
        Converts a Mol object into a canonical SMILES string.
        :param molecule: Mol object.
        :return: A SMILES string.
        """
        if molecule:
            return MolToSmiles(molecule, isomericSmiles=isomericSmiles, canonical=canonical)

    def mol_to_random_smiles(self, molecule: Mol) -> str:
        """
        Converts a Mol object into a random SMILES string.
        :param molecule: Mol object
        :return: A SMILES string.
        """
        if molecule:
            new_atom_order = list(range(molecule.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = RenumberAtoms(molecule, newOrder=new_atom_order)
            return MolToSmiles(random_mol, canonical=False, isomericSmiles=False)

    def convert_to_rdkit_smiles(self, smiles: str, allowTautomers=True, sanitize=False, isomericSmiles=False) -> str:
        """
        :param smiles: Converts a smiles string into a canonical SMILES string.
        :type allowTautomers: allows having same molecule represented in different tautomeric forms
        """
        if allowTautomers:
            return MolToSmiles(MolFromSmiles(smiles, sanitize=sanitize), isomericSmiles=isomericSmiles)
        else:
            return MolStandardize.canonicalize_tautomer_smiles(smiles)

    def copy_mol(self, molecule: Mol) -> Mol:
        """
        Copies, sanitizes, canonicalizes and cleans a molecule.
        :param molecule: A Mol object to copy.
        :return : Another Mol object copied, sanitized, canonicalized and cleaned.
        """
        return self.smile_to_mol(self.mol_to_smiles(molecule))

    def randomize_smiles(self, smiles: str) -> str:
        """
        Returns a random SMILES given a SMILES of a molecule.
        :param smiles: A smiles string
        :param random_type: The type (unrestricted, restricted) of randomization performed.
        :return : A random SMILES string of the same molecule or None if the molecule is invalid.
        """
        mol = MolFromSmiles(smiles)
        if mol:
            new_atom_order = list(range(mol.GetNumHeavyAtoms()))
            random.shuffle(new_atom_order)
            random_mol = RenumberAtoms(mol, newOrder=new_atom_order)
            return MolToSmiles(random_mol, canonical=False, isomericSmiles=False)

    def mol_to_inchi_key(self, molecule: Mol) -> str:
        """ Returns the standard InChI key for a molecule """
        if molecule:
            inchi_key = MolToInchiKey(molecule)
            return inchi_key

    def mol_to_sdf(self, molecules: List, input_sdf_path: str):
        """ Write a set of molecules to sdf file"""
        writer = SDWriter(input_sdf_path)
        for mol in molecules:
            writer.write(mol)