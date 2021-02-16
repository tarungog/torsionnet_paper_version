import numpy as np
import bisect
import torch
import logging
from deepchem.utils import conformers
from tqdm import tqdm
import mdtraj as md
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, TorsionFingerprints

class ConformerGeneratorCustom(conformers.ConformerGenerator):
    # pruneRmsThresh=-1 means no pruning done here
    # I don't use embed_molecule() because it does AddHs() & EmbedMultipleConfs()
    def __init__(self, *args, **kwargs):
        super(ConformerGeneratorCustom, self).__init__(*args, **kwargs)


    # add progress bar
    def minimize_conformers(self, mol):
        """
        Minimize molecule conformers.

        Parameters
        ----------
        mol : RDKit Mol
                Molecule.
        """
        pbar = tqdm(total=mol.GetNumConformers())
        for conf in mol.GetConformers():
            ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
            ff.Minimize()
            pbar.update(1)
        pbar.close()

    def prune_conformers(self, mol, rmsd, heavy_atoms_only=True):
        """
        Prune conformers from a molecule using an RMSD threshold, starting
        with the lowest energy conformer.

        Parameters
        ----------
        mol : RDKit Mol
                Molecule.

        Returns
        -------
        new: A new RDKit Mol containing the chosen conformers, sorted by
                 increasing energy.
        new_rmsd: matrix of conformer-conformer RMSD
        """
        if self.rmsd_threshold < 0 or mol.GetNumConformers() <= 1:
            return mol
        energies = self.get_conformer_energies(mol)
    #     rmsd = get_conformer_rmsd_fast(mol)

        sort = np.argsort(energies)  # sort by increasing energy
        keep = []  # always keep lowest-energy conformer
        discard = []

        for i in sort:
            # always keep lowest-energy conformer
            if len(keep) == 0:
                keep.append(i)
                continue

            # discard conformers after max_conformers is reached
            if len(keep) >= self.max_conformers:
                discard.append(i)
                continue

            # get RMSD to selected conformers
            this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

            # discard conformers within the RMSD threshold
            if np.all(this_rmsd >= self.rmsd_threshold):
                keep.append(i)
            else:
                discard.append(i)

        # create a new molecule to hold the chosen conformers
        # this ensures proper conformer IDs and energy-based ordering
        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        for i in keep:
            conf = mol.GetConformer(conf_ids[i])
            new.AddConformer(conf, assignId=True)

        new_rmsd = get_conformer_rmsd_fast(new, heavy_atoms_only=heavy_atoms_only)
        return new, new_rmsd

def prune_last_conformer(mol, tfd_thresh, energies=None, quick=False):
    """
    Checks that most recently added conformer meats TFD threshold.

    Parameters
    ----------
    mol : RDKit Mol
            Molecule.
    tfd_thresh : TFD threshold
    energies: energies of all conformers minus the last one
    Returns
    -------
    new: A new RDKit Mol containing the chosen conformers, sorted by
             increasing energy.
    """

    confgen = ConformerGeneratorCustom()

    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    idx = bisect.bisect(energies[:-1], energies[-1])

    tfd = Chem.TorsionFingerprints.GetTFDBetweenConformers(mol, range(0, mol.GetNumConformers() - 1), [mol.GetNumConformers() - 1], useWeights=False)
    tfd = np.array(tfd)

    # if lower energy conformer is within threshold, drop new conf
    if not np.all(tfd[:idx] >= tfd_thresh):
        new_energys = list(range(0, mol.GetNumConformers() - 1))
        mol.RemoveConformer(mol.GetNumConformers() - 1)

        logging.debug('tossing conformer')

        return mol, new_energys


    else:
        logging.debug('keeping conformer', idx)
        keep = list(range(0,idx))
        # print('keep 1', keep)
        keep += [mol.GetNumConformers() - 1]
        # print('keep 2', keep)

        l = np.array(range(idx, len(tfd)))
        # print('L 1', l)
        # print('tfd', tfd)
        l = l[tfd[idx:] >= tfd_thresh]
        # print('L 2', l)

        keep += list(l)
        # print('keep 3', keep)

        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]

        for i in keep:
            conf = mol.GetConformer(conf_ids[i])
            new.AddConformer(conf, assignId=True)

        return new, keep

def prune_last_conformer_quick(mol, tfd_thresh, energies=None):
    """
    Checks that most recently added conformer meats TFD threshold.

    Parameters
    ----------
    mol : RDKit Mol
            Molecule.
    tfd_thresh : TFD threshold
    energies: energies of all conformers minus the last one
    Returns
    -------
    new: A new RDKit Mol containing the chosen conformers, sorted by
             increasing energy.
    """

    confgen = ConformerGeneratorCustom()

    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    tfd = Chem.TorsionFingerprints.GetTFDBetweenConformers(mol, range(0, mol.GetNumConformers() - 1), [mol.GetNumConformers() - 1], useWeights=False)
    tfd = np.array(tfd)

    if not np.all(tfd >= tfd_thresh):
        logging.debug('tossing conformer')
        mol.RemoveConformer(mol.GetNumConformers() - 1)
        return mol, 0.0
    else:
        logging.debug('keeping conformer')
        return mol, 1.0



def prune_conformers(mol, tfd_thresh, rmsd=False):
    """
    Prune conformers from a molecule using an TFD/RMSD threshold, starting
    with the lowest energy conformer.

    Parameters
    ----------
    mol : RDKit Mol
            Molecule.
    tfd_thresh : TFD threshold
    Returns
    -------
    new: A new RDKit Mol containing the chosen conformers, sorted by
             increasing energy.
    """

    confgen = ConformerGeneratorCustom()

    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    energies = confgen.get_conformer_energies(mol)

    if not rmsd:
        tfd = array_to_lower_triangle(Chem.TorsionFingerprints.GetTFDMatrix(mol, useWeights=False), True)
    else:
        tfd = get_conformer_rmsd_fast(mol)
    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    discard = []

    for i in sort:
        # always keep lowest-energy conformer
        if len(keep) == 0:
            keep.append(i)
            continue

        # get RMSD to selected conformers
        this_tfd = tfd[i][np.asarray(keep, dtype=int)]
        # discard conformers within the RMSD threshold
        if np.all(this_tfd >= tfd_thresh):
            keep.append(i)
        else:
            discard.append(i)

    # create a new molecule to hold the chosen conformers
    # this ensures proper conformer IDs and energy-based ordering
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    for i in keep:
        conf = mol.GetConformer(conf_ids[i])
        new.AddConformer(conf, assignId=True)

    return new

def print_torsions(mol):
    nonring, ring = TorsionFingerprints.CalculateTorsionLists(mol)
    conf = mol.GetConformer(id=0)
    tups = [atoms[0] for atoms, ang in nonring]
    degs = [Chem.rdMolTransforms.GetDihedralDeg(conf, *tup) for tup in tups]
    print(degs)

def print_energy(mol):
    confgen = ConformerGeneratorCustom(max_conformers=1,
                                 rmsd_threshold=None,
                                 force_field='mmff',
                                 pool_multiplier=1)
    print(confgen.get_conformer_energies(mol))

def array_to_lower_triangle(arr, get_symm=False):
    # convert list to lower triangle mat
    n = int(np.sqrt(len(arr)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    lt_mat = np.zeros((n,n))
    lt_mat[idx] = arr
    if get_symm == True:
        return lt_mat + np.transpose(lt_mat) # symmetric matrix
    return lt_mat
