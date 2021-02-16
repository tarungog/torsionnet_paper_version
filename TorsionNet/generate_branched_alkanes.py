import time
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
import json
import tqdm
import numpy as np

import random
from concurrent.futures import ProcessPoolExecutor

from main.utils import *

def create_branched(i):
    m = Chem.MolFromSmiles('CCCC')
    e = Chem.RWMol(m)

    numatoms = len(e.GetAtoms())
    tot = np.random.choice(list(range(7,20)))
    while numatoms < tot:
        x = Chem.rdchem.Atom(6)
        randidx = np.random.randint(len(e.GetAtoms()))
        atom = e.GetAtomWithIdx(randidx)
        if atom.GetDegree() > 2:
            continue
        if atom.GetDegree() == 2 and random.random() <= 0.5:
            continue
        idx = e.AddAtom(x)
        e.AddBond(idx, randidx, Chem.rdchem.BondType.SINGLE)
        numatoms = len(e.GetAtoms())


    Chem.SanitizeMol(e)
    m = Chem.rdmolops.AddHs(e.GetMol())
    AllChem.EmbedMultipleConfs(m, numConfs=200, numThreads=-1)
    Chem.AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=-1)


    confgen = ConformerGeneratorCustom(max_conformers=1,
                        rmsd_threshold=None,
                        force_field='mmff',
                        pool_multiplier=1)

    m = prune_conformers(m, 0.05)

    energys = confgen.get_conformer_energies(m)
    print(len(TorsionFingerprints.CalculateTorsionLists(m)[0]))
    standard = energys.min()
    total = np.sum(np.exp(-(energys-standard)))

    nonring, ring = Chem.TorsionFingerprints.CalculateTorsionLists(m)
    rbn = len(nonring)
    out = {
        'mol': Chem.MolToSmiles(m, isomericSmiles=False),
        'standard': standard,
        'total': total
    }

    with open(f'gen_out/{rbn}_{i}.json', 'w') as fp:
        json.dump(out, fp)


with ProcessPoolExecutor() as executor:
    executor.map(create_branched, range(10000))
