import numpy as np

# Chemical Drawing
from rdkit.Chem import MolFromMolBlock, MolToMolBlock, MolToMolFile, MolToSmiles, AddHs
from rdkit import Chem
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import MolToImage

import json
from utils import *
from tqdm import tqdm

# Lignin-KMC functions and global variables used in this notebook
from ligninkmc.kmc_functions import (run_kmc, generate_mol)
from ligninkmc.create_lignin import (calc_rates, create_initial_monomers, create_initial_events,
                                     create_initial_state, analyze_adj_matrix, adj_analysis_to_stdout)
from ligninkmc.kmc_common import (DEF_E_BARRIER_KCAL_MOL, ADJ_MATRIX, MONO_LIST, MONOMER, OX, GROW, Monomer, Event)


# change the 3 vars below
save_dir = "lignins_out"
num_monos = [1,2,3,4,5,6,7,8,9,10]
lignin_oligomers = []

# Calculate the rates of reaction in 1/s (or 1/monomer-s if biomolecular) at the specified temp
temp = 298.15  # K
rxn_rates = calc_rates(temp, ea_kcal_mol_dict=DEF_E_BARRIER_KCAL_MOL)


confgen = ConformerGeneratorCustom(max_conformers=1, 
                 rmsd_threshold=None, 
                 force_field='mmff',
                 pool_multiplier=1)  

if __name__ == '__main__':

    for n in tqdm(num_monos):
        # Set the percentage of S
        sg_ratio = 0
        pct_s = sg_ratio / (1 + sg_ratio)

        # Set the initial and maximum number of monomers to be modeled.
        ini_num_monos = 1
        max_num_monos = n

        # Maximum time to simulate, in seconds
        t_max = 1  # seconds
        mono_add_rate = 1e4  # monomers/second

        # Use a random number and the given sg_ratio to determine the monolignol types to be initially modeled
        monomer_draw = np.random.rand(ini_num_monos)
        initial_monomers = create_initial_monomers(pct_s, monomer_draw)

        # Initially allow only oxidation events. After they are used to determine the initial state, add 
        #     GROW to the events, which allows additional monomers to be added to the reaction at the 
        #     specified rate and with the specified ratio
        initial_events = create_initial_events(initial_monomers, rxn_rates)
        initial_state = create_initial_state(initial_events, initial_monomers)
        initial_events.append(Event(GROW, [], rate=mono_add_rate))

        # simulate lignin creation
        result = run_kmc(rxn_rates, initial_state,initial_events, n_max=max_num_monos, t_max=t_max, sg_ratio=sg_ratio)
        # using RDKit
        nodes = result[MONO_LIST]
        adj = result[ADJ_MATRIX]
        block = generate_mol(adj, nodes)
        mol = MolFromMolBlock(block)
    #     # cap with hydrogens
    #     mol = AddHs(mol)
        lignin_oligomers.append(mol)
        # save
        fn = save_dir + f"/{sg_ratio}sgr_{max_num_monos}monos.mol"
        Chem.AllChem.EmbedMultipleConfs(mol, numConfs=200, numThreads=-1)
        Chem.AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=-1)
        mol = prune_conformers(mol, 0.05)
        energys = confgen.get_conformer_energies(mol)
        standard = energys.min()
        total = np.sum(np.exp(-(energys-standard)))
        
        out = {
            'mol': Chem.MolToSmiles(mol, isomericSmiles=False),
            'standard': standard,
            'total': total
        }
        print(out)
        with open(fn, 'w') as fp:
            json.dump(out, fp)