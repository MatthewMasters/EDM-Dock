import os
import pathlib
import contextlib
from subprocess import call
from collections import defaultdict

import numpy as np
from openforcefield.topology import Molecule
from simtk import unit, openmm
from simtk.openmm import app, Platform, LangevinIntegrator
from simtk.openmm.app import PDBFile, Simulation, Modeller
from openmmforcefields.generators import SystemGenerator


TLEAP = '/home/masters/.conda/envs/masters/bin/tleap'
PDB4AMBER = '/home/masters/.conda/envs/masters/bin/pdb4amber'
DGSOL = '/data/masters/projects/EDM-Dock/DGSOL/build/src/dgsol_s/dgsol_s'
K_UNIT = unit.kilojoule_per_mole / unit.angstrom ** 2


def to_scientific_notation(number):
    a, b = '{:.17E}'.format(number).split('E')
    num = '{:.12f}E{:+03d}'.format(float(a) / 10, int(b) + 1)
    return num[1:]


def write_input(filepath, mu, var, ligand_bm, pocket_dm, k=1.0):
    ligand_n = len(ligand_bm)
    pocket_n = len(pocket_dm)

    outfile = open(filepath, 'w')
    for i in range(ligand_n):
        for j in range(i + 1, ligand_n):
            lb = to_scientific_notation(ligand_bm[j, i])
            ub = to_scientific_notation(ligand_bm[i, j])
            outfile.write(f'{i + 1:9.0f}{j + 1:10.0f}   {lb}   {ub}\n')

    for i in range(pocket_n):
        for j in range(i + 1, pocket_n):
            lb = to_scientific_notation(pocket_dm[j, i])
            ub = to_scientific_notation(pocket_dm[i, j])
            outfile.write(f'{ligand_n + i + 1:9.0f}{ligand_n + j + 1:10.0f}   {lb}   {ub}\n')

    # drop = np.where(var < np.quantile(var, 0.8), 0, 1)
    # drop = np.where((mu > np.quantile(mu, 0.9)) & (var > np.quantile(var, 0.9)), 1, 0)
    # print(np.mean(drop))
    # drop = np.zeros_like(mu)
    # drop[np.argsort(mu, axis=1)[:10]] = 1
    for i in range(ligand_n):
        for j in range(pocket_n):
            mu_i = mu[i, j]
            var_i = k * np.log(var[i, j] + 1.01)
            # if drop[i, j]:
            #     continue
            lb = np.clip(mu_i - var_i, 1.0, 30.0)
            ub = np.clip(mu_i + var_i, 3.0, 30.0)
            outfile.write(f'{i + 1:9.0f}{ligand_n + j + 1:10.0f}   {to_scientific_notation(lb)}   {to_scientific_notation(ub)}\n')

    outfile.close()


def run_dgsol(input_path, output_path, summary_path, n_sol=10):
    cmd = f'{DGSOL} -s{n_sol} {input_path} {output_path} {summary_path} > /dev/null'
    call(cmd, shell=True)


def parse_dgsol_errors(filepath):
    '''
    There are 4 types of errors in the dgsol output:

    f_err      The value of the merit function
    derr_min      The smallest error in the distances
    derr_avg      The average error in the distances
    derr_max      The largest error in the distances
    '''
    with open(filepath, 'r') as input:
        lines = input.readlines()

    errors = []
    # skip the header lines
    for line in lines[5:]:
        errors.append(line.split()[2:])   # the first two entries are n_atoms and n_distances
    return np.array(errors).astype('float32')


def get_results(output_path, summary_path, ligand_n, pocket_n):
    coords = []
    for line in open(output_path).read().split('\n'):
        if not line.startswith('\n') and len(line) > 30:
            coords.append([float(n) for n in line.split()])
    coords = np.array(coords).reshape((-1, ligand_n + pocket_n, 3))

    errors = parse_dgsol_errors(summary_path)
    idx = np.argmin(errors[:, 2])
    coords = coords[idx]
    return coords


def align_coords(coords, ligand_n, pocket_coords):
    recon_ligand_coords, recon_pocket_coords = coords[:ligand_n], coords[ligand_n:]
    ref_center = np.mean(pocket_coords, axis=0)
    ref_coords = pocket_coords - ref_center
    centroid = np.mean(recon_pocket_coords, axis=0)
    recon_pocket_coords -= centroid
    recon_ligand_coords -= centroid
    cm = np.dot(np.transpose(recon_pocket_coords), ref_coords)
    u, d, vt = np.linalg.svd(cm)
    rot = np.transpose(np.dot(np.transpose(vt), np.transpose(u)))
    recon_pocket_coords = np.dot(recon_pocket_coords, rot)
    recon_ligand_coords = np.dot(recon_ligand_coords, rot)
    recon_pocket_coords += ref_center
    recon_ligand_coords += ref_center
    return recon_pocket_coords, recon_ligand_coords


def c_to_d(coords):
    return np.linalg.norm(np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1), axis=2)


def get_rmsd(mobile, reference):
    return np.sqrt(np.sum((mobile - reference) ** 2) / len(mobile))



class Minimizer:
    def __init__(self):
        # check whether we have a GPU platform and if so set the precision to mixed
        self.platform = None
        self.speed = 0
        for i in range(Platform.getNumPlatforms()):
            p = Platform.getPlatform(i)
            if p.getSpeed() > self.speed:
                self.platform = p
                self.speed = p.getSpeed()
             # TODO: force cpu
            # if p.getName() == 'CPU': break

        if self.platform.getName() == 'CUDA' or self.platform.getName() == 'OpenCL':
            self.platform.setPropertyDefaultValue('Precision', 'mixed')
            print('Set precision for platform', self.platform.getName(), 'to mixed')

        self.temperature = 300 * unit.kelvin
        ff_kwargs = {
            'constraints': app.HBonds,
            'rigidWater': True,
            'removeCMMotion': False,
            'hydrogenMass': 4 * unit.amu
        }
        self.system_generator = SystemGenerator(
            forcefields=['amber/protein.ff14SB.xml'],
            small_molecule_forcefield='gaff-2.11',
            forcefield_kwargs=ff_kwargs
        )

    def get_restraint_force(self, modeller):
        const = 4 * K_UNIT
        force = openmm.CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        force.addGlobalParameter('k', const)
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')
        for i, (crd, atom) in enumerate(zip(modeller.positions, modeller.topology.atoms())):
            if hasattr(crd, 'x'):
                if atom.name == 'CA':
                    force.addParticle(i, crd.value_in_unit(unit.nanometer))
        return force

    def get_dock_force(self, ligand_idx, protein_idx, mu, var, k_scaling=2.0, cutoff=5):
        force = openmm.HarmonicBondForce()
        # cutoff = np.sort(var.flatten())[int(cutoff * var.size)]
        for i, l_idx in enumerate(ligand_idx):
            for j, p_idx in enumerate(protein_idx):
                r = mu[i, j]
                # v = var[i, j]
                # if v > cutoff: continue
                if r > cutoff: continue
                k = k_scaling / r
                force.addBond(int(l_idx), int(p_idx), float(r) * unit.angstrom, float(k) * K_UNIT)
        return force

    def create_system(self, path, ligand_mol, pred_pocket_coords, mu, var):
        protein_pdb = PDBFile(os.path.join(path, 'protein.pdb'))
        # protein_pdb = PDBFile(os.path.join(path, 'protein.pdb'))
        protein_coords = np.array([[c.x, c.y, c.z] for c in protein_pdb.positions]) * 10

        # Prepare and create openmm system for docking
        rdkitmolh = Molecule.from_rdkit(ligand_mol, allow_undefined_stereo=True)
        modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
        modeller.add(rdkitmolh.to_topology().to_openmm(), rdkitmolh.conformers[0])

        # Hacky way to ensure indexing is correct
        dis_mat = np.linalg.norm(pred_pocket_coords.reshape(1, -1, 3) - protein_coords.reshape(-1, 1, 3), axis=-1)
        protein_idx = np.argmin(dis_mat, axis=0)
        ligand_atoms = list(modeller.topology.chains())[-1].atoms()
        ligand_idx = np.array([a.index for a in ligand_atoms if a.element.symbol != 'H'])

        system = self.system_generator.create_system(modeller.topology, molecules=rdkitmolh)
        system.addForce(self.get_dock_force(ligand_idx, protein_idx, mu, var, k_scaling=1.0, cutoff=100))
        system.addForce(self.get_restraint_force(modeller))
        return system, modeller

    def minimize(self, path, ligand_mol, pred_pocket_coords, mu, var, tolerance=0.5):
        key = os.path.basename(path)
        output_path = os.path.join(path, f'{key}_min.pdb')
        if not os.path.exists(output_path):
            system, modeller = self.create_system(path, ligand_mol, pred_pocket_coords, mu, var)

            # Run minimization
            integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
            integrator.setRandomNumberSeed(123)
            simulation = Simulation(modeller.topology, system, integrator, platform=self.platform)
            simulation.context.setPositions(modeller.positions)
            simulation.minimizeEnergy(tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer)
            # state = simulation.context.getState(getEnergy=True)
            # energy = state.getPotentialEnergy()._value

            # write out the minimised PDB
            with open(output_path, 'w') as outfile:
                PDBFile.writeFile(
                    modeller.topology,
                    simulation.context.getState(
                        getPositions=True,
                        enforcePeriodicBox=False
                    ).getPositions(),
                    file=outfile,
                    keepIds=True
                )

        # quickly load minimized coordinates
        skip_res = ['NME', 'ACE']
        coords = np.array([
            [float(l[30:38]), float(l[38:46]), float(l[46:54])]
            for l in open(output_path, 'r').read().strip().split('\n')
            if l[:6] == 'HETATM' and l[17:20] not in skip_res and l[76:78].strip() != 'H'
        ])
        return coords
