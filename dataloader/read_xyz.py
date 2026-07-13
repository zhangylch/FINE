import ase.io
from ase import Atoms
import numpy as np
from typing import List, Dict, Any
import re

def read_xyz(
    filename,
    force_table=True,
    stress_table=False,
    dipole_table=False,
    bec_table=False,
):
    
    # 手动读取文件内容
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    coor = []
    cell = []
    pbc = []
    pot = [] 
    field = []
    force = None
    stress = None
    dipole = None
    bec = None
    species = []
    numatoms = []
    ncol = 4
    if force_table: force = []
    if stress_table: stress = []
    if dipole_table: dipole = []
    if bec_table:
        ncol = 13
        bec = []

    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
            
        natoms = int(lines[index].strip())
        numatoms.append(natoms)
        index += 1
        
        header = lines[index].strip()
        index += 1
        
        lattice_match = re.search(r'Lattice="([^"]+)"', header)
        icell = np.eye(3)
        if lattice_match:
            icell = np.fromstring(lattice_match.group(1), sep=' ').reshape(3,3)
        cell.append(icell)
        
        pbc_match = re.search(r'pbc="([^"]+)"', header)
        ipbc = [False, False, False]
        if pbc_match:
            ipbc = [x == 'T' for x in pbc_match.group(1).split()]
        pbc.append(np.array(ipbc).astype(np.int32))
        
        energy_match = re.search(r'energy=([-\d.eE+]+)', header)
        energy = float(energy_match.group(1)) if energy_match else None
        pot.append(energy)

        field_match = re.search(r'(?:Field|field)="([^"]+)"', header)
        if field_match:
            ifield = np.fromstring(field_match.group(1), sep=' ').reshape(3)
        else:
            ifield = np.zeros(3)
        field.append(ifield)
        
        if stress_table:
            stress_match = re.search(r'stress="([^"]+)"', header)
            if stress_match:
                istress = np.fromstring(stress_match.group(1), sep=' ').reshape(3,3)
                stress.append(istress)

        if dipole_table:
            dipole_match = re.search(r'dipole="([^"]+)"', header)
            if dipole_match:
                idipole = np.fromstring(dipole_match.group(1), sep=' ').reshape(3)
            else:
                idipole = np.zeros(3)
            dipole.append(idipole)
        
        symbols = []
        positions = []
        iforce = []
        ibec = []
        for _ in range(natoms):
            parts = lines[index].split()
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])
            if bec_table: ibec.append([float(x) for x in parts[4:13]])
            if force_table: iforce.append([float(x) for x in parts[ncol:ncol+3]])
            index += 1
        coor.append(np.array(positions, order="F").T)
        if force_table: force.append(np.array(iforce))
        if bec_table: bec.append(np.array(ibec))
          
        atoms = Atoms(symbols=symbols, positions=positions, cell=icell, pbc=ipbc)
        species.append(atoms.get_atomic_numbers())
        
        
    return coor, field, cell, pbc, species, numatoms, pot, force, stress, dipole, bec
