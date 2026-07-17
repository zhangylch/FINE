#! /usr/bin/env python3

import sys
import numpy as np
from src.read_json import load_config
from src.gpu_sel import gpu_sel

full_config = load_config("full_config.json")
gpu_sel(full_config.local_size)

import train_model.MPNN as MPNN
import dataloader.dataloader as dataloader
import dataloader.cudaloader as cudaloader
import jax
import jax.numpy as jnp
from src.save_checkpoint import restore_checkpoint
from src.data_config import ModelConfig, checkpoint_tp_config
from src.jax_sharding import device_put_pmap_replicated, get_jax_devices

# Configure JAX precision.
if full_config.jnp_dtype=='float64':
    jax.config.update("jax_enable_x64", True)

if full_config.jnp_dtype=='float32':
    jax.config.update("jax_default_matmul_precision", "highest")

data_load = dataloader.Dataloader(full_config.maxneigh_per_node, full_config.batchsize, local_size=full_config.local_size, initpot=full_config.initpot, ncyc=full_config.ncyc, cutoff=full_config.cutoff, datafolder=full_config.datafolder, ene_shift=full_config.ene_shift, force_table=full_config.force_table, stress_table=full_config.stress_table, dipole_table=full_config.dipole_table, bec_table=full_config.bec_table, cross_val=full_config.cross_val, jnp_dtype=full_config.jnp_dtype, seed=full_config.data_seed, Fshuffle=False, ntrain=full_config.ntrain, eval_mode=True, node_cap=full_config.node_cap, edge_cap=full_config.edge_cap)
# generate random data for initialization

#ntrain = data_load.ntrain
numatoms = data_load.numatoms[:full_config.ntrain]
ntrain = full_config.ntrain#jnp.sum(numatoms * numatoms)
natom = np.sum(numatoms)
nforce = np.sum(numatoms) * 3

nprop = 1
prop_length = full_config.ntrain
if full_config.stress_table:
    nprop = 3
    prop_length = jnp.array(np.array([ntrain, nforce, full_config.ntrain*9]))
elif full_config.force_table and full_config.dipole_table and full_config.bec_table:
    nprop = 4
    prop_length = jnp.array(np.array([ntrain, nforce, 3*full_config.ntrain, natom*9]))
elif full_config.force_table and full_config.dipole_table:
    nprop = 3
    prop_length = jnp.array(np.array([ntrain, nforce, 3*full_config.ntrain]))
elif full_config.force_table:
    nprop = 2
    prop_length = jnp.array(np.array([ntrain, nforce]))
elif full_config.dipole_table:
    nprop = 2
    prop_length = jnp.array(np.array([ntrain, 3*full_config.ntrain]))

data_load = cudaloader.CudaDataLoader(data_load, queue_size=full_config.queue_size)


devices = get_jax_devices(full_config.local_size, log=True)
restored = restore_checkpoint(
    full_config.ckpath, 
    devices
)

if restored is not None:
    start_step, params, ema_params, opt_state, model_config = restored

#==============================Equi MPNN==============================================================
model_config = checkpoint_tp_config(model_config, full_config)
config = ModelConfig(**model_config)

model = MPNN.MPNN(config)

if full_config.stress_table:
    def pes_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        (_, ene), (force, stress) = jax.value_and_grad(model.apply, argnums=[1, 4], has_aux=True)(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        volume = jnp.sum(cell[:, 0] * jnp.cross(cell[:, 1], cell[:, 2]), axis=-1)
        return ene, force, stress/volume[:, None, None]*jnp.array(full_config.stress_sign)
elif full_config.force_table and full_config.dipole_table and full_config.bec_table:
    def pes_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        (_, ene), (force, dipole) = jax.value_and_grad(model.apply, argnums=[1, 2], has_aux=True)(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)

        def energy_only(coor_arg, field_arg):
            total_energy, _ = model.apply(params, coor_arg, field_arg, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
            return total_energy

        bec_full = jax.jacfwd(jax.grad(energy_only, argnums=0), argnums=1)(coor, field)
        bec = bec_full[jnp.arange(coor.shape[0]), :, celllist, :]
        return ene, force, dipole*jnp.array(full_config.dipole_sign), bec*jnp.array(full_config.bec_sign)
elif full_config.force_table and full_config.dipole_table:
    def pes_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        (_, ene), (force, dipole) = jax.value_and_grad(model.apply, argnums=[1, 2], has_aux=True)(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        return ene, force, dipole*jnp.array(full_config.dipole_sign)
elif full_config.force_table:
    def pes_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        (_, ene), force = jax.value_and_grad(model.apply, argnums=1, has_aux=True)(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        return ene, force
elif full_config.dipole_table:
    def pes_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        (_, ene), dipole = jax.value_and_grad(model.apply, argnums=2, has_aux=True)(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        return ene, dipole*jnp.array(full_config.dipole_sign)
else:
    def pes_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        _, ene = model.apply(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        return ene,

def make_loss(pes_model, nprop):

    def get_loss(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop):

        nnprop = pes_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        ploss = jnp.zeros(nprop)
        if full_config.force_table and full_config.dipole_table and full_config.bec_table:
            abpot, abforce, abdipole, abbec = abprop
            nnpot, nnforce, nndipole, nnbec = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            delta_bec = (abbec - nnbec) * center_factor[:, None, None]
            ploss = ploss.at[0].set(jnp.sum(jnp.square(nnpot - abpot)))
            ploss = ploss.at[1].set(jnp.sum(jnp.square(nnforce - abforce)))
            ploss = ploss.at[2].set(jnp.sum(jnp.square(delta_dipole)))
            ploss = ploss.at[3].set(jnp.sum(jnp.square(delta_bec)))
        elif full_config.force_table and full_config.dipole_table:
            abpot, abforce, abdipole = abprop
            nnpot, nnforce, nndipole = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            ploss = ploss.at[0].set(jnp.sum(jnp.square(nnpot - abpot)))
            ploss = ploss.at[1].set(jnp.sum(jnp.square(nnforce - abforce)))
            ploss = ploss.at[2].set(jnp.sum(jnp.square(delta_dipole)))
        elif full_config.dipole_table:
            abpot, abdipole = abprop
            nnpot, nndipole = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            ploss = ploss.at[0].set(jnp.sum(jnp.square(nnpot - abpot)))
            ploss = ploss.at[1].set(jnp.sum(jnp.square(delta_dipole)))
        else:
            for i, iprop in enumerate(abprop):
                ploss = ploss.at[i].set(jnp.sum(jnp.square(nnprop[i] - iprop)))
        
        return ploss


    return get_loss
 
value_fn = make_loss(pes_model, nprop)

def val_loop(nstep):
    def get_loss(params, ploss_out, data):
        def body(i, carry):
            params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop, ploss_fn = carry
            inabprop = (iabprop[i] for iabprop in abprop)
            ploss = value_fn(params, coor[i], field[i], cell[i], disp_cell[i], neighlist[i], celllist[i], shiftimage[i], center_factor[i], species[i], inabprop)
            ploss_fn = ploss_fn + ploss
            return params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop, ploss_fn

        coor, field, cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop = data
        disp_cell = jnp.zeros_like(cell)
        params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop, ploss_out = \
        jax.lax.fori_loop(0, nstep, body, (params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop, ploss_out))
        return ploss_out
    return get_loss


val_ens = jax.pmap(val_loop(full_config.ncyc), axis_name="eval_GPUs")
ploss_val = device_put_pmap_replicated(jnp.zeros((nprop,)), devices)
for data in data_load:
    ploss_val = val_ens(ema_params, ploss_val, data)

ploss_val = jnp.sqrt(jnp.sum(ploss_val, axis=0) / prop_length)
print(ploss_val)

