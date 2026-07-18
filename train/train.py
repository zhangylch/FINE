#! /usr/bin/env python3
import os
import sys
import pickle
import math
import time
import numpy as np
from src.params import *
import train_model.MPNN as MPNN
import dataloader.dataloader as dataloader
import dataloader.cudaloader as cudaloader
import src.print_info as print_info
import optax
from src.print_params import print_params
from src.save_checkpoint import save_checkpoint, restore_checkpoint
from jax import vmap, jit
from optax import tree_utils as otu
from src.data_config import ModelConfig
from src.jax_sharding import device_put_pmap_replicated, get_jax_devices
from dataclasses import replace, asdict
import json
from typing import Optional, Any


def zero_nonfinite_gradients(grads):
    def clean_leaf(grad):
        if grad is None or not hasattr(grad, "dtype"):
            return grad
        if not jnp.issubdtype(grad.dtype, jnp.inexact):
            return grad
        return jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad))

    return jax.tree_util.tree_map(clean_leaf, grads)


def loss_normalizers(center_factor, celllist, numatoms):
    dtype = center_factor.dtype
    graph_atoms = jax.ops.segment_sum(
        center_factor,
        celllist,
        num_segments=numatoms.shape[0],
    )
    graph_mask = (graph_atoms > 0).astype(dtype)
    atom_mask = center_factor.astype(dtype)
    safe_numatoms = jnp.maximum(numatoms, jnp.array(1.0, dtype=dtype))
    atom_numatoms = safe_numatoms[celllist]
    # Denominator for masked batch means; raw jnp.mean would include padding graphs.
    graph_count = jnp.maximum(jnp.sum(graph_mask), jnp.array(1.0, dtype=dtype))
    return graph_mask, atom_mask, safe_numatoms, atom_numatoms, graph_count


# train function
def train(params, ema_params, config, optim, opt_state, lr_state, schedule_fn, value_and_grad_fn, value_fn, data_load, warm_lr, slr, elr, warm_epoch, Epoch, ncyc, ntrain, nval, nprop, start_step):

    def train_loop(nstep):

        def optimize_epoch(params, opt_state, ema_params, scale, loss_out, weight, data):

            def body(i, carry):
                params, opt_state, ema_params, scale, weight, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, loss_fn = carry
                inabprop = (iabprop[i] for iabprop in abprop)
                loss, grads = value_and_grad_fn(params, coor[i], field[i], cell[i], disp_cell[i], neighlist[i], celllist[i], shiftimage[i], center_factor[i], species[i], numatoms[i], inabprop, weight)
                _, _, _, _, graph_count = loss_normalizers(center_factor[i], celllist[i], numatoms[i])
                grads = zero_nonfinite_gradients(grads)
                grads = jax.lax.pmean(grads, axis_name="train_GPUs")
                grads = zero_nonfinite_gradients(grads)
                updates, opt_state = optim.update(grads, opt_state, params)
                updates = otu.tree_scalar_mul(scale, updates)
                params = optax.apply_updates(params, updates)
                ema_params = optax.incremental_update(params, ema_params, 0.001)
                loss_fn += loss * graph_count
                return params, opt_state, ema_params, scale, weight, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, loss_fn
            
            coor, field, cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop = data
            disp_cell = jnp.zeros_like(cell)
            params, opt_state, ema_params, scale, weight, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, loss_out = \
            jax.lax.fori_loop(0, nstep, body, (params, opt_state, ema_params, scale, weight, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, loss_out))
            return params, opt_state, ema_params, loss_out

        return optimize_epoch
        #return optimize_epoch

    
    def val_loop(nstep):
        def get_loss(params, scale, loss_out, ploss_out, weight, data):
            def body(i, carry):
                params, weight, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, loss_fn, ploss_fn = carry
                inabprop = (iabprop[i] for iabprop in abprop)
                loss, ploss = value_fn(params, coor[i], field[i], cell[i], disp_cell[i], neighlist[i], celllist[i], shiftimage[i], center_factor[i], species[i], numatoms[i], inabprop, weight)
                loss_fn = loss_fn + loss
                ploss_fn = ploss_fn + ploss
                return params, weight, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, loss_fn, ploss_fn

            coor, field, cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop = data
            disp_cell = jnp.zeros_like(cell)
            params, weight, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, loss_out, ploss_out = \
            jax.lax.fori_loop(0, nstep, body, (params, weight, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, loss_out, ploss_out))
            return loss_out, ploss_out
        return get_loss

    devices = get_jax_devices(full_config.local_size)
    train_ens = jax.pmap(train_loop(ncyc), axis_name="train_GPUs")
    val_ens = jax.pmap(val_loop(ncyc), axis_name="val_GPUs")

    print_err = print_info.Print_Info(ferr)

    best_loss = jnp.sum(jnp.array([1e20]))

   
    scale = device_put_pmap_replicated(warm_lr / slr, devices)
    max_scale =  slr / warm_lr
    weight = device_put_pmap_replicated(jnp.array(full_config.init_weight), devices)
    init_weight = device_put_pmap_replicated(jnp.array(full_config.init_weight), devices)
    final_weight = device_put_pmap_replicated(jnp.array(full_config.final_weight), devices)
    ones_replicated = device_put_pmap_replicated(jnp.array(1.0), devices)
    for iepoch in range(Epoch): 

        loss_train = device_put_pmap_replicated(jnp.array(0.0), devices)
        for data in data_load:
            params, opt_state, ema_params, loss_train = train_ens(params, opt_state, ema_params, scale, loss_train, weight, data)
        out_train = jnp.sqrt(jnp.sum(loss_train) / ntrain)

        loss_val = device_put_pmap_replicated(jnp.array(0.0), devices)
        ploss_val = device_put_pmap_replicated(jnp.zeros((nprop,)), devices)
        for data in data_load:
            loss_val, ploss_val = val_ens(ema_params, scale, loss_val, ploss_val, weight, data)
        out_val = jnp.sqrt(jnp.sum(loss_val) / nval)
        ploss_out = jnp.sqrt(jnp.sum(ploss_val, axis=0) / nval)



# print and save information
        lr = slr * scale[0]
        print_err(iepoch, lr, out_train, out_val, ploss_out)

        _, lr_state = schedule_fn.update(updates=params, state=lr_state, value=out_val)
        scale = ones_replicated * lr_state.scale * (1.0 / max_scale + min(((iepoch+1) / warm_epoch), 1) * (1.0 - 1.0 / max_scale))

        if iepoch > warm_epoch:
            weight = final_weight + (lr - elr) / (slr - elr) * (init_weight-final_weight)

        if out_val > 1e1 * best_loss or (not jnp.isfinite(out_train)) or (not jnp.isfinite(out_val)):

            restored = restore_checkpoint(
                full_config.ckpath, 
                devices
            )
            
            if restored is not None:
                start_step, params, ema_params, _, _ = restored
                opt_state = optim.init(params)
                params = device_put_pmap_replicated(params, devices)
                ema_params = device_put_pmap_replicated(ema_params, devices)
                opt_state = device_put_pmap_replicated(opt_state, devices)
    

        if out_val < best_loss:
            best_loss = out_val
             
            start_step = start_step + 1
            aveparams = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), params)
            ave_ema_params = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), ema_params)
            ave_opt_state = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), opt_state)
            save_checkpoint(
                step=start_step, 
                params=aveparams, 
                ema_params=ave_ema_params, 
                opt_state=ave_opt_state, 
                config=config, 
                ckpt_dir=full_config.ckpath,
                max_to_keep=5  # keep the latest checkpoints
            )

            print(f"Step {start_step}: Saved checkpoint")

            

        if lr < elr+1e-10: 
            break
        sys.stdout.flush()

    

key = jax.random.split(key[-1], 2)

data_load = dataloader.Dataloader(full_config.maxneigh_per_node, full_config.batchsize, local_size=full_config.local_size, initpot=full_config.initpot, ncyc=full_config.ncyc, cutoff=full_config.cutoff, datafolder=full_config.datafolder, ene_shift=full_config.ene_shift, force_table=full_config.force_table, stress_table=full_config.stress_table, dipole_table=full_config.dipole_table, bec_table=full_config.bec_table, cross_val=full_config.cross_val, jnp_dtype=full_config.jnp_dtype, seed=full_config.data_seed, Fshuffle=full_config.Fshuffle, ntrain=full_config.ntrain, node_cap = full_config.node_cap, edge_cap = full_config.edge_cap)

full_config = replace(full_config, initpot=data_load.initpot)
with open("full_config.json", "w") as f:
    json.dump(asdict(full_config), f, indent=4) 
# get some system information
ntrain = full_config.ntrain
nval = data_load.nval
nspec = data_load.nspec
reduce_spec = jnp.array(data_load.reduce_spec)
com_spec = jnp.array(data_load.com_spec)
force_std = data_load.std

nprop = 1
if full_config.stress_table:
    nprop = 3
elif full_config.force_table and full_config.dipole_table and full_config.bec_table:
    nprop = 4
elif full_config.force_table and full_config.dipole_table:
    nprop = 3
elif full_config.force_table:
    nprop = 2
elif full_config.dipole_table:
    nprop = 2

final_weight = jnp.array(full_config.final_weight[:nprop])
init_weight = jnp.array(full_config.init_weight[:nprop])

get_jax_devices(full_config.local_size, log=True)
data_load = cudaloader.CudaDataLoader(data_load, queue_size=full_config.queue_size)
for data in data_load:
    pass

for data in data_load:
    pass

get_gpu0_data_op = lambda sharded_array: sharded_array[0]
data_on_gpu0_pytree = jax.tree.map(get_gpu0_data_op, data)
coor, field, cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop = data_on_gpu0_pytree

initdata = (coor[0], field[0], cell[0], jnp.zeros_like(cell[0]), neighlist[0], celllist[0], shiftimage[0], center_factor[0], species[0])

#=================================================Equi MPNN===================================================================
config = ModelConfig(nspec=nspec, emb_nl=full_config.emb_nl, MP_nl=full_config.MP_nl, radial_nl=full_config.radial_nl, out_nl=full_config.out_nl, reduce_spec=reduce_spec, com_spec=com_spec, index_l=index_l, initbias_neigh=initbias_neigh, cutoff=full_config.cutoff, npaircode=full_config.npaircode, nradial=full_config.nradial, nwave=full_config.nwave, rmaxl=rmaxl, prmaxl=prmaxl, MP_loop=full_config.MP_loop, pn=full_config.pn, tp_method=full_config.tp_method, tp_mode=full_config.tp_mode, use_norm=full_config.use_norm, use_bias=full_config.use_bias, std=force_std, cst=1.67462)

model = MPNN.MPNN(config)

params_rng = {"params": key[1]}

params = model.init(params_rng, *initdata)

print("NN structure for pes")
print_params(params)

#ceta = jnp.pi/5
#rotate = jnp.array([[1, 0, 0], [0, jnp.cos(ceta), jnp.sin(ceta)], [0, -jnp.sin(ceta), jnp.cos(ceta)]])


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

def make_gradient(energy_model):

    def wf_loss(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, weight):

        nnprop = energy_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        graph_mask, atom_mask, safe_numatoms, atom_numatoms, graph_count = loss_normalizers(center_factor, celllist, numatoms)
        if full_config.stress_table:
            abpot, abforce, abstress = abprop
            nnpot, nnforce, nnstress = nnprop
            loss = weight[0] * jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask) \
                 + weight[1] * jnp.sum(jnp.square(abforce - nnforce) * atom_mask[:, None] / (jnp.array(3.0) * atom_numatoms)[:, None]) \
                 + weight[2] * jnp.sum(jnp.square(abstress - nnstress) * graph_mask[:, None, None] / jnp.array(9.0))
        elif full_config.force_table and full_config.dipole_table and full_config.bec_table:
            abpot, abforce, abdipole, abbec = abprop
            nnpot, nnforce, nndipole, nnbec = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            delta_bec = abbec - nnbec
            loss = weight[0] * jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask) \
                 + weight[1] * jnp.sum(jnp.square(abforce - nnforce) * atom_mask[:, None] / (jnp.array(3.0) * atom_numatoms)[:, None]) \
                 + weight[2] * jnp.sum(jnp.square(delta_dipole) * graph_mask[:, None] / jnp.array(3.0)) \
                 + weight[3] * jnp.sum(jnp.square(delta_bec) * atom_mask[:, None, None] / (jnp.array(9.0) * atom_numatoms)[:, None, None])
        elif full_config.force_table and full_config.dipole_table:
            abpot, abforce, abdipole = abprop
            nnpot, nnforce, nndipole = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            loss = weight[0] * jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask) \
                 + weight[1] * jnp.sum(jnp.square(abforce - nnforce) * atom_mask[:, None] / (jnp.array(3.0) * atom_numatoms)[:, None]) \
                 + weight[2] * jnp.sum(jnp.square(delta_dipole) * graph_mask[:, None] / jnp.array(3.0))
        elif full_config.force_table:
            abpot, abforce = abprop
            nnpot, nnforce = nnprop
            loss = weight[0] * jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask) \
                 + weight[1] * jnp.sum(jnp.square(abforce - nnforce) * atom_mask[:, None] / (jnp.array(3.0) * atom_numatoms)[:, None])
        elif full_config.dipole_table:
            abpot, abdipole = abprop
            nnpot, nndipole = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            loss = weight[0] * jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask) \
                 + weight[1] * jnp.sum(jnp.square(delta_dipole) * graph_mask[:, None] / jnp.array(3.0))
        else:
            abpot, = abprop
            nnpot, = nnprop
            loss = jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask) * weight[0]
        
        return loss / graph_count


    return jax.value_and_grad(wf_loss)
        
value_and_grad_fn = make_gradient(pes_model)

def make_loss(pes_model, nprop):

    def get_loss(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop, weight):

        nnprop = pes_model(params, coor, field, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        graph_mask, atom_mask, safe_numatoms, atom_numatoms, _ = loss_normalizers(center_factor, celllist, numatoms)
        if full_config.stress_table:
            abpot, abforce, abstress = abprop
            nnpot, nnforce, nnstress = nnprop
            loss1 = jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask)
            loss2 = jnp.sum(jnp.square(abforce - nnforce) * atom_mask[:, None] / (jnp.array(3.0) * atom_numatoms)[:, None])
            loss3 = jnp.sum(jnp.square(abstress - nnstress) * graph_mask[:, None, None] / jnp.array(9.0))
            ploss = jnp.stack([loss1, loss2, loss3])
            loss = loss1*weight[0] + loss2*weight[1] + loss3*weight[2]
        elif full_config.force_table and full_config.dipole_table and full_config.bec_table:
            abpot, abforce, abdipole, abbec = abprop
            nnpot, nnforce, nndipole, nnbec = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            delta_bec = abbec - nnbec
            loss1 = jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask)
            loss2 = jnp.sum(jnp.square(abforce - nnforce) * atom_mask[:, None] / (jnp.array(3.0) * atom_numatoms)[:, None])
            loss3 = jnp.sum(jnp.square(delta_dipole) * graph_mask[:, None] / jnp.array(3.0))
            loss4 = jnp.sum(jnp.square(delta_bec) * atom_mask[:, None, None] / (jnp.array(9.0) * atom_numatoms)[:, None, None])
            ploss = jnp.stack([loss1, loss2, loss3, loss4])
            loss = loss1*weight[0] + loss2*weight[1] + loss3*weight[2] + loss4*weight[3]
        elif full_config.force_table and full_config.dipole_table:
            abpot, abforce, abdipole = abprop
            nnpot, nnforce, nndipole = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            loss1 = jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask)
            loss2 = jnp.sum(jnp.square(abforce - nnforce) * atom_mask[:, None] / (jnp.array(3.0) * atom_numatoms)[:, None])
            loss3 = jnp.sum(jnp.square(delta_dipole) * graph_mask[:, None] / jnp.array(3.0))
            ploss = jnp.stack([loss1, loss2, loss3])
            loss = loss1*weight[0] + loss2*weight[1] + loss3*weight[2]
        elif full_config.force_table:
            abpot, abforce = abprop
            nnpot, nnforce = nnprop
            loss1 = jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask)
            loss2 = jnp.sum(jnp.square(abforce - nnforce) * atom_mask[:, None] / (jnp.array(3.0) * atom_numatoms)[:, None])
            ploss = jnp.stack([loss1, loss2])
            loss = loss1*weight[0] + loss2*weight[1]
        elif full_config.dipole_table:
            abpot, abdipole = abprop
            nnpot, nndipole = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.round(jnp.einsum("ij,ijk->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij,ijk->ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            loss1 = jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask)
            loss2 = jnp.sum(jnp.square(delta_dipole) * graph_mask[:, None] / jnp.array(3.0))
            ploss = jnp.stack([loss1, loss2])
            loss = loss1*weight[0] + loss2*weight[1]
        else:
            abpot, = abprop
            nnpot, = nnprop
            ploss = jnp.sum(jnp.square((abpot - nnpot) / safe_numatoms) * graph_mask)
            loss = ploss * weight[0]
        return loss, ploss


    return get_loss
 
value_fn = make_loss(pes_model, nprop)       


schedule_fn = optax.contrib.reduce_on_plateau(factor=full_config.decay_factor, patience=full_config.patience_step, cooldown=full_config.cooldown, min_scale=full_config.elr/full_config.slr)

optim = optax.chain(
    optax.add_decayed_weights(full_config.weight_decay),
    optax.clip_by_global_norm(full_config.clip_norm),
    optax.amsgrad(learning_rate=full_config.slr),
)

opt_state = optim.init(params)
lr_state = schedule_fn.init(params)
    

ferr=open("nn.err","w")
ferr.write("Hybrid Equivariant MPNN package based on three-body descriptors \n")
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))

                                    
start_step = 0
devices = get_jax_devices(full_config.local_size)
params = device_put_pmap_replicated(params, devices)
ema_params = params
opt_state = device_put_pmap_replicated(opt_state, devices)
if full_config.restart:
    restored = restore_checkpoint(
        full_config.ckpath, 
        devices
    )

    if restored is not None:
        start_step, params, ema_params, _, _ = restored
        opt_state = optim.init(params)
        params = device_put_pmap_replicated(params, devices)
        ema_params = device_put_pmap_replicated(ema_params, devices)
        opt_state = device_put_pmap_replicated(opt_state, devices)
    


train(params, ema_params, config, optim, opt_state, lr_state, schedule_fn, value_and_grad_fn, value_fn, data_load, full_config.warm_lr, full_config.slr, full_config.elr, full_config.warm_epoch, full_config.Epoch, full_config.ncyc, full_config.ntrain, nval, nprop, start_step)
         
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
ferr.close()
print("Normal termination")
