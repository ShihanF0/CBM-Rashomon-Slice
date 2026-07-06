import os
import sys
import pickle
from PIL import Image
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm.auto import tqdm

import math
from statistics import mean, stdev
import torch
import torch.nn.functional as F
import torchvision
from transformers import AutoModel, CLIPVisionModel
import numpy as np
from torch.utils.data import DataLoader
from analysis import Logger, AverageMeter, accuracy, binary_accuracy
from datasets.cifar10_dataset import load_cifar10_data, setup_cifar10_dataset
from datasets.Awa2_dataset import load_awa2_data, setup_awa2_dataset
from datasets.CelebA_dataset import load_celeba_data, setup_celeba_dataset
from datasets.CUB_dataset import load_CUB_data, setup_CUB_dataset
from datasets.HAM10000_dataset import load_ham10000_data, setup_ham10000_dataset
from config import BASE_DIR, MIN_LR, LR_DECAY_SIZE, CUB_N_CLASSES, AWA2_N_CLASSES, CIFAR10_N_CLASSES, CELEBA_N_CLASSES, HAM10000_N_CLASSES,WANDB_PROJECT, WANDB_ENTITY, WANDB_ENABLE
from models import (
    EnsembleXtoCtoY, 
    EnsembleWithSharedXtoC, 
    EnsembleWithPartialSharing, 
    ConvParEnsemble,
    SingleE2EBranch, 
    SConvParEnsemble,
    LoraEnsemble,
    LoraEnsemblePshared,
    LoraEnsembleshared,
    LoraEnsemble_gc,
    RevolverEnsemble,
    DropoutEnsemble,
    MEDICAL_VIT_MODEL_NAME,
)
import warnings
import wandb

def print_memory_usage(step_name=""):

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{step_name:^20s}] CUDA Memory: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def q_cosine(a, b, eps=1e-8):

    a_norm = F.normalize(a, p=2, dim=1, eps=eps)
    b_norm = F.normalize(b, p=2, dim=1, eps=eps)
    return (a_norm * b_norm).sum(dim=1).mean()

def alpha_from_avg_grad(avg_grad: torch.Tensor, args) -> torch.Tensor:
    """
    Map avg_grad -> alpha in [alpha_min, alpha_max].

    Note: avg_grad is computed as mean(abs(grad)), so avg_grad >= 0 and
    sigmoid(avg_grad) is in [0.5, 1). We rescale that to [0, 1) so alpha_min
    is attainable (e.g., avg_grad == 0 => alpha == alpha_min).
    """
    alpha_min = float(getattr(args, "alpha_min", 0.5))
    alpha_max = float(getattr(args, "alpha_max", 1.0))
    if alpha_max < alpha_min:
        raise ValueError(f"alpha_max ({alpha_max}) must be >= alpha_min ({alpha_min}).")
    t01 = (torch.sigmoid(avg_grad) - 0.5) * 2.0  # [0, 1)
    t01 = t01.clamp(0.0, 1.0)
    return alpha_min + (alpha_max - alpha_min) * t01

def log_metrics_to_wandb(phase, epoch, total_loss_meter, ensemble_acc_meter, detailed_meters, args, model=None):
    if not WANDB_ENABLE: return
    
    metrics = {
        f"{phase}/total_loss": total_loss_meter.avg,
        f"{phase}/ensemble_accuracy": ensemble_acc_meter.avg,
        "epoch": epoch
    }
    
    if model is not None and hasattr(model, 'alpha'):
        metrics[f"{phase}/model_alpha"] = model.alpha.item()
    
    for i in range(args.num_models):
        if detailed_meters["branch_total_loss"][i].count > 0:
            metrics.update({
                f"{phase}/branch_{i}_total_loss": detailed_meters["branch_total_loss"][i].avg,
                f"{phase}/branch_{i}_y_loss": detailed_meters["y_acc_loss"][i].avg,
                f"{phase}/branch_{i}_c_loss": detailed_meters["c_acc_loss"][i].avg,
                f"{phase}/branch_{i}_c_div": detailed_meters["c_div_loss"][i].avg,
                f"{phase}/branch_{i}_y_acc": detailed_meters["y_acc"][i].avg,
                f"{phase}/branch_{i}_c_acc": detailed_meters["c_acc"][i].avg,
            })
    
    wandb.log(metrics)

def init_wandb(args, model=None):
    if not WANDB_ENABLE: 
        return

    run_name = f"{args.exp}_{args.dataname}_{args.num_models}models_lambda{args.lambda_c_acc}_seed{args.seed}"

    tags = [args.exp, args.dataname, args.encoder, f"models_{args.num_models}"]
    if args.pretrained:
        tags.append("pretrained")
    if args.freeze:
        tags.append("freeze")
    if hasattr(args, 'wandb_tags') and args.wandb_tags:
        tags.extend(args.wandb_tags)

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        tags=tags,
        notes=getattr(args, 'wandb_notes', ''),
        config={
            "experiment_type": args.exp,
            "dataset": args.dataname,
            "encoder": args.encoder,
            "num_models": args.num_models,
            "seed": args.seed,

            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "scheduler_step": args.scheduler_step,
            "lambda_c_acc": args.lambda_c_acc,
            "beta": getattr(args, 'beta', None),

            "pretrained": args.pretrained,
            "freeze": args.freeze,
            "use_aux": args.use_aux,
            "n_attributes": args.n_attributes,
            "n_class_attr": args.n_class_attr,
            "expand_dim": args.expand_dim,

            "split_point": getattr(args, 'split_point', None),
            "bottleneck_dim": getattr(args, 'bottleneck_dim', None),
            "tuning_strategy": getattr(args, 'tuning_strategy', None),
            "tiny_lr": getattr(args, 'tiny_lr', None),
        }
    )

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    if model is not None:
        wandb.watch(model, log="gradients", log_freq=1000)


def run_epoch_ensemble_e2e(model, optimizer, loader, y_criterion, c_criterion, args, is_training):
    num_models    = model.num_models
    n_attributes  = args.n_attributes

    total_loss_meter = AverageMeter()
    y_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_div_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_acc_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_meters = [AverageMeter() for _ in range(num_models)]
    branch_total_loss_meters = [AverageMeter() for _ in range(num_models)]
    ensemble_y_acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    for batch_idx, batch in enumerate(loader):
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)
        
        if batch_idx == 0:
            print_memory_usage("Before Forward")
        
        if is_training:
            optimizer.zero_grad()

        all_y_preds, all_c_preds = model(inputs_var)

        y_losses, c_losses, d_losses = [], [], []
        concatenated = []
        for i in range(num_models):
            concatenated.append(torch.cat(all_c_preds[i], dim=1))
        if batch_idx == 0:
            print_memory_usage("After Forward")

        for i in range(num_models):
            loss_y = y_criterion(all_y_preds[i], y_labels_var)
            y_losses.append(loss_y)

            loss_c = sum(
                c_criterion[attr_idx](
                    all_c_preds[i][attr_idx].flatten(),
                    c_labels_var[:, attr_idx]
                )
                for attr_idx in range(n_attributes)
            ) / n_attributes
            c_losses.append(loss_c)

            sim = sum(q_cosine(concatenated[i], concatenated[j].detach())
                for j in range(num_models) if j != i
            )
            loss_d = 1.0 - sim / (num_models - 1)
            d_losses.append(loss_d)

            branch_total_loss = loss_y + args.lambda_c_acc * (loss_c - (model.alpha) * loss_d)
            branch_total_loss_meters[i].update(branch_total_loss.item(), inputs_var.size(0))

            y_acc_loss_meters[i].update(loss_y.item(), inputs_var.size(0))
            c_acc_loss_meters[i].update(loss_c.item(), inputs_var.size(0))
            c_div_loss_meters[i].update(loss_d.item(), inputs_var.size(0))

            acc_y_i = accuracy(all_y_preds[i], y_labels_var, topk=(1,))[0].item()
            y_acc_meters[i].update(acc_y_i, inputs_var.size(0))

            c_sig = torch.sigmoid(concatenated[i])
            acc_c_i = binary_accuracy(c_sig, c_labels_var).item()
            c_acc_meters[i].update(acc_c_i, inputs_var.size(0))

        total_y = sum(y_losses) / num_models
        total_c = sum(c_losses) / num_models
        total_d = sum(d_losses) / num_models

        max_y = max(y_losses)
        max_c = max(c_losses)

        total_loss = max_y + args.lambda_c_acc * (max_c - model.alpha * total_d)

        if is_training:
            total_loss.backward()
            with torch.no_grad():
                grad_sum, count = 0.0, 0
                for branch in model.branches:
                    for fc in branch.model_x_to_c.all_fc:
                        for p in fc.parameters():
                            if p.grad is not None:
                                grad_sum += p.grad.abs().mean()
                                count    += 1
                if count > 0:
                    avg_grad = grad_sum / count
                    model.alpha.data.copy_(alpha_from_avg_grad(avg_grad, args))
            optimizer.step()

        total_loss_meter.update(total_loss.item(), inputs_var.size(0))

        avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0)
        ens_acc  = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))

    detailed_meters = {
        "y_acc_loss": y_acc_loss_meters,
        "c_acc_loss": c_acc_loss_meters,
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters,
        "c_acc": c_acc_meters,
        "branch_total_loss": branch_total_loss_meters
    }
    return total_loss_meter, ensemble_y_acc_meter, detailed_meters

def run_epoch_convpar(model, optimizer, loader, y_criterion, c_criterion, args, is_training):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = model.num_models
    A = args.n_attributes
    lam = args.lambda_c_acc

    total_loss_meter = AverageMeter()
    y_acc_loss_meters = [AverageMeter() for _ in range(K)]
    c_acc_loss_meters = [AverageMeter() for _ in range(K)]
    c_div_loss_meters = [AverageMeter() for _ in range(K)]
    y_acc_meters      = [AverageMeter() for _ in range(K)]
    c_acc_meters      = [AverageMeter() for _ in range(K)]
    branch_total_loss_meters = [AverageMeter() for _ in range(K)]
    ensemble_y_acc_meter = AverageMeter()

    model.train(is_training)

    for batch_idx, batch in enumerate(loader):
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)

        if batch_idx == 0:
            print_memory_usage("Before Zero Grad")
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        _prev_training = model.training
        model.eval()
        with torch.no_grad():
            y_preds_all, c_preds_all = model(inputs_var)
            C_anchors = [torch.cat(c_list, dim=1) for c_list in c_preds_all]
        model.train(_prev_training)

        y_losses, c_losses, d_losses = [], [], []
        for i in range(K):
            Li_y = y_criterion(y_preds_all[i], y_labels_var)
            y_losses.append(Li_y)

            Li_c = sum(
                c_criterion[a](C_anchors[i][:, a], c_labels_var[:, a]) for a in range(A)
            ) / A
            c_losses.append(Li_c)

            sim_sum = torch.zeros((), device=x.device)
            for j in range(K):
                if j == i:
                    continue
                sim_sum = sim_sum + q_cosine(C_anchors[i], C_anchors[j].detach())
            sim_avg = sim_sum / max(K - 1, 1)
            Li_d = 1.0 - (sim_avg ** 2)

            d_losses.append(Li_d)

            task_i = Li_y + lam * Li_c
            branch_total_loss_meters[i].update((task_i - model.alpha * Li_d).item(), inputs_var.size(0))
            y_acc_loss_meters[i].update(Li_y.item(), inputs_var.size(0))
            c_acc_loss_meters[i].update(Li_c.item(), inputs_var.size(0))
            c_div_loss_meters[i].update(Li_d.item(), inputs_var.size(0))

            acc_y_i = accuracy(y_preds_all[i], y_labels_var, topk=(1,))[0].item()
            y_acc_meters[i].update(acc_y_i, inputs_var.size(0))
            c_sig = torch.sigmoid(C_anchors[i])
            acc_c_i = binary_accuracy(c_sig, c_labels_var).item()
            c_acc_meters[i].update(acc_c_i, inputs_var.size(0))

        i_y = torch.argmax(torch.stack(y_losses)).item()
        i_c = torch.argmax(torch.stack(c_losses)).item()
        winners = sorted(set([i_y, i_c]))
        others  = [i for i in range(K) if i not in winners]

        if not is_training:
            max_y = max(y_losses); max_c = max(c_losses)
            total_d = sum(d_losses) / K
            total_loss = max_y + lam * (max_c - model.alpha * total_d)
            total_loss_meter.update(total_loss.item(), inputs_var.size(0))

            avg_pred = torch.mean(torch.stack(y_preds_all, dim=0), dim=0)
            ens_acc  = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
            ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))
            continue

        total = 0.0
        for wi in winners:
            feats = model._reconstruct_forward_with_adapters(
                inputs_var, model.shared_backbone, model.adapter_sets[wi], use_checkpoint=True
            )
            head = model.final_heads[wi]
            Cw = torch.cat([fc(feats) for fc in head['concept_head']], dim=1)
            yw = head['classifier'](Cw)

            Ly = y_criterion(yw, y_labels_var) if wi == i_y else 0.0
            Lc = (sum(c_criterion[a](Cw[:, a], c_labels_var[:, a]) for a in range(A)) / A) if wi == i_c else 0.0

            sim_sum = 0.0
            for j in range(K):
                if j == wi: continue
                sim_sum += q_cosine(Cw, C_anchors[j].detach())
            Ld_w = 1.0 - (sim_sum / max(K - 1, 1)) ** 2

            total = total + Ly + lam * (Lc - model.alpha * Ld_w)

        if batch_idx == 0:
            print_memory_usage("After Forward")

        total.backward()
        if batch_idx == 0:
            print_memory_usage("After Backward")

        for i in others:
            feats_i = model._reconstruct_forward_with_adapters(
                inputs_var, model.shared_backbone, model.adapter_sets[i], use_checkpoint=True
            )
            head_i = model.final_heads[i]
            Ci = torch.cat([fc(feats_i) for fc in head_i['concept_head']], dim=1)

            sim_sum_i = torch.zeros((), device=x.device)
            for j in range(K):
                if j == i:
                    continue
                sim_sum_i = sim_sum_i + q_cosine(Ci, C_anchors[j].detach())
            sim_avg_i = sim_sum_i / max(K - 1, 1)
            Ld_i = 1.0 - (sim_avg_i ** 2)
            eps_c = 0.05
            if eps_c > 0:
                Lc_i_small = sum(
                    c_criterion[a](Ci[:, a], c_labels_var[:, a]) for a in range(A)
                ) / A
            else:
                Lc_i_small = 0.0

            loss_i = lam * (- model.alpha * Ld_i) + lam * eps_c * Lc_i_small
            loss_i.backward()

        with torch.no_grad():
            grad_sum, count = 0.0, 0
            for head in model.final_heads:
                for fc_module in head['concept_head']:
                    for p in fc_module.parameters():
                        if p.grad is not None:
                            grad_sum += p.grad.abs().mean(); count += 1
            if count > 0:
                avg_grad = grad_sum / count
                model.alpha.data.copy_(alpha_from_avg_grad(avg_grad, args))

        optimizer.step()
        if batch_idx == 0:
            print_memory_usage("After Optimizer Step")

        max_y = max(y_losses); max_c = max(c_losses)
        total_d = sum(d_losses) / K
        total_loss = max_y + lam * (max_c - model.alpha * total_d)
        total_loss_meter.update(total_loss.item(), inputs_var.size(0))

        avg_pred = torch.mean(torch.stack(y_preds_all, dim=0), dim=0)
        ens_acc  = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))

    detailed_meters = {
        "y_acc_loss": y_acc_loss_meters, 
        "c_acc_loss": c_acc_loss_meters, 
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters, 
        "c_acc": c_acc_meters, 
        "branch_total_loss": branch_total_loss_meters
    }
    return total_loss_meter, ensemble_y_acc_meter, detailed_meters

def run_epoch_partial_ensemble(model, optimizer, loader, y_criterion, c_criterion, args, is_training):
    num_models    = model.num_models
    n_attributes  = args.n_attributes

    total_loss_meter = AverageMeter()
    y_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_div_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_acc_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_meters = [AverageMeter() for _ in range(num_models)]
    branch_total_loss_meters = [AverageMeter() for _ in range(num_models)]
    ensemble_y_acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    for batch_idx, batch in enumerate(loader):
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)
        
        if batch_idx == 0:
            print_memory_usage("Before Zero Grad")

        if is_training:
            optimizer.zero_grad()

        all_y_preds, all_c_preds = model(inputs_var)
        if batch_idx == 0:
            print_memory_usage("After Forward")

        y_losses, c_losses, d_losses = [], [], []
        all_branch_losses = []
        all_task_losses = [] 
        concatenated = []
        for i in range(num_models):
            concatenated.append(torch.cat(all_c_preds[i], dim=1))

        for i in range(num_models):
            loss_y = y_criterion(all_y_preds[i], y_labels_var)
            y_losses.append(loss_y)

            loss_c = sum(
                c_criterion[attr_idx](
                    all_c_preds[i][attr_idx].flatten(),
                    c_labels_var[:, attr_idx]
                )
                for attr_idx in range(n_attributes)
            ) / n_attributes
            c_losses.append(loss_c)

            sim = sum(q_cosine(concatenated[i], concatenated[j].detach())
                for j in range(num_models) if j != i
            )
            loss_d = 1.0 - sim / (num_models - 1)
            d_losses.append(loss_d)

            task_loss = loss_y + args.lambda_c_acc * loss_c
            branch_total_loss = task_loss - model.alpha * loss_d

            all_task_losses.append(task_loss)
            all_branch_losses.append(branch_total_loss)

            branch_total_loss_meters[i].update(branch_total_loss.item(), inputs_var.size(0))
            y_acc_loss_meters[i].update(loss_y.item(), inputs_var.size(0))
            c_acc_loss_meters[i].update(loss_c.item(), inputs_var.size(0))
            c_div_loss_meters[i].update(loss_d.item(), inputs_var.size(0))

            acc_y_i = accuracy(all_y_preds[i], y_labels_var, topk=(1,))[0].item()
            y_acc_meters[i].update(acc_y_i, inputs_var.size(0))

            c_sig = torch.sigmoid(concatenated[i])
            acc_c_i = binary_accuracy(c_sig, c_labels_var).item()
            c_acc_meters[i].update(acc_c_i, inputs_var.size(0))

        max_y = max(y_losses)
        max_c = max(c_losses)
        total_d = sum(d_losses) / num_models
        
        total_loss = max_y + args.lambda_c_acc * (max_c - model.alpha * total_d)
        
        if is_training:
            total_loss.backward()
            with torch.no_grad():
                grad_sum, count = 0.0, 0
                for branch in model.branches:
                    for fc_module in branch['concept_head']:
                        for p in fc_module.parameters():
                            if p.grad is not None:
                                grad_sum += p.grad.abs().mean()
                                count += 1
                if count > 0:
                    avg_grad = grad_sum / count
                    model.alpha.data.copy_(alpha_from_avg_grad(avg_grad, args))
            optimizer.step()

        total_loss_meter.update(total_loss.item(), inputs_var.size(0))

        avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0)
        ens_acc  = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))

    detailed_meters = {
        "y_acc_loss": y_acc_loss_meters,
        "c_acc_loss": c_acc_loss_meters,
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters,
        "c_acc": c_acc_meters,
        "branch_total_loss": branch_total_loss_meters
    }
    return total_loss_meter, ensemble_y_acc_meter, detailed_meters

def run_ensembleDivEns(model, optimizer, loader, y_criterion, c_criterion, args, is_training):
    num_models = model.num_models
    n_attributes = args.n_attributes

    total_loss_meter = AverageMeter()
    y_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_div_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_acc_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_meters = [AverageMeter() for _ in range(num_models)]
    branch_total_loss_meters = [AverageMeter() for _ in range(num_models)]
    ensemble_y_acc_meter = AverageMeter()

    model.train() if is_training else model.eval()

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            if batch_idx == 0:
                print_memory_usage("Before Forward")
            inputs_var   = batch['img'].cuda(non_blocking=True)
            y_labels_var = batch['class_label'].cuda(non_blocking=True)
            c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)

            if is_training: 
                optimizer.zero_grad()

            all_y_preds, all_c_preds_list = model(inputs_var)
            if batch_idx == 0:
                print_memory_usage("After Forward")

            y_losses, c_losses, d_losses = [], [], []
            concatenated_concepts = torch.cat(all_c_preds_list, dim=1)

            for i in range(num_models):
                loss_y = y_criterion(all_y_preds[i], y_labels_var)
                y_losses.append(loss_y)
                
                loss_c = sum(c_criterion[attr_idx](all_c_preds_list[attr_idx].flatten(), c_labels_var[:, attr_idx]) for attr_idx in range(n_attributes)) / n_attributes
                c_losses.append(loss_c)

                sim_y = sum(q_cosine(all_y_preds[i], all_y_preds[j].detach()) for j in range(num_models) if j != i)
                loss_d = 1.0 - sim_y / (num_models - 1) if num_models > 1 else torch.tensor(0.0).to(args.device)
                d_losses.append(loss_d)
                
                branch_total_loss_meters[i].update((loss_y + args.lambda_c_acc * loss_c - model.alpha * loss_d).item(), inputs_var.size(0))
                y_acc_loss_meters[i].update(loss_y.item(), inputs_var.size(0))
                c_acc_loss_meters[i].update(loss_c.item(), inputs_var.size(0))
                c_div_loss_meters[i].update(loss_d.item(), inputs_var.size(0))
                y_acc_meters[i].update(accuracy(all_y_preds[i], y_labels_var)[0].item(), inputs_var.size(0))
                c_acc_meters[i].update(binary_accuracy(torch.sigmoid(concatenated_concepts), c_labels_var).item(), inputs_var.size(0))

            max_y, max_c, total_d = max(y_losses), max(c_losses), sum(d_losses) / num_models
            total_loss = max_y - model.alpha * total_d + args.lambda_c_acc * max_c

            if is_training:
                total_loss.backward()
                with torch.no_grad():
                    grad_sum, count = 0.0, 0
                    for branch in model.branches_c_to_y:
                        for p in branch.model_c_to_y.parameters():
                            if p.grad is not None:
                                grad_sum += p.grad.abs().mean()
                                count += 1
                    if count > 0:
                        avg_grad = grad_sum / count
                        model.alpha.data.copy_(alpha_from_avg_grad(avg_grad, args))
                        

                optimizer.step()

            total_loss_meter.update(total_loss.item(), inputs_var.size(0))
            avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0)
            ensemble_y_acc_meter.update(accuracy(avg_pred, y_labels_var)[0].item(), inputs_var.size(0))

    detailed_meters = {"y_acc_loss": y_acc_loss_meters, "c_acc_loss": c_acc_loss_meters, "c_div_loss": c_div_loss_meters, "y_acc": y_acc_meters, "c_acc": c_acc_meters, "branch_total_loss": branch_total_loss_meters}
    return total_loss_meter, ensemble_y_acc_meter, detailed_meters

def run_epoch_independent(models, optimizers, loader, y_criterion, c_criterion, args, is_training):

    num_models = len(models)
    n_attributes = args.n_attributes

    total_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_div_loss_meters = [AverageMeter() for _ in range(num_models)] 
    y_acc_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_meters = [AverageMeter() for _ in range(num_models)]
    ensemble_y_acc_meter = AverageMeter()

    for model in models:
        model.train() if is_training else model.eval()

    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print_memory_usage("Before Forward")
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)

        all_y_preds = []
        all_c_pred_lists = []
        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for model in models:
                y_pred, c_pred_list = model(inputs_var, is_training)
                all_y_preds.append(y_pred)
                all_c_pred_lists.append(c_pred_list)

        concatenated_concepts = [torch.cat(c_list, dim=1) for c_list in all_c_pred_lists]
        if batch_idx == 0:
            print_memory_usage("After Forward")


        for i in range(num_models):
            if is_training:
                optimizers[i].zero_grad()

            y_pred = all_y_preds[i]
            c_pred_list = all_c_pred_lists[i]

            loss_y = y_criterion(y_pred, y_labels_var)
            loss_c = sum(
                c_criterion[attr_idx](c_pred_list[attr_idx].flatten(), c_labels_var[:, attr_idx]) 
                for attr_idx in range(n_attributes)
            ) / n_attributes

            total_loss = loss_y + args.lambda_c_acc * loss_c

            if num_models > 1:
                sim = sum(
                    q_cosine(concatenated_concepts[i], concatenated_concepts[j].detach())
                    for j in range(num_models) if j != i
                )
                loss_d = 1.0 - sim / (num_models - 1)
            else:
                loss_d = torch.tensor(0.0).to(inputs_var.device)
            
            c_div_loss_meters[i].update(loss_d.item(), inputs_var.size(0))

            if is_training:
                total_loss.backward()
                optimizers[i].step()

            total_loss_meters[i].update(total_loss.item(), inputs_var.size(0))
            y_loss_meters[i].update(loss_y.item(), inputs_var.size(0))
            c_loss_meters[i].update(loss_c.item(), inputs_var.size(0))
            y_acc_meters[i].update(accuracy(y_pred, y_labels_var, topk=(1,))[0].item(), inputs_var.size(0))
            c_acc_meters[i].update(binary_accuracy(torch.sigmoid(concatenated_concepts[i]), c_labels_var).item(), inputs_var.size(0))

        avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0)
        ens_acc = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))

    avg_total_loss_meter = AverageMeter()
    avg_total_loss_meter.update(np.mean([m.avg for m in total_loss_meters if m.count > 0]))

    detailed_meters = {
        "branch_total_loss": total_loss_meters,
        "y_acc_loss": y_loss_meters,
        "c_acc_loss": c_loss_meters,
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters,
        "c_acc": c_acc_meters,
    }
    
    return avg_total_loss_meter, ensemble_y_acc_meter, detailed_meters
    
def run_epoch_independent_seq(models, optimizers, loader, y_criterion, c_criterion, args, is_training):

    device = torch.device("cuda")

    num_models = len(models)
    n_attributes = args.n_attributes

    total_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_loss_meters     = [AverageMeter() for _ in range(num_models)]
    c_loss_meters     = [AverageMeter() for _ in range(num_models)]
    c_div_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_acc_meters      = [AverageMeter() for _ in range(num_models)]
    c_acc_meters      = [AverageMeter() for _ in range(num_models)]
    ensemble_y_acc_meter = AverageMeter()

    for model in models:
        model.train() if is_training else model.eval()

    for batch_idx, batch in enumerate(loader):
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)
        bs = inputs_var.size(0)

        all_y_preds = []
        all_c_pred_lists = []
        concatenated_concepts = []

        with torch.no_grad():
            for i, model in enumerate(models):
                for j, m in enumerate(models):
                    m.to(device if j == i else "cpu")

                y_pred_i, c_pred_list_i = model(inputs_var, is_training=False)
                all_y_preds.append(y_pred_i.detach())
                all_c_pred_lists.append([t.detach() for t in c_pred_list_i])
                concatenated_concepts.append(torch.cat(c_pred_list_i, dim=1).detach())

                models[i].to("cpu")
                torch.cuda.empty_cache()

        for i in range(num_models):
            if is_training:
                models[i].to(device)
                optimizers[i].zero_grad(set_to_none=True)

                y_pred, c_pred_list = models[i](inputs_var, is_training=True)

                loss_y = y_criterion(y_pred, y_labels_var)

                loss_c = sum(
                    c_criterion[attr_idx](
                        c_pred_list[attr_idx].flatten(),
                        c_labels_var[:, attr_idx]
                    )
                    for attr_idx in range(n_attributes)
                ) / n_attributes

                total_loss = loss_y + args.lambda_c_acc * loss_c

                total_loss.backward()
                optimizers[i].step()

                total_loss_meters[i].update(total_loss.item(), bs)
                y_loss_meters[i].update(loss_y.item(), bs)
                c_loss_meters[i].update(loss_c.item(), bs)

                y_acc_meters[i].update(accuracy(all_y_preds[i], y_labels_var, topk=(1,))[0].item(), bs)
                c_acc_meters[i].update(binary_accuracy(torch.sigmoid(concatenated_concepts[i]), c_labels_var).item(), bs)

                if num_models > 1:
                    sim = sum(
                        q_cosine(concatenated_concepts[i], concatenated_concepts[j])
                        for j in range(num_models) if j != i
                    )
                    loss_d = 1.0 - sim / (num_models - 1)
                else:
                    loss_d = torch.tensor(0.0, device=inputs_var.device)
                c_div_loss_meters[i].update(loss_d.item(), bs)

                models[i].to("cpu")
                torch.cuda.empty_cache()

            else:
                loss_y = y_criterion(all_y_preds[i], y_labels_var)
                loss_c = sum(
                    c_criterion[attr_idx](
                        all_c_pred_lists[i][attr_idx].flatten(),
                        c_labels_var[:, attr_idx]
                    )
                    for attr_idx in range(n_attributes)
                ) / n_attributes
                total_loss = loss_y + args.lambda_c_acc * loss_c

                total_loss_meters[i].update(total_loss.item(), bs)
                y_loss_meters[i].update(loss_y.item(), bs)
                c_loss_meters[i].update(loss_c.item(), bs)

                y_acc_meters[i].update(accuracy(all_y_preds[i], y_labels_var, topk=(1,))[0].item(), bs)
                c_acc_meters[i].update(binary_accuracy(torch.sigmoid(concatenated_concepts[i]), c_labels_var).item(), bs)

                if num_models > 1:
                    sim = sum(
                        q_cosine(concatenated_concepts[i], concatenated_concepts[j])
                        for j in range(num_models) if j != i
                    )
                    loss_d = 1.0 - sim / (num_models - 1)
                else:
                    loss_d = torch.tensor(0.0, device=inputs_var.device)
                c_div_loss_meters[i].update(loss_d.item(), bs)

        avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0)
        ens_acc = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, bs)

    avg_total_loss_meter = AverageMeter()
    avg_total_loss_meter.update(np.mean([m.avg for m in total_loss_meters if m.count > 0]))

    detailed_meters = {
        "branch_total_loss": total_loss_meters,
        "y_acc_loss": y_loss_meters,
        "c_acc_loss": c_loss_meters,
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters,
        "c_acc": c_acc_meters,
    }
    return avg_total_loss_meter, ensemble_y_acc_meter, detailed_meters


def run_epoch_sconvpar(model, optimizer, loader, y_criterion, c_criterion, args, is_training):
    num_models    = model.num_models
    n_attributes  = args.n_attributes
    lambda_c      = getattr(args, 'lambda_c_acc', 1.0)

    total_loss_meter = AverageMeter()
    y_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_div_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_acc_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_meters = [AverageMeter() for _ in range(num_models)]
    branch_total_loss_meters = [AverageMeter() for _ in range(num_models)]
    ensemble_y_acc_meter = AverageMeter()

    model.train() if is_training else model.eval()

    for batch_idx, batch in enumerate(loader):
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        if batch_idx == 0:
            print_memory_usage("Before Forward")
        use_ckpt = getattr(args, "gradient_checkpointing", False)

        all_y_preds, all_c_preds = model(inputs_var, use_checkpoint=use_ckpt)
        if batch_idx == 0:
            print_memory_usage("After Forward")

        y_losses, c_losses, d_losses = [], [], []
        concatenated = [torch.cat(all_c_preds[i], dim=1) for i in range(num_models)]

        for i in range(num_models):
            loss_y = y_criterion(all_y_preds[i], y_labels_var)
            y_losses.append(loss_y)

            loss_c = sum(
                c_criterion[a](all_c_preds[i][a].flatten(), c_labels_var[:, a])
                for a in range(n_attributes)
            ) / n_attributes
            c_losses.append(loss_c)

            sim = sum(q_cosine(concatenated[i], concatenated[j].detach())
                      for j in range(num_models) if j != i)
            loss_d = 1.0 - sim / (num_models - 1)
            d_losses.append(loss_d)

            branch_total = loss_y + lambda_c * loss_c - model.alpha * loss_d
            branch_total_loss_meters[i].update(branch_total.item(), inputs_var.size(0))
            y_acc_loss_meters[i].update(loss_y.item(), inputs_var.size(0))
            c_acc_loss_meters[i].update(loss_c.item(), inputs_var.size(0))
            c_div_loss_meters[i].update(loss_d.item(), inputs_var.size(0))

            acc_y_i = accuracy(all_y_preds[i], y_labels_var, topk=(1,))[0].item()
            y_acc_meters[i].update(acc_y_i, inputs_var.size(0))
            c_sig = torch.sigmoid(concatenated[i])
            acc_c_i = binary_accuracy(c_sig, c_labels_var).item()
            c_acc_meters[i].update(acc_c_i, inputs_var.size(0))

        max_y = torch.stack(y_losses).max()
        max_c = torch.stack(c_losses).max()
        avg_d = torch.stack(d_losses).mean()
        total_loss = max_y + lambda_c * (max_c - model.alpha * avg_d)

        if is_training:
            total_loss.backward()

            if batch_idx == 0:
                print_memory_usage("After Backward")

            with torch.no_grad():
                grad_sum, count = 0.0, 0
                for branch in model.final_heads:
                    for fc_module in branch['concept_head']:
                        for p in fc_module.parameters():
                            if p.grad is not None:
                                grad_sum += p.grad.abs().mean()
                                count += 1
                if count > 0:
                    avg_grad = grad_sum / count
                    model.alpha.data.copy_(alpha_from_avg_grad(avg_grad, args))

            optimizer.step()

            if batch_idx == 0:
                print_memory_usage("After Optimizer Step")

        total_loss_meter.update(total_loss.item(), inputs_var.size(0))
        
        avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0)
        ens_acc  = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))

    detailed_meters = {
        "y_acc_loss": y_acc_loss_meters,
        "c_acc_loss": c_acc_loss_meters,
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters,
        "c_acc": c_acc_meters,
        "branch_total_loss": branch_total_loss_meters
    }
    return total_loss_meter, ensemble_y_acc_meter, detailed_meters

def run_epoch_lora_par(model, optimizer, loader, y_criterion, c_criterion, args, is_training):
    num_models    = model.num_models
    n_attributes  = args.n_attributes
    lambda_c      = getattr(args, 'lambda_c_acc', 1.0)

    total_loss_meter = AverageMeter()
    y_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_div_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_acc_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_meters = [AverageMeter() for _ in range(num_models)]
    branch_total_loss_meters = [AverageMeter() for _ in range(num_models)]
    ensemble_y_acc_meter = AverageMeter()

    model.train() if is_training else model.eval()

    for batch_idx, batch in enumerate(loader):
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        if batch_idx == 0:
            print_memory_usage("Before Forward")
        all_y_preds, all_c_preds = model(inputs_var)

        if batch_idx == 0:
            print_memory_usage("After Forward")

        y_losses, c_losses, d_losses = [], [], []
        concatenated = [torch.cat(all_c_preds[i], dim=1) for i in range(num_models)]

        for i in range(num_models):
            loss_y = y_criterion(all_y_preds[i], y_labels_var)
            y_losses.append(loss_y)

            loss_c = sum(
                c_criterion[a](all_c_preds[i][a].flatten(), c_labels_var[:, a])
                for a in range(n_attributes)
            ) / n_attributes
            c_losses.append(loss_c)

            sim = sum(q_cosine(concatenated[i], concatenated[j].detach())
                      for j in range(num_models) if j != i)
            loss_d = 1.0 - sim / (num_models - 1)
            d_losses.append(loss_d)

            branch_total = loss_y + lambda_c * loss_c - model.alpha * loss_d
            branch_total_loss_meters[i].update(branch_total.item(), inputs_var.size(0))
            y_acc_loss_meters[i].update(loss_y.item(), inputs_var.size(0))
            c_acc_loss_meters[i].update(loss_c.item(), inputs_var.size(0))
            c_div_loss_meters[i].update(loss_d.item(), inputs_var.size(0))

            acc_y_i = accuracy(all_y_preds[i], y_labels_var, topk=(1,))[0].item()
            y_acc_meters[i].update(acc_y_i, inputs_var.size(0))
            c_sig = torch.sigmoid(concatenated[i])
            acc_c_i = binary_accuracy(c_sig, c_labels_var).item()
            c_acc_meters[i].update(acc_c_i, inputs_var.size(0))

        max_y = torch.stack(y_losses).max()
        max_c = torch.stack(c_losses).max()
        avg_d = torch.stack(d_losses).mean()
        total_loss = max_y + lambda_c * (max_c - model.alpha * avg_d)

        if is_training:
            total_loss.backward()

            if batch_idx == 0:
                print_memory_usage("After Backward")

            with torch.no_grad():
                grad_sum, count = 0.0, 0
                for branch in model.final_heads:
                    for fc_module in branch['concept_head']:
                        for p in fc_module.parameters():
                            if p.grad is not None:
                                grad_sum += p.grad.abs().mean()
                                count += 1
                if count > 0:
                    avg_grad = grad_sum / count
                    model.alpha.data.copy_(alpha_from_avg_grad(avg_grad, args))

            # Restore requires_grad for all LoRA adapter params before optimizer.step().
            # set_adapter() during forward disables grad on non-active adapters;
            # without this fix, weight_decay would only apply to the last active adapter.
            for name, param in model.lora_models.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

            optimizer.step()

            if batch_idx == 0:
                print_memory_usage("After Optimizer Step")

        total_loss_meter.update(total_loss.item(), inputs_var.size(0))

        avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0)
        ens_acc  = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))

    detailed_meters = {
        "y_acc_loss": y_acc_loss_meters,
        "c_acc_loss": c_acc_loss_meters,
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters,
        "c_acc": c_acc_meters,
        "branch_total_loss": branch_total_loss_meters
    }
    return total_loss_meter, ensemble_y_acc_meter, detailed_meters

def run_epoch_revolver(model, optimizer, loader, y_criterion, c_criterion, args, is_training):
    num_models    = model.n_models
    n_attributes  = args.n_attributes
    lambda_c      = getattr(args, 'lambda_c_acc', 1.0)

    total_loss_meter = AverageMeter()
    y_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_div_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_acc_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_meters = [AverageMeter() for _ in range(num_models)]
    branch_total_loss_meters = [AverageMeter() for _ in range(num_models)]
    ensemble_y_acc_meter = AverageMeter()

    model.train(is_training)

    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        if batch_idx == 0:
            print_memory_usage("Before Forward")
        use_ckpt = getattr(args, "gradient_checkpointing", False)

        all_y_preds, all_c_preds = model(inputs_var, use_checkpoint=use_ckpt)
        if batch_idx == 0:
            print_memory_usage("After Forward")

        y_losses, c_losses, d_losses = [], [], []
        concatenated = [torch.cat(all_c_preds[i], dim=1) for i in range(num_models)]

        for i in range(num_models):
            loss_y = y_criterion(all_y_preds[i], y_labels_var)
            y_losses.append(loss_y)

            loss_c = sum(
                c_criterion[a](all_c_preds[i][a].flatten(), c_labels_var[:, a])
                for a in range(n_attributes)
            ) / n_attributes
            c_losses.append(loss_c)

            sim = sum(q_cosine(concatenated[i], concatenated[j].detach())
                      for j in range(num_models) if j != i)
            loss_d = 1.0 - sim / (num_models - 1)
            d_losses.append(loss_d)

            branch_total = loss_y + lambda_c * loss_c - model.alpha * loss_d
            branch_total_loss_meters[i].update(branch_total.item(), inputs_var.size(0))
            y_acc_loss_meters[i].update(loss_y.item(), inputs_var.size(0))
            c_acc_loss_meters[i].update(loss_c.item(), inputs_var.size(0))
            c_div_loss_meters[i].update(loss_d.item(), inputs_var.size(0))

            acc_y_i = accuracy(all_y_preds[i], y_labels_var, topk=(1,))[0].item()
            y_acc_meters[i].update(acc_y_i, inputs_var.size(0))
            c_sig = torch.sigmoid(concatenated[i])
            acc_c_i = binary_accuracy(c_sig, c_labels_var).item()
            c_acc_meters[i].update(acc_c_i, inputs_var.size(0))

        max_y = torch.stack(y_losses).max()
        max_c = torch.stack(c_losses).max()
        avg_d = torch.stack(d_losses).mean()
        total_loss = max_y + lambda_c * (max_c - model.alpha * avg_d)

        if is_training:
            total_loss.backward()

            if batch_idx == 0:
                print_memory_usage("After Backward")

            with torch.no_grad():
                grad_sum, count = 0.0, 0
                for branch in model.heads:
                    for fc_module in branch["concept_head"]:
                        for p in fc_module.parameters():
                            if p.grad is not None:
                                grad_sum += p.grad.abs().mean()
                                count += 1
                if count > 0:
                    avg_grad = grad_sum / count
                    model.alpha.data.copy_(alpha_from_avg_grad(avg_grad, args))

            optimizer.step()

            if batch_idx == 0:
                print_memory_usage("After Optimizer Step")

        total_loss_meter.update(total_loss.item(), inputs_var.size(0))
        
        avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0)
        ens_acc  = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))

    detailed_meters = {
        "y_acc_loss": y_acc_loss_meters,
        "c_acc_loss": c_acc_loss_meters,
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters,
        "c_acc": c_acc_meters,
        "branch_total_loss": branch_total_loss_meters
    }
    return total_loss_meter, ensemble_y_acc_meter, detailed_meters

def run_epoch_dropout(model, optimizer, loader, y_criterion, c_criterion, args, is_training):
    num_models    = model.n_models
    n_attributes  = args.n_attributes
    total_loss_meter = AverageMeter()
    y_loss_meters = [AverageMeter() for _ in range(num_models)]
    c_loss_meters = [AverageMeter() for _ in range(num_models)]
    y_acc_meters = [AverageMeter() for _ in range(num_models)]
    c_acc_meters = [AverageMeter() for _ in range(num_models)]

    c_div_loss_meters = [AverageMeter() for _ in range(num_models)]
    branch_total_loss_meters = [AverageMeter() for _ in range(num_models)]

    ensemble_y_acc_meter = AverageMeter()

    model.train() if is_training else model.eval()

    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        inputs_var   = batch['img'].cuda(non_blocking=True)
        y_labels_var = batch['class_label'].cuda(non_blocking=True)
        c_labels_var = batch['attribute_label'].float().cuda(non_blocking=True)

        if batch_idx == 0:
            print_memory_usage("Before Forward")
        assert model.masking != is_training

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            all_y_preds, all_c_preds = model(inputs_var, use_checkpoint=False)

        if batch_idx == 0:
            print_memory_usage("After Forward")

        y_losses, c_losses = [], []
        concatenated = [torch.cat(c_pred, dim=1) for c_pred in all_c_preds] \
            if not is_training else torch.cat(all_c_preds, dim=1)

        if not is_training:
            for i in range(num_models):
                loss_y = y_criterion(all_y_preds[i], y_labels_var)
                y_losses.append(loss_y)

                loss_c = sum(
                    c_criterion[a](all_c_preds[i][a].flatten(), c_labels_var[:, a])
                    for a in range(n_attributes)
                ) / n_attributes
                c_losses.append(loss_c)

                branch_total = loss_y + args.lambda_c_acc * loss_c
                branch_total_loss_meters[i].update(branch_total.item(), inputs_var.size(0))

                y_loss_meters[i].update(loss_y.item(), inputs_var.size(0))
                c_loss_meters[i].update(loss_c.item(), inputs_var.size(0))

                acc_y_i = accuracy(all_y_preds[i], y_labels_var, topk=(1,))[0].item()
                y_acc_meters[i].update(acc_y_i, inputs_var.size(0))
                c_sig = torch.sigmoid(concatenated[i])
                acc_c_i = binary_accuracy(c_sig, c_labels_var).item()
                c_acc_meters[i].update(acc_c_i, inputs_var.size(0))

            max_y = torch.stack(y_losses).max()
            max_c = torch.stack(c_losses).max()
            total_loss = max_y + args.lambda_c_acc * max_c
        else:
            optimizer.zero_grad()

            loss_y = y_criterion(all_y_preds, y_labels_var)
            loss_c = sum(
                c_criterion[a](all_c_preds[a].flatten(), c_labels_var[:, a])
                for a in range(n_attributes)
            ) / n_attributes

            total_loss = loss_y + args.lambda_c_acc * loss_c

            total_loss.backward()

            if batch_idx == 0:
                print_memory_usage("After Backward")

            optimizer.step()

            if batch_idx == 0:
                print_memory_usage("After Optimizer Step")

        total_loss_meter.update(total_loss.item(), inputs_var.size(0))  
        
        avg_pred = torch.mean(torch.stack(all_y_preds, dim=0), dim=0) if not is_training else all_y_preds
        ens_acc  = accuracy(avg_pred, y_labels_var, topk=(1,))[0].item()
        ensemble_y_acc_meter.update(ens_acc, inputs_var.size(0))

    detailed_meters = {
        "branch_total_loss": branch_total_loss_meters,
        "y_acc_loss": y_loss_meters,
        "c_acc_loss": c_loss_meters,
        "c_div_loss": c_div_loss_meters,
        "y_acc": y_acc_meters,
        "c_acc": c_acc_meters,
    }
    return total_loss_meter, ensemble_y_acc_meter, detailed_meters


def train(args):
    init_wandb(args)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.flush()
    
    if args.dataname == 'CUB':
        N_CLASSES = CUB_N_CLASSES
        setup_CUB_dataset(os.path.join(BASE_DIR, args.data_dir))
        train = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')

        train_loader = load_CUB_data(train, batch_size=args.batch_size, is_training=True)
        val_loader = load_CUB_data(val, batch_size=args.batch_size, is_training=False)
        test_loader = load_CUB_data(test, batch_size=args.batch_size, is_training=False)

    elif args.dataname == 'Awa2':
        N_CLASSES = AWA2_N_CLASSES
        setup_awa2_dataset(os.path.join(BASE_DIR, args.data_dir))
        train = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')

        train_loader = load_awa2_data(train, batch_size=args.batch_size, is_training=True)
        val_loader = load_awa2_data(val, batch_size=args.batch_size, is_training=False)
        test_loader = load_awa2_data(test, batch_size=args.batch_size, is_training=False)

    elif args.dataname == 'CelebA':
        N_CLASSES = CELEBA_N_CLASSES
        setup_celeba_dataset(os.path.join(BASE_DIR, args.data_dir))
        train = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')

        train_loader = load_celeba_data(train, batch_size=args.batch_size, is_training=True)
        val_loader = load_celeba_data(val, batch_size=args.batch_size, is_training=False)
        test_loader = load_celeba_data(test, batch_size=args.batch_size, is_training=False)

    elif args.dataname == 'cifar10':
        N_CLASSES = CIFAR10_N_CLASSES
        setup_cifar10_dataset(os.path.join(BASE_DIR, args.data_dir), os.path.join(BASE_DIR, args.data_dir))
        train = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')

        train_loader = load_cifar10_data(train, batch_size=args.batch_size, is_training=True)
        val_loader = load_cifar10_data(val, batch_size=args.batch_size, is_training=False)
        test_loader = load_cifar10_data(test, batch_size=args.batch_size, is_training=False)
    
    elif args.dataname == 'HAM10000':
        N_CLASSES = HAM10000_N_CLASSES
        ham_out_dir = os.path.join(BASE_DIR, args.data_dir)
        setup_ham10000_dataset(root_dir=BASE_DIR, out_dir=ham_out_dir, topk=20, num_workers=8)

        # HAM10000 concepts are selected from per-class candidates and de-duplicated across classes,
        # so the final union size may be != 7 * topk. Override args.n_attributes to match the data.
        concepts_txt = os.path.join(ham_out_dir, "concepts_used.txt")
        if os.path.exists(concepts_txt):
            with open(concepts_txt, "r") as f:
                concepts = [ln.strip() for ln in f if ln.strip()]
            if concepts and (len(concepts) != args.n_attributes):
                print(f"[warn] Overriding n_attributes {args.n_attributes} -> {len(concepts)} based on {concepts_txt}")
                args.n_attributes = len(concepts)
        train = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')

        train_loader = load_ham10000_data(train, batch_size=args.batch_size, is_training=True, encoder=args.encoder)
        val_loader = load_ham10000_data(val, batch_size=args.batch_size, is_training=False, encoder=args.encoder)
        test_loader = load_ham10000_data(test, batch_size=args.batch_size, is_training=False, encoder=args.encoder)
    
    def split_test_loader_into_three(test_loader, seed = 42):
        dataset = test_loader.dataset
        n = len(dataset)

        sizes = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]

        g = torch.Generator()
        g.manual_seed(seed)
        subsets = torch.utils.data.random_split(dataset, sizes, generator=g)

        num_workers = getattr(test_loader, "num_workers", 0)
        common_kwargs = {
            "batch_size": test_loader.batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": getattr(test_loader, "pin_memory", False),
            "collate_fn": getattr(test_loader, "collate_fn", None),
            "drop_last": False,
            "persistent_workers": getattr(test_loader, "persistent_workers", False) if num_workers > 0 else False,
        }
        if num_workers > 0 and hasattr(test_loader, "prefetch_factor"):
            common_kwargs["prefetch_factor"] = getattr(test_loader, "prefetch_factor")

        common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}

        loaders = [DataLoader(subset, **common_kwargs) for subset in subsets]
        return loaders

    test_loaders = split_test_loader_into_three(test_loader)

  
    model = None
    models = None
    optimizer = None
    optimizers = None
    start_epoch = 1

    if args.exp == 'X2C':
        model_class = EnsembleXtoCtoY
        epoch_runner_func = run_epoch_ensemble_e2e
    elif args.exp == 'DivEns':
        model_class = EnsembleWithSharedXtoC
        epoch_runner_func = run_ensembleDivEns
    elif args.exp == 'random':
        models = [
            SingleE2EBranch(
                n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                expand_dim=args.expand_dim, encoder=args.encoder
            ).cuda() for _ in range(args.num_models)
        ]
        def _make_random_optimizer(m):
            base_lr = float(args.lr)
            wd = float(getattr(args, "weight_decay", 0.0))
            tiny = float(getattr(args, "tiny_lr", 1.0))

            backbone_params = []
            if hasattr(m, "model_x_to_c") and hasattr(m.model_x_to_c, "model"):
                backbone_params = [p for p in m.model_x_to_c.model.parameters() if p.requires_grad]
            backbone_ids = {id(p) for p in backbone_params}
            other_params = [p for p in m.parameters() if p.requires_grad and id(p) not in backbone_ids]

            param_groups = []
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": base_lr * tiny, "weight_decay": wd})
            if other_params:
                param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": wd})
            return torch.optim.Adam(param_groups)

        optimizers = [_make_random_optimizer(m) for m in models]
        schedulers = [torch.optim.lr_scheduler.StepLR(o, step_size=args.scheduler_step, gamma=0.9) for o in optimizers]
        trainable_params = list(models[0].model_x_to_c.parameters()) + \
                                list(models[0].model_c_to_y.parameters()) 
        print(f"Trainable parameters: {args.num_models*sum(p.numel() for p in trainable_params)}")
        epoch_runner_func = run_epoch_independent
    
    elif args.exp == 'PartialX2C':
        model = EnsembleWithPartialSharing(
            num_models=args.num_models,
            n_class_attr=args.n_class_attr,
            pretrained=args.pretrained,
            num_classes=N_CLASSES,
            use_aux=args.use_aux,
            n_attributes=args.n_attributes,
            expand_dim=args.expand_dim,
            encoder=args.encoder,
            split_point=args.split_point,
            bottleneck_dim=args.bottleneck_dim,
            tiny_lr=args.tiny_lr
        ).cuda()

        trunk_params = [p for n, p in model.named_parameters() if 'trunk' in n and p.requires_grad]
        branch_params = [p for n, p in model.named_parameters() if 'trunk' not in n and p.requires_grad]
        optimizer_grouped_parameters = [
            {
                "params": branch_params,
                "lr": args.lr,
            },
            {
                "params": trunk_params,
                "lr": args.lr * args.tiny_lr,
            },
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.9)

        epoch_runner_func = run_epoch_partial_ensemble
    elif args.exp == 'ConvParX2C':
        model = ConvParEnsemble(
            num_models=args.num_models,
            num_classes=N_CLASSES,
            n_attributes=args.n_attributes,
            encoder=args.encoder,
            bottleneck_dim=args.bottleneck_dim,
            expand_dim=args.expand_dim
        )
        model = model.cuda()

        trainable_params = list(model.adapter_sets.parameters()) + \
                        list(model.final_heads.parameters()) + \
                        [model.alpha]

        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.9)
        epoch_runner_func = run_epoch_convpar

    elif args.exp == 'ConvAda':
        if args.encoder != 'resnet18':
            raise NotImplementedError("exp='ConvAda' only supports encoder='resnet18'.")
        model = SConvParEnsemble(
            num_models=args.num_models,
            num_classes=N_CLASSES,
            n_attributes=args.n_attributes,
            encoder=args.encoder,
            bottleneck_dim=args.bottleneck_dim,
            expand_dim=args.expand_dim,
            share_mask=args.share_mask,
            use_pretrained=True
        ).cuda()

        shared_params = list(model.shared_adapters.parameters())
        branch_params = []
        for adapters_i in model.branch_adapters:
            branch_params += list(adapters_i.parameters())
        head_params = list(model.final_heads.parameters())
        trainable_params = shared_params + branch_params + head_params + [model.alpha]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.9)
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
        epoch_runner_func = run_epoch_sconvpar
    elif args.exp == 'Lora':
        if args.encoder not in ('vit', 'medical_vit'):
            raise NotImplementedError("exp='Lora' only supports encoder in {'vit','medical_vit'}.")
        if set(args.share_mask)>set(['0','1']) or len(args.share_mask)!=12:
            raise ValueError("share_mask should be a string of 0 and 1 with length 12 for vit small and base model, e.g., '110011001100'")
        if args.share_mask=='0'*len(args.share_mask):
            print('none of the lora layers are shared')
            if args.gradient_checkpointing:
                model = LoraEnsemble_gc(
                    num_models=args.num_models,
                    num_classes=N_CLASSES,
                    n_attributes=args.n_attributes,
                    encoder=args.encoder,
                    expand_dim=args.expand_dim,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    lora_block_mask=args.share_mask,
                )
            else:
                model = LoraEnsemble(
                    num_models=args.num_models,
                    num_classes=N_CLASSES,
                    n_attributes=args.n_attributes,
                    encoder=args.encoder,
                    expand_dim=args.expand_dim,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    lora_block_mask=args.share_mask,
                )
        elif args.share_mask=='1'*len(args.share_mask):
            print('all of the lora layers are shared')
            model = LoraEnsembleshared(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=args.n_attributes,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_block_mask=args.share_mask,
            )
            if args.gradient_checkpointing:
                warnings.warn("gradient checkpointing is not implemented for this model")
        else:
            print('some of the lora layers are shared')
            model = LoraEnsemblePshared(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=args.n_attributes,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_block_mask=args.share_mask,
            )
            if args.gradient_checkpointing:
                warnings.warn("gradient checkpointing is not implemented for this model")
        model = model.cuda()

        # Fix: PEFT's set_adapter() disables requires_grad for non-active adapters.
        # Since only lora_0 is active after __init__, lora_1..lora_N-1 would be
        # excluded from the optimizer. Re-enable requires_grad for ALL LoRA params.
        for name, param in model.lora_models.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # model.alpha is already in model.parameters(); adding it twice breaks optimizer construction.
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=args.lr, 
            weight_decay=args.weight_decay 
        )
        

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.9)

        if args.resume:
            if not args.checkpoint_path:
                raise ValueError("--checkpoint_path must be provided when --resume is set.")
            model_path = os.path.join(args.checkpoint_path, 'checkpoint_model.pth')
            optimizer_path = os.path.join(args.checkpoint_path, 'checkpoint_optimizer.pt')
            scheduler_path = os.path.join(args.checkpoint_path, 'checkpoint_scheduler.pt')
            misc_path = os.path.join(args.checkpoint_path, 'checkpoint_misc.pt')

            if os.path.exists(model_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model.load_state_dict(torch.load(model_path, map_location=device))
                
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
                
                checkpoint_misc = torch.load(misc_path, map_location="cpu")

                start_epoch = checkpoint_misc['epoch'] + 1 
                
                print(f"resume successfully, training from epoch {start_epoch}.")
            else:
                print("didn't find checkpoint, training from scratch.")
        epoch_runner_func = run_epoch_lora_par

    elif args.exp == 'Revolver':
        baseline_model = torchvision.models.resnet18().cuda()
        n_features = baseline_model.fc.in_features
        baseline_model.fc = torch.nn.Identity()

        model = RevolverEnsemble(
            n_models=args.num_models,
            n_classes=N_CLASSES,
            n_attributes=args.n_attributes,
            n_features=n_features,
            baseline=baseline_model,
            targets=[
                "layer3",
                "layer4"
            ],
            encoder=args.encoder
        ).cuda()

        cylinder_params = []
        for cylinder in model.ensemble.cylinders:
            for module in cylinder.values():
                cylinder_params.extend(module.parameters())

        head_params = list(model.heads.parameters())

        trainable_params = cylinder_params + head_params + [model.alpha]

        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.9)

        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

        epoch_runner_func = run_epoch_revolver

    elif args.exp == 'Dropout':
        if not args.passthrough:
            if args.encoder == "resnet18":
                baseline_model = torchvision.models.resnet18().cuda()
                n_features = baseline_model.fc.in_features
                baseline_model.fc = torch.nn.Identity()

                targets = [
                    "layer1",
                    "layer2",
                    "layer3",
                    # "layer4"
                ]
            elif args.encoder == "vit":
                baseline_model = AutoModel.from_pretrained("timm/vit_small_patch16_224.augreg_in21k_ft_in1k")
                n_features = baseline_model.timm_model.embed_dim
                num_layers = len(baseline_model.timm_model.blocks)
                targets = [
                    f"timm_model.blocks.{i}" for i in range(max(0, num_layers - 1))
                ]
            elif args.encoder == "medical_vit":
                # MEDICAL_VIT_MODEL_NAME is a CLIP-style checkpoint; use the vision tower to get a
                # (last_hidden_state, ...) output like other HF vision encoders.
                baseline_model = CLIPVisionModel.from_pretrained(MEDICAL_VIT_MODEL_NAME)
                n_features = baseline_model.config.hidden_size
                num_layers = len(baseline_model.vision_model.encoder.layers)
                targets = [
                    f"vision_model.encoder.layers.{i}" for i in range(max(0, num_layers - 1))
                ]
            else:
                raise ValueError(f"Unknown encoder specified for Dropout exp: {args.encoder}")
        else:
            baseline_model = SingleE2EBranch(
                n_class_attr=args.n_class_attr,
                pretrained=args.pretrained,
                freeze=args.freeze,
                num_classes=N_CLASSES,
                use_aux=args.use_aux,
                n_attributes=args.n_attributes,
                expand_dim=args.expand_dim,
                encoder=args.encoder,
            ).cuda()

            if args.encoder == "resnet18":
                targets = [
                    "model_x_to_c.model.layer1",
                    "model_x_to_c.model.layer2",
                    "model_x_to_c.model.layer3",
                ]

            elif args.encoder == "vit":
                num_layers = len(baseline_model.model_x_to_c.model.timm_model.blocks)
                targets = [
                    f"model_x_to_c.model.timm_model.blocks.{i}" for i in range(max(0, num_layers - 1))
                ]
            elif args.encoder == "medical_vit":
                num_layers = len(baseline_model.model_x_to_c.model.vision_model.encoder.layers)
                targets = [
                    f"model_x_to_c.model.vision_model.encoder.layers.{i}" for i in range(max(0, num_layers - 1))
                ]
            else:
                raise ValueError(f"Unknown encoder specified for Dropout exp: {args.encoder}")

            n_features = 0

        model = DropoutEnsemble(
            n_models=args.num_models,
            n_classes=N_CLASSES,
            n_attributes=args.n_attributes,
            n_features=n_features,
            baseline=baseline_model,
            targets=targets,
            dropouts=args.dropout,
            encoder=args.encoder,
            passthrough=args.passthrough,
        ).cuda()

        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.9)

        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

        epoch_runner_func = run_epoch_dropout

    else:
        raise ValueError(f"Unknown exp: {args.exp}. Must be 'X2C', 'DivEns' or 'random'.")

    if args.exp in ['X2C', 'DivEns']:
        model = model_class(
            num_models=args.num_models,
            n_class_attr=args.n_class_attr,
            pretrained=args.pretrained,
            freeze=args.freeze,
            num_classes=N_CLASSES,
            use_aux=args.use_aux,
            n_attributes=args.n_attributes,
            expand_dim=args.expand_dim,
            encoder=args.encoder
        )
        model = model.cuda()
        if args.exp =='DivEns':
            trainable_params =list(model.model_x_to_c.parameters())\
                            + list(model.branches_c_to_y.parameters())\
                            + [model.alpha]
            print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

        base_lr = float(args.lr)
        wd = float(getattr(args, "weight_decay", 0.0))
        tiny = float(getattr(args, "tiny_lr", 1.0))

        # Apply a smaller LR to the underlying vision backbone if exposed as `model_x_to_c.model`.
        # This improves generalization for full fine-tuning, especially for ViT/CLIP encoders.
        backbone_params = []
        if hasattr(model, "model_x_to_c") and hasattr(model.model_x_to_c, "model"):
            backbone_params = [p for p in model.model_x_to_c.model.parameters() if p.requires_grad]
        elif hasattr(model, "branches"):
            for br in model.branches:
                if hasattr(br, "model_x_to_c") and hasattr(br.model_x_to_c, "model"):
                    backbone_params.extend([p for p in br.model_x_to_c.model.parameters() if p.requires_grad])

        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr * tiny, "weight_decay": wd})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": wd})
        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.9)        

    y_criterion = torch.nn.CrossEntropyLoss()
    c_criterion = [torch.nn.BCEWithLogitsLoss() for _ in range(args.n_attributes)]

    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    early_stop_metric = getattr(args, "early_stop_metric", "overall_val_loss")
    best_val_loss = np.inf
    best_val_acc = -1.0
    best_val_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs + 1):
        val_optimizer = None

        if args.exp == 'random':
            train_total_loss, train_ensemble_acc, train_detailed_meters = epoch_runner_func(
                models, optimizers, train_loader, y_criterion, c_criterion, args, is_training=True
            )
            with torch.no_grad():
                val_total_loss, val_ensemble_acc, val_detailed_meters = epoch_runner_func(
                    models, None, val_loader, y_criterion, c_criterion, args, is_training=False
                )
        else:
            train_total_loss, train_ensemble_acc, train_detailed_meters = epoch_runner_func(
                model, optimizer, train_loader, y_criterion, c_criterion, args, is_training=True
            )
            with torch.no_grad():
                val_total_loss, val_ensemble_acc, val_detailed_meters = epoch_runner_func(
                    model, val_optimizer, val_loader, y_criterion, c_criterion, args, is_training=False
                )
        if args.exp == 'random':
            for s in schedulers:
                s.step()
        else:
            scheduler.step()
            if epoch % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'Current lr: {current_lr}')
        test_fold_results = []
        with torch.no_grad():
            for t_idx, t_loader in enumerate(test_loaders):
                if args.exp == 'random':
                    t_loss, t_acc, t_meters = epoch_runner_func(
                        models, None, t_loader, y_criterion, c_criterion, args, is_training=False
                    )
                else:
                    t_loss, t_acc, t_meters = epoch_runner_func(
                        model, None, t_loader, y_criterion, c_criterion, args, is_training=False
                    )
                test_fold_results.append((t_loss, t_acc, t_meters))
                log_metrics_to_wandb(f"test_fold{t_idx+1}", epoch, t_loss, t_acc, t_meters, args)

        test_loss_vals = [r[0].avg for r in test_fold_results]
        test_acc_vals  = [r[1].avg for r in test_fold_results]
        test_loss_mean = mean(test_loss_vals)
        test_loss_std  = stdev(test_loss_vals) if len(test_loss_vals) > 1 else 0.0
        test_acc_mean  = mean(test_acc_vals)
        test_acc_std   = stdev(test_acc_vals) if len(test_acc_vals) > 1 else 0.0

        try:
            wandb.log({
                "test_mean/loss": test_loss_mean,
                "test_std/loss":  test_loss_std,
                "test_mean/acc":  test_acc_mean,
                "test_std/acc":   test_acc_std,
            }, step=epoch)
        except Exception:
            pass

        log_metrics_to_wandb("train", epoch, train_total_loss, train_ensemble_acc, train_detailed_meters, args)
        log_metrics_to_wandb("val",   epoch, val_total_loss,   val_ensemble_acc,   val_detailed_meters,   args)

        if early_stop_metric == "overall_val_loss":
            val_score = val_total_loss.avg
        elif early_stop_metric == "sum_child_val_tloss":
            val_score = sum(m.avg for m in val_detailed_meters["branch_total_loss"])
        elif early_stop_metric == "sum_child_val_yplusc":
            val_score = sum(
                val_detailed_meters["y_acc_loss"][i].avg + val_detailed_meters["c_acc_loss"][i].avg
                for i in range(args.num_models)
            )
        elif early_stop_metric == "mean_child_val_miscls":
            # Misclassification loss on val (lower is better):
            #   miscls_i = 2 - (C-Acc_i + Y-Acc_i) where accuracies are fractions in [0,1].
            # In this repo logs, accuracies are tracked in percent, so divide by 100.
            if "c_acc" not in val_detailed_meters or "y_acc" not in val_detailed_meters:
                raise ValueError(
                    "early_stop_metric='mean_child_val_miscls' requires val_detailed_meters to contain "
                    "'c_acc' and 'y_acc' meters."
                )
            val_score = sum(
                2.0
                - (val_detailed_meters["c_acc"][i].avg / 100.0 + val_detailed_meters["y_acc"][i].avg / 100.0)
                for i in range(args.num_models)
            ) / float(args.num_models)
        else:
            raise ValueError(
                f"Unknown early_stop_metric={early_stop_metric!r}. "
                "Choose from: overall_val_loss, sum_child_val_tloss, sum_child_val_yplusc, mean_child_val_miscls."
            )

        log_str  = (f"\nEpoch [{epoch}/{args.epochs}]:\n"
                    f"  [Overall]  Train Loss: {train_total_loss.avg:.4f}, Train Ensemble Acc: {train_ensemble_acc.avg:.2f}%\n"
                    f"  [Overall]  Val   Loss: {val_total_loss.avg:.4f}, Val   Ensemble Acc:   {val_ensemble_acc.avg:.2f}%\n")
        for f_idx, (t_loss, t_acc, _) in enumerate(test_fold_results, start=1):
            log_str += (f"  [Overall]  Test{f_idx} Loss: {t_loss.avg:.4f}, Test{f_idx} Ensemble Acc: {t_acc.avg:.2f}%\n")
        log_str += (f"  [Overall]  Test (3-fold) Loss: {test_loss_mean:.4f} ± {test_loss_std:.4f}, "
                    f"Ensemble Acc: {test_acc_mean:.2f}% ± {test_acc_std:.2f}%\n")

        if args.exp != 'random':
            log_str += (f"  [Info]     Best Val Epoch: {best_val_epoch}, Best Val Loss: {best_val_loss:.4f} "
                        f"(metric={early_stop_metric}), Current Val Score: {val_score:.4f}, "
                        f"Model Alpha: {model.alpha.item():.4f}\n")
        else:
            log_str += (f"  [Info]     Best Val Epoch: {best_val_epoch}, Best Val Loss: {best_val_loss:.4f} "
                        f"(metric={early_stop_metric}), Current Val Score: {val_score:.4f}\n")

        for i in range(args.num_models):
            log_str += (f"  [Child {i}] Train | T-Loss: {train_detailed_meters['branch_total_loss'][i].avg:.4f} "
                        f"Y-Loss: {train_detailed_meters['y_acc_loss'][i].avg:.4f} "
                        f"C-Loss: {train_detailed_meters['c_acc_loss'][i].avg:.4f} "
                        f"C-Div: {train_detailed_meters['c_div_loss'][i].avg:.4f} | "
                        f"C-Acc: {train_detailed_meters['c_acc'][i].avg:.2f}% "
                        f"Y-Acc: {train_detailed_meters['y_acc'][i].avg:.2f}%\n")
            log_str += (f"  [Child {i}] Val   | T-Loss: {val_detailed_meters['branch_total_loss'][i].avg:.4f} "
                        f"Y-Loss: {val_detailed_meters['y_acc_loss'][i].avg:.4f} "
                        f"C-Loss: {val_detailed_meters['c_acc_loss'][i].avg:.4f} "
                        f"C-Div: {val_detailed_meters['c_div_loss'][i].avg:.4f} | "
                        f"C-Acc: {val_detailed_meters['c_acc'][i].avg:.2f}% "
                        f"Y-Acc: {val_detailed_meters['y_acc'][i].avg:.2f}%\n") 
            for f_idx, (_, _, meters) in enumerate(test_fold_results, start=1):
                log_str += (f"  [Child {i}] Test{f_idx} | T-Loss: {meters['branch_total_loss'][i].avg:.4f} "
                            f"Y-Loss: {meters['y_acc_loss'][i].avg:.4f} "
                            f"C-Loss: {meters['c_acc_loss'][i].avg:.4f} "
                            f"C-Div: {meters['c_div_loss'][i].avg:.4f} | "
                            f"C-Acc: {meters['c_acc'][i].avg:.2f}% "
                            f"Y-Acc: {meters['y_acc'][i].avg:.2f}%\n")

        logger.write(log_str)
        logger.flush()
        if val_score < best_val_loss:
            best_val_loss = val_score
            best_val_epoch = epoch
            logger.write('New best model saved at epoch %d\n' % epoch)
            if args.exp == 'random':
                torch.save([m.state_dict() for m in models], os.path.join(args.log_dir, 'best_model.pth'))
            elif args.exp == 'Lora':
                torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
                for i in range(args.num_models):
                    model.save_adapter_cbm(i, args.log_dir)
                if isinstance(model, (LoraEnsemblePshared, LoraEnsembleshared)):
                    model.save_shared_adapter(args.log_dir)
            else:
                torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))

        if args.exp == 'random':
            torch.save([m.state_dict() for m in models], os.path.join(args.log_dir, 'checkpoint_model.pth'))
            torch.save([o.state_dict() for o in optimizers], os.path.join(args.log_dir, "checkpoint_optimizers.pt"))
            torch.save([s.state_dict() for s in schedulers], os.path.join(args.log_dir, "checkpoint_schedulers.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'checkpoint_model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.log_dir,"checkpoint_optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(args.log_dir,"checkpoint_scheduler.pt"))
        torch.save({"epoch": epoch}, os.path.join(args.log_dir,"checkpoint_misc.pt"))

        if args.save_traj and epoch%10==1:
            if args.exp == 'random':
                torch.save([m.state_dict() for m in models], os.path.join(args.log_dir, f'epoch{epoch}_model.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'epoch{epoch}_model.pth'))




        if epoch - best_val_epoch >= args.patience:
            logger.write("Early stopping.\n")
            print(
                f"Early stopping because monitored metric ({early_stop_metric}) hasn't improved "
                f"for {args.patience} epochs."
            )
            break



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', type=str, default='Lora', choices=['DivEns', 'random', 'ConvAda', 'Lora', 'Dropout'], help='Name of experiment to run.')
    parser.add_argument('-seed', required=True, type=int, help='Numpy and torch seed.')
    parser.add_argument('-log_dir', default='log', help='where the trained model is saved')
    parser.add_argument('-batch_size', type=int, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-lr', type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-patience', type=int, default=30, help='patience for early stopping')
    parser.add_argument(
        '--early_stop_metric', '-early_stop_metric',
        type=str,
        default='overall_val_loss',
        choices=['overall_val_loss', 'sum_child_val_tloss', 'sum_child_val_yplusc', 'mean_child_val_miscls'],
        help=("Metric for early stopping / best_model.pth selection (lower is better). "
              "overall_val_loss = val_total_loss.avg; "
              "sum_child_val_tloss = sum_i child_i Val T-Loss; "
              "sum_child_val_yplusc = sum_i (child_i Val Y-Loss + Val C-Loss); "
              "mean_child_val_miscls = mean_i (2 - (child_i Val C-Acc + child_i Val Y-Acc)) with accuracies in [0,1].")
    )
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-dropout', type=float, default=1e-1, help="dropout rate for the dropout ensemble")
    parser.add_argument('-passthrough', action="store_true", help="directly use a random init instance for dropout")
    parser.add_argument('-freeze', action='store_true', help='whether to freeze the encoder backbone')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    # Backwards-compat: some older scripts pass -use_attr; training uses n_attributes regardless.
    parser.add_argument('-use_attr', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-n_attributes', type=int)
    parser.add_argument('-expand_dim', type=int, default=0,
                        help='dimension of hidden layer (if we want to increase model capacity)')
    parser.add_argument('-n_class_attr', type=int, default=2,
                        help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', help='directory to the training data')
    parser.add_argument('-scheduler_step', type=int, default=10,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-num_models', type=int, default=5, help='Number of models in ensemble')
    parser.add_argument('-lambda_c_acc', type=float, default=0.5,
                        help='Weight for concept accuracy loss in end-to-end ensemble model')
    parser.add_argument('-dataname', type=str, default='CUB',
                        choices=['CUB', 'cifar10', 'Awa2', 'CelebA', 'HAM10000'],
                        help='Name of the dataset to use for training')
    parser.add_argument('-encoder', type=str, default='vit',
                        choices=['resnet18', 'vit', 'medical_vit'],
                        help='Type of encoder to use for the model')
    parser.add_argument('-split_point', type=str, default='layer4',
                        choices=['layer1', 'layer2', 'layer3', 'layer4'],
                        help="The backbone layer after which to split into branches for ResNet18.")
    parser.add_argument('-bottleneck_dim', type=int, default=128,
                        help="Dimension of the bottleneck in the adapter.")
    parser.add_argument('-tiny_lr', type=float, default=0.1,
                        help="Factor for the tiny learning rate for the backbone.")
    parser.add_argument('--wandb_tags', nargs='*', default=[],
                        help='Extra Weights & Biases tags (space-separated).')
    parser.add_argument('--wandb_notes', type=str, default='',
                        help='Free-form notes to attach to the W&B run.')
    parser.add_argument('-share_mask', type=str, default='00000')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--alpha_min', type=float, default=0.5,
                        help='Minimum value for model.alpha (default matches current behavior).')
    parser.add_argument('--alpha_max', type=float, default=1.0,
                        help='Maximum value for model.alpha.')
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="turn on gradient checkpointing or not")
    parser.add_argument("--resume", action="store_true", help="resuming from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--save_traj",action="store_true", help="logging midway ckpts, please use only when you want to draw a traj")


    args = parser.parse_args()
    args.three_class = (args.n_class_attr == 3)
    return args

if __name__ == '__main__':
    args = parse_arguments()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
