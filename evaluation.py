import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from transformers import AutoModel

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
import csv
import itertools
from torch.utils.data import Dataset, DataLoader
import shap

import warnings
from collections import defaultdict
from datasets.CUB_dataset import load_CUB_data
from datasets.cifar10_dataset import load_cifar10_data, setup_cifar10_dataset

from config import BASE_DIR, CUB_N_CLASSES, AWA2_N_CLASSES, CIFAR10_N_CLASSES, CELEBA_N_CLASSES
from template_model import MLP
from models import (
    SingleE2EBranch, 
    EnsembleXtoCtoY, 
    EnsembleWithSharedXtoC, 
    EnsembleWithPartialSharing,
    ConvParEnsemble,
    SConvParEnsemble,
    LoraEnsemble,
    LoraEnsemblePshared,
    LoraEnsembleshared,
    RevolverEnsemble,
    DropoutEnsemble
)
from datasets.Awa2_dataset import load_awa2_data
from datasets.CelebA_dataset import load_celeba_data
import copy
from torch.utils.checkpoint import checkpoint as _ckpt
from collections import OrderedDict
from functools import partial
from torch.utils.data import random_split, DataLoader
from train import (
    run_epoch_ensemble_e2e, 
    run_epoch_sconvpar, 
    run_epoch_lora_par, 
    run_ensembleDivEns,
    run_epoch_convpar,
    run_epoch_independent,
    run_epoch_partial_ensemble,
    run_epoch_revolver,
    run_epoch_dropout
)
# warnings.filterwarnings("ignore")

class Logger(object):
    """Logs results to a file and the console, and flushes to view instant updates."""
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            log_dir = os.path.dirname(fpath)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def write(self, msg):
        self.console.write(msg + '\n')
        if self.file is not None:
            self.file.write(msg + '\n')

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()

# Single Branch Extraction for PartialX2C
def load_child_models_Partial(ensemble_or_ckpt, args, device=None, eval_mode=True, strict=False):
    class _XtoC(nn.Module):
        def __init__(self, trunk, deep_layers, concept_head, adapter=None):
            super().__init__()
            self.trunk = copy.deepcopy(trunk)
            self.deep_layers = copy.deepcopy(deep_layers)
            self.concept_head = copy.deepcopy(concept_head)  # ModuleList of FC heads
            self.adapter = copy.deepcopy(adapter) if adapter is not None else None

        def forward(self, x):
            feats = self.trunk(x)
            feats = self.deep_layers(feats)
            if self.adapter is not None:
                feats = self.adapter(feats)
            pooled = F.adaptive_avg_pool2d(feats, (1, 1))
            flat = torch.flatten(pooled, 1)
            concept_preds = [fc(flat) for fc in self.concept_head]
            return concept_preds

    class _ChildModel(nn.Module):
        def __init__(self, trunk, deep_layers, concept_head, classifier, adapter=None):
            super().__init__()
            self.model_x_to_c = _XtoC(trunk, deep_layers, concept_head, adapter)
            self.model_c_to_y = copy.deepcopy(classifier)   # MLP: [B, n_attributes] -> logits

        def forward(self, x):
            concept_preds = self.model_x_to_c(x)  # list[Tensor]  Tensor
            if isinstance(concept_preds, (list, tuple)):
                concat = torch.cat(concept_preds, dim=1)    # [B, n_attributes]
            else:
                concat = concept_preds
            y_pred = self.model_c_to_y(concat)
            return y_pred, concept_preds

    def _unwrap_state_dict(sd):
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], (dict, OrderedDict)):
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], (dict, OrderedDict)):
            sd = sd["model"]
        if any(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd

    def _infer_num_models_from_keys(sd, fallback=1):
        idxs = []
        for k in sd.keys():
            if isinstance(k, str) and k.startswith("branches."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    idxs.append(int(parts[1]))
        return (max(idxs) + 1) if idxs else fallback

    def _ensure_model(obj):
        if isinstance(obj, nn.Module):
            return obj
        import torch
        if isinstance(obj, str):
            raw = torch.load(obj, map_location=device)
        elif isinstance(obj, (dict, OrderedDict)):
            raw = obj
        else:
            raise TypeError("ensemble_or_ckpt must be nn.Module / str / Dict.")
        sd = _unwrap_state_dict(raw)
        inferred_num = _infer_num_models_from_keys(sd, fallback=getattr(args, "num_models", 1))
        model = EnsembleWithPartialSharing(
            num_models=inferred_num,
            num_classes=N_CLASSES,
            n_attributes=CONCEPT_DIM,
            encoder=getattr(args, "encoder", "resnet18"),
            expand_dim=getattr(args, "expand_dim", 0),
            split_point=getattr(args, "split_point", "layer4"),
            bottleneck_dim=getattr(args, "bottleneck_dim", 64),
        )
        model.load_state_dict(sd, strict=strict)
        return model
    ensemble = _ensure_model(ensemble_or_ckpt)
    children = []
    for b in ensemble.branches:
        adapter = getattr(b, "adapter", None) if hasattr(b, "adapter") else None
        child = _ChildModel(
            trunk=ensemble.trunk,
            deep_layers=b["deep_layers"],
            concept_head=b["concept_head"],
            classifier=b["classifier"],
            adapter=adapter,
        )
        if device is not None:
            child.to(device)
        if eval_mode:
            child.eval()
        children.append(child)

    return children

# Single Branch Extraction for ConvAda
class _BackboneView(nn.Module):
    def __init__(self, stem, l1, l2, l3, l4):
        super().__init__()
        self.conv1   = stem['conv1']
        self.bn1     = stem['bn1']
        self.relu    = stem['relu']
        self.maxpool = stem['maxpool']
        self.layer1, self.layer2, self.layer3, self.layer4 = l1, l2, l3, l4
        self.aux_logits = False

    def forward(self, x): 
        raise NotImplementedError

def _collect_adapters_for_branch(ensemble, i, deepcopy_modules=True):
    ad = nn.ModuleDict()
    def _dc(m): return copy.deepcopy(m) if deepcopy_modules else m
    if hasattr(ensemble, 'shared_adapters'):
        for k, v in ensemble.shared_adapters.items():
            ad[k] = _dc(v)
    if hasattr(ensemble, 'branch_adapters'):
        branch = ensemble.branch_adapters[i]
        for k, v in branch.items():
            ad[k] = _dc(v)
    return ad

def _extract_from_ensemble_flex(ensemble, device, deepcopy_modules=True, share_backbone=False, use_checkpoint=True):
    n = int(ensemble.num_models)
    children = []

    base_view = _BackboneView(ensemble.stem, ensemble.layer1, ensemble.layer2, ensemble.layer3, ensemble.layer4)
    if share_backbone:
        backbones = [base_view] * n
    else:
        backbones = [copy.deepcopy(base_view) if deepcopy_modules else base_view for _ in range(n)]

    for i in range(n):
        adapters = _collect_adapters_for_branch(ensemble, i, deepcopy_modules=deepcopy_modules)
        heads    = ensemble.final_heads[i]
        concept  = copy.deepcopy(heads['concept_head']) if deepcopy_modules else heads['concept_head']
        clf      = copy.deepcopy(heads['classifier'])   if deepcopy_modules else heads['classifier']

        x_to_c = _XtoCWithAdapters(backbones[i], adapters, concept, use_checkpoint=use_checkpoint)
        child  = _ChildForSHAP(x_to_c, clf).to(device).eval()
        children.append(child)
    return children

def load_child_models_flex(ensemble_model_path, args, device, num_concepts, num_classes,
                              *, deepcopy_modules=True, share_backbone=False, use_checkpoint=True):
    obj = torch.load(ensemble_model_path, map_location=device)

    if isinstance(obj, nn.Module):
        ensemble = obj
        if hasattr(ensemble, 'shared_adapters') and hasattr(ensemble, 'branch_adapters'):
            return _extract_from_ensemble_flex(
                ensemble, device,
                deepcopy_modules=deepcopy_modules,
                share_backbone=share_backbone,
                use_checkpoint=use_checkpoint
            )
        else:
            raise AttributeError("Ensemble Structure cannot be identified")

    elif isinstance(obj, dict):
        num_models = getattr(args, "num_models", None)
        if num_models is None:
            idxs = []
            for k in obj.keys():
                if k.startswith("branch_adapters."):
                    try: idxs.append(int(k.split(".")[1]))
                    except: pass
            num_models = max(idxs) + 1 if idxs else 1

        def _has_prefix(sd, prefix):
            return any(k.startswith(prefix) for k in sd.keys())

        def _infer_share_mask(sd):
            segs = []
            segs.append('1' if _has_prefix(sd, "shared_adapters.conv1.") else '0')
            for li in (1,2,3,4):
                bits = []
                for bi in (0,1):
                    key_u = f"shared_adapters.layer{li}_{bi}_conv1."
                    key_b = f"branch_adapters.0.layer{li}_{bi}_conv1."
                    if _has_prefix(sd, key_u):
                        bits.append('1')
                    elif _has_prefix(sd, key_b):
                        bits.append('0')
                    else:
                        bits.append('1')
                segs.append(''.join(bits))
            return f"{segs[0]}|{segs[1]}|{segs[2]}|{segs[3]}|{segs[4]}"

        share_mask = getattr(args, "share_mask", None) or _infer_share_mask(obj)

        encoder        = getattr(args, "encoder", "resnet18")
        expand_dim     = getattr(args, "expand_dim", 0)
        bottleneck_dim = getattr(args, "bottleneck_dim", 64)

        ensemble = SConvParEnsemble(
            num_models=num_models,
            num_classes=num_classes,
            n_attributes=num_concepts,
            encoder=encoder,
            bottleneck_dim=bottleneck_dim,
            expand_dim=expand_dim,
            share_mask=share_mask,
            use_pretrained=True,
        )
        ensemble.load_state_dict(obj, strict=False) 
        return _extract_from_ensemble_flex(
            ensemble, device,
            deepcopy_modules=deepcopy_modules,
            share_backbone=share_backbone,
            use_checkpoint=use_checkpoint
        )
    else:
        raise TypeError("Cannot load model files.")
    
def load_child_model_revolver(ensemble: RevolverEnsemble):
    children = []

    for i in range(ensemble.n_models):
        curr_child = _ChildForSHAPExtended(
            backbone=ensemble.ensemble if ensemble.encoder != "vit" else \
                lambda x: ensemble.ensemble(x).last_hidden_state[:, 0],
            x_to_c=ensemble.heads[i]["concept_head"],
            c_to_y=ensemble.heads[i]["class_head"],
            original=ensemble
        )

        children.append(curr_child)

    return children

def load_child_model_dropout(ensemble: DropoutEnsemble):
    children = []

    ensemble.eval()

    for i in range(ensemble.n_models):
        if not ensemble.passthrough:
            curr_child = _ChildForSHAPExtended(
                backbone=ensemble.ensemble if ensemble.encoder != "vit" else \
                    lambda x: ensemble.ensemble(x).last_hidden_state[:, 0],
                x_to_c=lambda raw_outputs : [fc(raw_outputs) for fc in ensemble.heads["concept_head"]],
                c_to_y=ensemble.heads["class_head"],
                original=ensemble
            )
        else:
            curr_child = _ChildForSHAP(
                x_to_c=ensemble.ensemble.baseline.model_x_to_c,
                c_to_y=ensemble.ensemble.baseline.model_c_to_y
            )

        children.append(curr_child)

    return children


# Single Branch Extraction for ConvParX2C
class _XtoCWithAdapters(nn.Module):
    def __init__(self, backbone: nn.Module, adapters: nn.ModuleDict, concept_head: nn.ModuleList, use_checkpoint=True):
        super().__init__()
        self.backbone = backbone
        self.adapters = adapters
        self.concept_head = concept_head
        self.use_checkpoint = use_checkpoint
        self.aux_logits = False

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.affine:
                    if m.weight is not None: m.weight.requires_grad_(False)
                    if m.bias   is not None: m.bias.requires_grad_(False)

    @staticmethod
    def _k(name: str) -> str:
        return name.replace('.', '_')

    def _forward_with_adapters(self, x: torch.Tensor) -> torch.Tensor:
        bb, ad = self.backbone, self.adapters
        out = bb.conv1(x) + ad[self._k('conv1')](x)
        out = bb.relu(bb.bn1(out))
        out = bb.maxpool(out)
        for li in range(1, 5):
            layer = getattr(bb, f'layer{li}')
            for bi, blk in enumerate(layer):
                def _blk(inp, _blk=blk, _li=li, _bi=bi):
                    identity = inp
                    if _blk.downsample is not None:
                        identity = _blk.downsample(identity)
                    res = _blk.conv1(inp) + ad[self._k(f'layer{_li}.{_bi}.conv1')](inp)
                    res = _blk.relu(_blk.bn1(res))
                    res = _blk.conv2(res) + ad[self._k(f'layer{_li}.{_bi}.conv2')](res)
                    res = _blk.bn2(res)
                    return _blk.relu(res + identity)
                out = _ckpt(_blk, out) if self.use_checkpoint else _blk(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor):
        self.backbone.eval()
        feats = self._forward_with_adapters(x)
        return [fc(feats) for fc in self.concept_head]

class _ChildForSHAP(nn.Module):
    def __init__(self, x_to_c: nn.Module, c_to_y: nn.Module):
        super().__init__()
        self.model_x_to_c = x_to_c
        self.model_c_to_y = c_to_y

    def forward(self, x, is_training: bool):
        c_list = self.model_x_to_c(x)
        y = self.model_c_to_y(torch.cat(c_list, dim=1))
        return y, c_list

class _ChildForSHAPExtended(nn.Module):
    def __init__(self, backbone, x_to_c: nn.Module, c_to_y: nn.Module, original: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.original = original

        self.model_x_to_c_func = x_to_c
        self.model_c_to_y = c_to_y
    
    def model_x_to_c(self, x):
        x_prime = self.backbone(x)
        c_list = self.model_x_to_c_func(x_prime)
        return c_list

    def forward(self, x, is_training: bool):
        x_prime = self.backbone(x)
        c_list = self.model_x_to_c(x_prime)
        y = self.model_c_to_y(torch.cat(c_list, dim=1))
        return y, c_list


def _extract_from_ensemble(ensemble, device, deepcopy_modules=True, share_backbone=False, use_checkpoint=True):
    n = ensemble.num_models
    children = []

    if share_backbone:
        backbones = [ensemble.shared_backbone] * n
    else:
        backbones = [
            (copy.deepcopy(ensemble.shared_backbone) if deepcopy_modules else ensemble.shared_backbone)
            for _ in range(n)
        ]

    for i in range(n):
        adapters = copy.deepcopy(ensemble.adapter_sets[i]) if deepcopy_modules else ensemble.adapter_sets[i]
        heads    = ensemble.final_heads[i]
        concept  = copy.deepcopy(heads['concept_head']) if deepcopy_modules else heads['concept_head']
        clf      = copy.deepcopy(heads['classifier'])   if deepcopy_modules else heads['classifier']

        x_to_c = _XtoCWithAdapters(backbones[i], adapters, concept, use_checkpoint=use_checkpoint)
        child = _ChildForSHAP(x_to_c, clf).to(device).eval()
        children.append(child)
    return children


def load_child_models_Adapter(ensemble_model_path, args, device, num_concepts, num_classes,
                          *, deepcopy_modules=True, share_backbone=False, use_checkpoint=True):
    obj = torch.load(ensemble_model_path, map_location=device)

    if isinstance(obj, nn.Module):
        ensemble = obj
    elif isinstance(obj, dict):
        encoder = getattr(args, "encoder", "resnet18")
        expand_dim = getattr(args, "expand_dim", 0)
        bottleneck_dim = getattr(args, "bottleneck_dim", 64)
        num_models = getattr(args, "num_models", None)
        if num_models is None:
            idxs = []
            for k in obj.keys():
                if k.startswith("adapter_sets."):
                    try: idxs.append(int(k.split(".")[1]))
                    except: pass
            num_models = max(idxs) + 1 if idxs else 1

        ensemble = ConvParEnsemble(
            num_models=num_models, num_classes=num_classes,
            n_attributes=num_concepts, encoder=encoder,
            bottleneck_dim=bottleneck_dim, expand_dim=expand_dim
        )
        ensemble.load_state_dict(obj, strict=False)
    else:
        raise TypeError("Cannot load model files.")

    return _extract_from_ensemble(
        ensemble, device,
        deepcopy_modules=deepcopy_modules, share_backbone=share_backbone, use_checkpoint=use_checkpoint
    )

# Single Branch Extraction for X2C
def load_child_models_X2C(ensemble_model_path, args, device, num_concepts, num_classes):
    # Load the entire state_dict of the trained ensemble model
    ensemble_state_dict = torch.load(ensemble_model_path, map_location=device)
    
    child_models = []
    num_models = args.num_models
    print(f"Extracting {num_models} independent child models from {ensemble_model_path}...")

    # Loop through each model to extract its weights
    for i in range(num_models):
        # Create a new, empty, and structurally complete child model instance
        child_model = SingleE2EBranch(n_class_attr=args.n_class_attr, pretrained=True, freeze=False, num_classes=num_classes, use_aux=True, n_attributes=num_concepts, expand_dim=args.expand_dim, encoder=args.encoder)
        
        # Prepare an empty state_dict for this child model's weights
        child_state_dict = {}
        
        # Define the prefix for this child model's weights in the ensemble state_dict
        prefix = f'branches.{i}.'
        
        # Iterate over the ensemble's state_dict, extracting weights that match the prefix
        for key, value in ensemble_state_dict.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, '', 1)
                child_state_dict[new_key] = value
        
        if not child_state_dict:
            print(f"Warning: Did not find weights for child model {i}! Checking prefix '{prefix}'.")
            continue
            
        # Load the extracted weights into the new child model instance
        child_model.load_state_dict(child_state_dict, strict=False)
        
        # Finalize the model and add it to our list
        child_model.to(device)
        child_model.eval()
        child_models.append(child_model)
        print(f"Successfully extracted and loaded independent child model {i}.")

    return child_models

# Single Branch Extraction for DivEns
def load_child_models_DivEns(ensemble_model_path, num_models, device, args, num_concepts, num_classes):
    
    print(f"Loading ensemble model from {ensemble_model_path}...")
    ensemble_state_dict = torch.load(ensemble_model_path, map_location=device)
    
    child_models = []

    # Extract the state_dict for the shared X->C part
    model_x_to_c_state_dict = {}
    shared_prefix = 'model_x_to_c.'
    for key, value in ensemble_state_dict.items():
        if key.startswith(shared_prefix):
            # Remove the prefix to get the original key for the X->C model's layers
            new_key = key.replace(shared_prefix, '')
            model_x_to_c_state_dict[new_key] = value
    
    if not model_x_to_c_state_dict:
        raise ValueError("Did not find weights for the shared X->C model in the saved file! Please check the path and model structure.")
    
    # Loop to create each complete child model
    for i in range(num_models):
        # Create a new, structurally complete SingleE2EBranch instance
        child_model = SingleE2EBranch(n_class_attr=args.n_class_attr, pretrained=True, freeze=False, num_classes=num_classes, use_aux=True, n_attributes=num_concepts, expand_dim=args.expand_dim, encoder=args.encoder)
        
        # Load the shared X->C weights into it
        child_model.model_x_to_c.load_state_dict(model_x_to_c_state_dict, strict=False)
        
        # Extract the weights for the i-th independent C->Y branch
        branch_c_to_y_state_dict = {}
        branch_prefix = f'branches_c_to_y.{i}.model_c_to_y.'
        for key, value in ensemble_state_dict.items():
            if key.startswith(branch_prefix):
                new_key = key.replace(branch_prefix, '')
                branch_c_to_y_state_dict[new_key] = value
        
        if not branch_c_to_y_state_dict:
            print(f"Warning: Did not find weights for C->Y branch {i}! Please check the model prefix '{branch_prefix}'.")
            continue
        
        # Load the independent C->Y weights into it
        child_model.model_c_to_y.load_state_dict(branch_c_to_y_state_dict)
        
        # Finalize the model and add it to the list
        child_model.to(device)
        child_model.eval() # Set to evaluation mode
        child_models.append(child_model)
        print(f"Successfully reconstructed and loaded complete X->C->Y child model {i}.")

    return child_models

# Single Branch Extraction for random
def load_independent_models(model_path, args, device, num_concepts, num_classes):
    print(f"Loading list of independent models from: {model_path}")
    list_of_state_dicts = torch.load(model_path, map_location=device)

    if not isinstance(list_of_state_dicts, list):
        raise TypeError(
            f"Expected a list of state_dicts from {model_path}, but got "
            f"{type(list_of_state_dicts)}. This file was likely not saved "
            "from an 'Independent' training run."
        )

    child_models = []
    num_models_in_file = len(list_of_state_dicts)
    print(f"Found {num_models_in_file} model states in the file. Reconstructing...")

    for i, state_dict in enumerate(list_of_state_dicts):
        child_model = SingleE2EBranch(
            n_class_attr=args.n_class_attr, pretrained=False, freeze=False,
            num_classes=num_classes, use_aux=args.use_aux, n_attributes=num_concepts,
            expand_dim=args.expand_dim, encoder=args.encoder
        )
        child_model.load_state_dict(state_dict)
        child_model.to(device)
        child_model.eval()
        child_models.append(child_model)
        print(f"  Successfully loaded independent model {i}.")

    return child_models

# SHAP analysis function
def compute_feature_importance_shap(model, dataloader, args, device, sample_size=50, nsamples=50):
    model.eval()
    background_samples = []
    print(f"Aggregating data for SHAP background (target size: {sample_size})...")
    
    # Loop through the dataloader to collect enough samples
    for batch in dataloader:
        if args.dataname in ['cifar10', 'Awa2', 'CelebA', 'CUB']:
            features = batch['img']
        else:
            raise ValueError(f"Unsupported dataset name for SHAP analysis: {args.dataname}")
        
        background_samples.append(features.cpu()) # Collect on CPU to save GPU memory
        
        # Check if we have collected enough samples
        current_size = sum(t.size(0) for t in background_samples)
        if current_size >= sample_size:
            print(f"Collected {current_size} samples, which meets the target of {sample_size}.")
            break
    
    if not background_samples:
        raise ValueError("Dataloader is empty or failed to load data, cannot create a background dataset for SHAP.")
    
    background_features_full = torch.cat(background_samples, dim=0)
    background_features = background_features_full[:sample_size].to(device)
  
    with torch.no_grad():
        # Get the list of concept predictions from the encoder
        concept_predictions_list = model.model_x_to_c(background_features)
        # Concatenate them into a single tensor [batch_size, n_attributes]
        concept_background = torch.cat(concept_predictions_list, dim=1)

    # Convert to numpy for SHAP. This is the CORRECT background data.
    background_for_shap = concept_background[:sample_size].cpu().numpy()

    # This function should only represent the C->Y part of the model.
    c_to_y_model = model.model_c_to_y

    def model_forward(concept_vectors_numpy):
        c_tensor = torch.from_numpy(concept_vectors_numpy).float().to(device)
        with torch.no_grad():
            y_pred = c_to_y_model(c_tensor)
        return y_pred.cpu().numpy()

    # Use KernelExplainer
    explainer = shap.KernelExplainer(model_forward, background_for_shap)
    shap_values = explainer.shap_values(background_for_shap, nsamples=nsamples)
    
    # Process multi-class SHAP values to get a single importance vector
    if isinstance(shap_values, list):
        # shape: (num_classes, sample_size, num_features)
        arr = np.array([np.abs(s) for s in shap_values])  
        # shape: (num_features,)
        importance_vector = np.mean(arr, axis=(0, 1))  
    else: # for binary classification or other cases
        shap_abs = np.abs(shap_values)
        if shap_abs.ndim == 3:
            importance_vector = np.mean(shap_abs, axis=(0, 2))
        else:
            importance_vector = np.mean(shap_abs, axis=0)
    return importance_vector

def compute_feature_importance_shap_lora(model,i, dataloader, args, device, sample_size=50, nsamples=50):
    if model is LoraEnsemble:
        print('model is LoraEnsemble')
        model.lora_models.set_adapter(f'lora_{i}')
    elif model is LoraEnsemblePshared:
        print('model is LoraEnsemblePshared')
        model.lora_models.set_adapter([f'lora_{i}']+['lora_shared'])
    elif model is LoraEnsembleshared:
        print('model is LoraEnsembleshared')
        model.lora_models.set_adapter('lora_shared')
    model.eval()
    background_samples = []
    print(f"Aggregating data for SHAP background (target size: {sample_size})...")
    
    # Loop through the dataloader to collect enough samples
    for batch in dataloader:
        if args.dataname in ['cifar10', 'Awa2', 'CelebA', 'CUB']:
            features = batch['img']
        else:
            raise ValueError(f"Unsupported dataset name for SHAP analysis: {args.dataname}")
        
        background_samples.append(features.cpu()) # Collect on CPU to save GPU memory
        
        # Check if we have collected enough samples
        current_size = sum(t.size(0) for t in background_samples)
        if current_size >= sample_size:
            print(f"Collected {current_size} samples, which meets the target of {sample_size}.")
            break
    
    if not background_samples:
        raise ValueError("Dataloader is empty or failed to load data, cannot create a background dataset for SHAP.")
    
    background_features_full = torch.cat(background_samples, dim=0)
    background_features = background_features_full[:sample_size].to(device)
  
    with torch.no_grad():
        # Get the list of concept predictions from the encoder
        concept_predictions_list = model.model_x_to_c(background_features,i)
        # Concatenate them into a single tensor [batch_size, n_attributes]
        concept_background = torch.cat(concept_predictions_list, dim=1)

    # Convert to numpy for SHAP. This is the CORRECT background data.
    background_for_shap = concept_background[:sample_size].cpu().numpy()

    # This function should only represent the C->Y part of the model.
    c_to_y_model = partial(model.model_c_to_y, i=i)

    def model_forward(concept_vectors_numpy):
        c_tensor = torch.from_numpy(concept_vectors_numpy).float().to(device)
        with torch.no_grad():
            y_pred = c_to_y_model(c_tensor)
        return y_pred.cpu().numpy()

    # Use KernelExplainer
    explainer = shap.KernelExplainer(model_forward, background_for_shap)
    shap_values = explainer.shap_values(background_for_shap, nsamples=nsamples)
    
    # Process multi-class SHAP values to get a single importance vector
    if isinstance(shap_values, list):
        # shape: (num_classes, sample_size, num_features)
        arr = np.array([np.abs(s) for s in shap_values])  
        # shape: (num_features,)
        importance_vector = np.mean(arr, axis=(0, 1))  
    else: # for binary classification or other cases
        shap_abs = np.abs(shap_values)
        if shap_abs.ndim == 3:
            importance_vector = np.mean(shap_abs, axis=(0, 2))
        else:
            importance_vector = np.mean(shap_abs, axis=0)
    return importance_vector

def shap_analysis(dataloader, args):
    """ Perform SHAP analysis on the ensemble model's child models."""
    model_path = os.path.join(args.log_dir, 'best_model.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all child models
    # We only care about the C->Y part, since input data are concepts
    if args.exp == 'X2C':
        child_models = load_child_models_X2C(
            ensemble_model_path=model_path,
            args=args,
            device=device,
            num_concepts=CONCEPT_DIM,
            num_classes=N_CLASSES
        )
    elif args.exp == 'DivEns':
        child_models = load_child_models_DivEns(
            ensemble_model_path=model_path,
            num_models=args.num_models,
            device=device,
            args=args,
            num_concepts=CONCEPT_DIM,
            num_classes=N_CLASSES
        )
    elif args.exp == 'random':
        child_models = load_independent_models(
            model_path=model_path, args=args, device=device,
            num_concepts=CONCEPT_DIM, num_classes=N_CLASSES
        )
    elif args.exp == 'PartialX2C':
        child_models = load_child_models_Partial(
            torch.load(model_path, map_location=device),
            args=args,
            device=device,
            eval_mode=True
        )
    elif args.exp == 'ConvParX2C':
        child_models = load_child_models_Adapter(
            ensemble_model_path=model_path,
            args=args,
            device=device,
            num_concepts=CONCEPT_DIM,
            num_classes=N_CLASSES,
            deepcopy_modules=True,
            share_backbone=False,
            use_checkpoint=True
        )
    elif args.exp == 'ConvAda':
        if args.encoder != "resnet18":
            raise ValueError("--exp ConvAda requires --encoder resnet18.")
        print(model_path)
        child_models = load_child_models_flex(
            ensemble_model_path=model_path,
            args=type("A", (), {
                "num_models": args.num_models,
                "encoder": args.encoder,
                "expand_dim": args.expand_dim,
                "bottleneck_dim": args.bottleneck_dim,
                "share_mask": args.share_mask,
            })(),
            device=device,
            num_concepts=CONCEPT_DIM,
            num_classes=N_CLASSES,
            deepcopy_modules=True,
            share_backbone=False,
            use_checkpoint=True
        )

    elif args.exp == "Revolver":
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

        child_models = load_child_model_revolver(model)
    
    elif args.exp == "Dropout":
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
                # TODO: Check with the google/vit-base-patch16-224?
                baseline_model = AutoModel.from_pretrained("timm/vit_small_patch16_224.augreg_in21k_ft_in1k")
                n_features = baseline_model.timm_model.embed_dim

                targets = [
                    f"timm_model.blocks.{i}" for i in range(11)
                ]
        else:
            baseline_model = SingleE2EBranch(
                n_class_attr=args.n_class_attr, 
                pretrained=True, 
                freeze=False,
                num_classes=N_CLASSES, 
                use_aux=False, 
                n_attributes=args.n_attributes,
                expand_dim=0, 
                encoder=args.encoder
            ).cuda()
            
            if args.encoder == "resnet18":
                targets = [
                    "model_x_to_c.model.layer1",
                    "model_x_to_c.model.layer2",
                    "model_x_to_c.model.layer3",
                    # "model_x_to_c.layer4"
                ]

            elif args.encoder == "vit":
                targets = [
                    f"model_x_to_c.model.timm_model.blocks.{i}" for i in range(11)
                ]
            
            n_features = 0

        # NOTE: We could force num_models = 1 as this doesn't effect training
        model = DropoutEnsemble(
            n_models=args.num_models,
            n_classes=N_CLASSES,
            n_attributes=args.n_attributes,
            n_features=n_features,
            baseline=baseline_model,
            targets=targets,
            dropouts=args.dropout,
            encoder=args.encoder,
            passthrough=args.passthrough
        ).cuda()

        child_models = load_child_model_dropout(model)

    else:
        raise ValueError(f"Unknown experiment type for loading: {args.exp}")

    if not child_models:
        print("Failed to load any child models, exiting.")
        exit()

    # Compute feature importance for each child model
    importance_results = {}
    for i, model in enumerate(child_models):
        print(f"Computing feature importance for child model {i}...")
        imp_vector = compute_feature_importance_shap(model, dataloader, args, device, sample_size=250, nsamples=250)
        importance_results[f"child_model_{i}"] = imp_vector
        print(f"Feature importance vector for child model {i} computed, shape: {imp_vector.shape}")

    # Compute cosine similarity between child models
    model_keys = list(importance_results.keys())
    similarity_matrix = np.zeros((len(model_keys), len(model_keys)))
    for i in range(len(model_keys)):
        for j in range(len(model_keys)):
            v1 = importance_results[model_keys[i]]
            v2 = importance_results[model_keys[j]]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            similarity_matrix[i, j] = cos_sim
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = similarity_matrix[triu_indices]

    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)
    # Save the similarity matrix
    output_file = os.path.join(args.log_dir, "VI_cosine_similarity_children.txt")
    with open(output_file, "w") as f:
        f.write("Cosine Similarity Matrix between child models (C->Y part):\n")
        f.write("      " + " ".join([f"{key:^12}" for key in model_keys]) + "\n")
        for i, key in enumerate(model_keys):
            row_str = " ".join([f"{similarity_matrix[i, j]:<12.4f}" for j in range(len(model_keys))])
            f.write(f"{key:<6}: {row_str}\n")
        f.write(f"\nMean Cosine Similarity between child models (C->Y part): {mean_cross:.4f} ± {std_cross:.4f}\n")


    print(f"Cosine similarity matrix saved to {output_file}")

    # Finding the union of Top 10 concepts from all child models
    print("Finding the union of Top 10 concepts from all child models...")
    all_top_indices = set()
    for model_key in importance_results.keys():
        imp_vector = importance_results[model_key]
        top_10_for_model = np.argsort(-imp_vector)[:10]
        all_top_indices.update(top_10_for_model)

    # Convert the set to a sorted list to ensure consistent index order
    union_indices = sorted(list(all_top_indices))
    print(f"A total of {len(union_indices)} unique important concepts will be used for comparison.")
    print(f"Union of important concept indices: {union_indices}")

    model_keys = list(importance_results.keys())
    similarity_matrix = np.zeros((len(model_keys), len(model_keys)))
    for i in range(len(model_keys)):
        for j in range(len(model_keys)):
            # Get the full original importance vector
            v1_full = importance_results[model_keys[i]]
            v2_full = importance_results[model_keys[j]]
            
            # Use union_indices to select the dimensions deemed important by any model
            v1_union = v1_full[union_indices]
            v2_union = v2_full[union_indices]
            
            # Compute cosine similarity on this subvector of union dimensions
            cos_sim = np.dot(v1_union, v2_union) / (np.linalg.norm(v1_union) * np.linalg.norm(v2_union) + 1e-8)
            similarity_matrix[i, j] = cos_sim
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = similarity_matrix[triu_indices]

    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)

    output_file = os.path.join(args.log_dir, "VI_cosine_similarity_children_union_top10.txt") 
    with open(output_file, "w") as f:
        f.write(f"A total of {len(union_indices)} unique important concepts will be used for comparison.")

        f.write("Cosine Similarity Matrix between child models (C->Y part), based on union of top-10 indices:\n")
        f.write("      " + " ".join([f"{key:^12}" for key in model_keys]) + "\n")
        for i, key in enumerate(model_keys):
            row_str = " ".join([f"{similarity_matrix[i, j]:<12.4f}" for j in range(len(model_keys))])
            f.write(f"{key:<6}: {row_str}\n")
        f.write(f"\nMean Cosine Similarity between child models (C->Y part): {mean_cross:.4f} ± {std_cross:.4f}\n")

    print(f"Cosine similarity matrix based on union of top-10 saved to {output_file}")

    output_file = os.path.join(args.log_dir, "top_features_children.txt")
    with open(output_file, "w") as f:
        f.write("Top 10 features for each child model (C->Y part):\n")
        for key in model_keys:
            imp = importance_results[key]
            top_indices = np.argsort(-imp)[:10]
            top_values = imp[top_indices]
            f.write(f"\n{key}:\n")
            for idx, val in zip(top_indices, top_values):
                f.write(f"  Feature id {idx:<4} importance: {val:.6f}\n")
    print(f"Top-10 important features saved to {output_file}")

def shap_analysis_lora(dataloader, args):
    """ Perform SHAP analysis on the ensemble model's child models."""
    model_path = os.path.join(args.log_dir, 'best_model.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.share_mask == '000000000000':
        model = LoraEnsemble(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )  
    elif args.share_mask == '111111111111':
        model = LoraEnsembleshared(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )  
    elif set(args.share_mask)<=set(['0','1']) or len(args.share_mask)==12:
        model = LoraEnsemblePshared(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
    else:
        raise ValueError(f"Unknown share_mask: {args.share_mask}")

    model.to(device)
    model.load_state_dict(torch.load(model_path,map_location=device),strict=True)
    model.eval()

    # Load all child models
    # We only care about the C->Y part, since input data are concepts

    if not (args.exp=='Lora' or args.exp=='Sparse_Concept'):
        print("This function is for lora model.")
        exit()

    # Compute feature importance for each child model
    
    importance_results = {}
    for i in range(args.num_models):
        print(f"Computing feature importance for child model {i}...")
        imp_vector = compute_feature_importance_shap_lora(model, i, dataloader, args, device, sample_size=250, nsamples=250)
        importance_results[f"child_model_{i}"] = imp_vector
        print(f"Feature importance vector for child model {i} computed, shape: {imp_vector.shape}")

    # Compute cosine similarity between child models
    model_keys = list(importance_results.keys())
    similarity_matrix = np.zeros((len(model_keys), len(model_keys)))
    for i in range(len(model_keys)):
        for j in range(len(model_keys)):
            v1 = importance_results[model_keys[i]]
            v2 = importance_results[model_keys[j]]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            similarity_matrix[i, j] = cos_sim
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = similarity_matrix[triu_indices]

    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)

    # Save the similarity matrix
    output_file = os.path.join(args.log_dir, "VI_cosine_similarity_children.txt")
    with open(output_file, "w") as f:
        f.write("Cosine Similarity Matrix between child models (C->Y part):\n")
        f.write("      " + " ".join([f"{key:^12}" for key in model_keys]) + "\n")
        for i, key in enumerate(model_keys):
            row_str = " ".join([f"{similarity_matrix[i, j]:<12.4f}" for j in range(len(model_keys))])
            f.write(f"{key:<6}: {row_str}\n")
        f.write(f"\nMean Cosine Similarity between child models (C->Y part): {mean_cross:.4f} ± {std_cross:.4f}\n")

    print(f"Cosine similarity matrix saved to {output_file}")

    # Finding the union of Top 10 concepts from all child models
    print("Finding the union of Top 10 concepts from all child models...")
    all_top_indices = set()
    for model_key in importance_results.keys():
        imp_vector = importance_results[model_key]
        top_10_for_model = np.argsort(-imp_vector)[:10]
        all_top_indices.update(top_10_for_model)

    # Convert the set to a sorted list to ensure consistent index order
    union_indices = sorted(list(all_top_indices))
    print(f"A total of {len(union_indices)} unique important concepts will be used for comparison.")
    print(f"Union of important concept indices: {union_indices}")

    model_keys = list(importance_results.keys())
    similarity_matrix = np.zeros((len(model_keys), len(model_keys)))
    for i in range(len(model_keys)):
        for j in range(len(model_keys)):
            # Get the full original importance vector
            v1_full = importance_results[model_keys[i]]
            v2_full = importance_results[model_keys[j]]
            
            # Use union_indices to select the dimensions deemed important by any model
            v1_union = v1_full[union_indices]
            v2_union = v2_full[union_indices]
            
            # Compute cosine similarity on this subvector of union dimensions
            cos_sim = np.dot(v1_union, v2_union) / (np.linalg.norm(v1_union) * np.linalg.norm(v2_union) + 1e-8)
            similarity_matrix[i, j] = cos_sim
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = similarity_matrix[triu_indices]

    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)

    output_file = os.path.join(args.log_dir, "VI_cosine_similarity_children_union_top10.txt") 
    with open(output_file, "w") as f:
        f.write(f"A total of {len(union_indices)} unique important concepts will be used for comparison.")
        f.write(f"Union of important concept indices: {union_indices}")

        f.write("Cosine Similarity Matrix between child models (C->Y part), based on union of top-10 indices:\n")
        f.write("      " + " ".join([f"{key:^12}" for key in model_keys]) + "\n")
        for i, key in enumerate(model_keys):
            row_str = " ".join([f"{similarity_matrix[i, j]:<12.4f}" for j in range(len(model_keys))])
            f.write(f"{key:<6}: {row_str}\n")
        f.write(f"\nMean Cosine Similarity between child models (C->Y part), based on union of top-10 indices: {mean_cross:.4f} ± {std_cross:.4f}\n")
    print(f"Cosine similarity matrix based on union of top-10 saved to {output_file}")

    output_file = os.path.join(args.log_dir, "top_features_children.txt")
    with open(output_file, "w") as f:
        f.write("Top 10 features for each child model (C->Y part):\n")
        for key in model_keys:
            imp = importance_results[key]
            top_indices = np.argsort(-imp)[:10]
            top_values = imp[top_indices]
            f.write(f"\n{key}:\n")
            for idx, val in zip(top_indices, top_values):
                f.write(f"  Feature id {idx:<4} importance: {val:.6f}\n")
    print(f"Top-10 important features saved to {output_file}")

# =======================================================
# PREDICTION SIMILARITY
# =======================================================
def q_cosine(a, b, eps=1e-8):
    """
    Calculates the cosine similarity between two tensors.
    """
    a_norm = F.normalize(a, p=2, dim=1, eps=eps)
    b_norm = F.normalize(b, p=2, dim=1, eps=eps)
    return (a_norm * b_norm).sum(dim=1).mean().item()

def gram_linear(x):
    """Compute the Gram matrix using a linear kernel."""
    return torch.mm(x, x.T)

def gram_rbf(x, threshold=1.0):
    """Compute the Gram matrix using an RBF kernel."""
    dot_products = torch.mm(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_dist = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_dist = torch.median(sq_dist)
    # Add a small epsilon for numerical stability
    return torch.exp(-sq_dist / (2 * threshold ** 2 * sq_median_dist + 1e-9))

def center_gram(gram):
    """Center the Gram matrix."""
    if not torch.all(torch.isfinite(gram)):
        raise ValueError('Gram matrix contains non-finite values.')
    
    n = gram.shape[0]
    means = torch.mean(gram, dim=0)
    mean_all = torch.mean(means)
    gram = gram - means[:, None] - means[None, :] + mean_all
    return gram

def cka(x1, x2, kernel=gram_rbf):
    # Move inputs to CPU and ensure float type for calculations
    x1 = x1.cpu().float()
    x2 = x2.cpu().float()
    
    # Compute and center the Gram matrices
    gram1 = center_gram(kernel(x1))
    gram2 = center_gram(kernel(x2))

    # Compute CKA score
    scaled_hsic = torch.dot(gram1.flatten(), gram2.flatten())
    
    norm1 = torch.norm(gram1, p='fro')
    norm2 = torch.norm(gram2, p='fro')
    
    # Add a small epsilon to prevent division by zero
    cka_val = scaled_hsic / (norm1 * norm2 + 1e-9)

    if cka_val > 1.0:
        cka_val = torch.tensor(1.0)
    
    return cka_val.item()

def analyze_model_similarity(loader, args):
    """
    Loads a trained ensemble model and computes the pairwise prediction similarity
    among its child models on a given dataset.

    Args:
        model (torch.nn.Module): The instantiated ensemble model with loaded weights.
        loader (DataLoader): The DataLoader for the test set.
        args (argparse.Namespace): Arguments used to configure the model and data.
    """
    output_file_path = os.path.join(args.log_dir, "similarity_analysis_results.txt")
    logger = Logger(output_file_path)
    logger.write(f"Starting similarity analysis. Results will be saved to {output_file_path}")

    logger.write("Instantiating model...")

    if args.exp == 'X2C':
        model = EnsembleXtoCtoY(
            num_models=args.num_models,
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=N_CLASSES,
            use_aux=args.use_aux,
            n_attributes=CONCEPT_DIM,
            expand_dim=args.expand_dim,
            encoder=args.encoder
        )
    elif args.exp == 'DivEns':
        model = EnsembleWithSharedXtoC(
            num_models=args.num_models,
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=N_CLASSES,
            use_aux=args.use_aux,
            n_attributes=CONCEPT_DIM,
            expand_dim=args.expand_dim,
            encoder=args.encoder
        )
    elif args.exp == 'PartialX2C':
        model = EnsembleWithPartialSharing(
            num_models=args.num_models,
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=N_CLASSES,
            use_aux=args.use_aux,
            n_attributes=CONCEPT_DIM,
            expand_dim=args.expand_dim,
            encoder=args.encoder
        )
    elif args.exp == 'ConvParX2C':
        model = ConvParEnsemble(
            num_models=args.num_models,
            num_classes=N_CLASSES,
            n_attributes=CONCEPT_DIM,
            encoder=args.encoder,
            bottleneck_dim=args.bottleneck_dim,
            expand_dim=args.expand_dim
        )
    elif args.exp == 'ConvAda':
        if args.encoder != "resnet18":
            raise ValueError("--exp ConvAda requires --encoder resnet18.")
        model = SConvParEnsemble(
            num_models=args.num_models,
            num_classes=N_CLASSES,
            n_attributes=CONCEPT_DIM,
            encoder=args.encoder,
            bottleneck_dim=args.bottleneck_dim,
            expand_dim=args.expand_dim,
            share_mask=args.share_mask,
            use_pretrained=True
        )
    elif args.exp=='Lora' or args.exp=='Sparse_Concept':
        print(args.share_mask,args.share_mask=='0'*len(args.share_mask))
        if set(args.share_mask)>set(['0','1']) or len(args.share_mask)!=12:
            raise ValueError("share_mask should be a string of 0 and 1 with length 12 for vit small and base model, e.g., '110011001100'")
        if args.share_mask == '0'*len(args.share_mask):
            print('none of the lora layers are shared')
            model=LoraEnsemble(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        elif args.share_mask == '1'*len(args.share_mask):
            print('all of the lora layers are shared')
            model = LoraEnsembleshared(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        else:
            print('some of the lora layers are shared')
            model = LoraEnsemblePshared(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        if args.encoder not in ('vit', 'medical_vit'):
            raise NotImplementedError("exp='Lora' only supports encoder in {'vit','medical_vit'}.")
        
    elif args.exp == "Revolver":
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

        epoch_runner_func = run_epoch_revolver
    
    elif args.exp == "Dropout":
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
                # TODO: Check with the google/vit-base-patch16-224?
                baseline_model = AutoModel.from_pretrained("timm/vit_small_patch16_224.augreg_in21k_ft_in1k")
                n_features = baseline_model.timm_model.embed_dim

                targets = [
                    f"timm_model.blocks.{i}" for i in range(11)
                ]
        else:
            baseline_model = SingleE2EBranch(
                n_class_attr=args.n_class_attr, 
                pretrained=True, 
                freeze=False,
                num_classes=N_CLASSES, 
                use_aux=False, 
                n_attributes=args.n_attributes,
                expand_dim=0, 
                encoder=args.encoder
            ).cuda()
            
            if args.encoder == "resnet18":
                targets = [
                    "model_x_to_c.model.layer1",
                    "model_x_to_c.model.layer2",
                    "model_x_to_c.model.layer3",
                    # "model_x_to_c.layer4"
                ]

            elif args.encoder == "vit":
                targets = [
                    f"model_x_to_c.model.timm_model.blocks.{i}" for i in range(11)
                ]
                
            n_features = 0

        # NOTE: We could force num_models = 1 as this doesn't effect training
        model = DropoutEnsemble(
            n_models=args.num_models,
            n_classes=N_CLASSES,
            n_attributes=args.n_attributes,
            n_features=n_features,
            baseline=baseline_model,
            targets=targets,
            dropouts=args.dropout,
            encoder=args.encoder,
            passthrough=args.passthrough
        ).cuda()

        epoch_runner_func = run_epoch_dropout

    model_path = os.path.join(args.log_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        logger.write(f"Error: Model file not found at {model_path}")
        exit()
    
    logger.write(f"Loading saved model from: {model_path}")
    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.cuda()

    # Use defaultdict to easily accumulate similarity scores
    y_similarities = defaultdict(float)
    c_similarities = defaultdict(float)
    batch_count = 0

    # Lists to store predictions from all batches for final calculations
    y_preds_for_hamming = [[] for _ in range(args.num_models)]
    c_preds_for_cka = [[] for _ in range(args.num_models)]

    logger.write("Running inference and calculating similarity...")
    with torch.no_grad():
        for _, data in enumerate(loader):
            if args.dataname in ['cifar10', 'Awa2', 'CelebA','CUB']:
                inputs = data['img']
                y_true = data['class_label']
                c_true = data['attribute_label']

            inputs_var = inputs.cuda()
                
            # Get predictions from the ensemble model
            if args.exp in ["X2C", 'ConvParX2C', 'PartialX2C', 'ConvAda','Lora', 'Dropout', 'Revolver','Sparse_Concept']:
                all_y_preds, all_c_preds = model(inputs_var)

                concatenated_c_preds = []
                for i in range(args.num_models):
                    model_c_preds = torch.cat([p.view(-1, 1) for p in all_c_preds[i]], dim=1)
                    concatenated_c_preds.append(model_c_preds)
            else:
                all_y_preds, _ = model(inputs_var)
            
            # Store batch predictions for final Hamming and CKA calculations
            batch_y_labels = [torch.argmax(p, dim=1) for p in all_y_preds]
            for i in range(args.num_models):
                y_preds_for_hamming[i].append(batch_y_labels[i])
                if args.exp in ["X2C", 'ConvParX2C', 'PartialX2C', 'ConvAda','Lora', 'Dropout', 'Revolver','Sparse_Concept']:
                    c_preds_for_cka[i].append(concatenated_c_preds[i].cpu())

            # Calculate pairwise similarity for all unique pairs of child models
            for i, j in itertools.combinations(range(args.num_models), 2):
                # Similarity for class predictions (Y)
                y_sim = F.cosine_similarity(all_y_preds[i], all_y_preds[j]).mean().item()
                y_similarities[(i, j)] += y_sim

                # Similarity for concept predictions (C)
                if args.exp in ["X2C", 'ConvParX2C', 'PartialX2C', 'ConvAda','Lora', 'Dropout', 'Revolver','Sparse_Concept']:
                    c_sim = F.cosine_similarity(concatenated_c_preds[i], concatenated_c_preds[j]).mean().item()
                    c_similarities[(i, j)] += c_sim
            
            batch_count += 1
            if batch_count % 50 == 0:
                logger.write(f"  Processed {batch_count} batches...")

    logger.write("\n--- Similarity Analysis Results ---")
    
    # --- Class Prediction (Y) Cosine Similarity ---
    logger.write("\nAverage Cosine Similarity of Class Predictions (Y):")
    y_sim_matrix = np.ones((args.num_models, args.num_models))
    for (i, j), total_sim in y_similarities.items():
        avg_sim = total_sim / batch_count
        y_sim_matrix[i, j] = avg_sim
        y_sim_matrix[j, i] = avg_sim # Symmetric matrix
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = y_sim_matrix[triu_indices]
    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)
    df_y_cos = pd.DataFrame(y_sim_matrix, index=[f"Model {i}" for i in range(args.num_models)], columns=[f"Model {i}" for i in range(args.num_models)])
    logger.write(df_y_cos.to_string(float_format="%.4f"))
    logger.write(f"Mean Cosine Similarity (Y) across different models: {mean_cross:.4f} ± {std_cross:.4f}")

    # --- Class Prediction (Y) Hamming Distance ---
    y_labels_per_model = [torch.cat(preds) for preds in y_preds_for_hamming]
    total_samples = len(y_labels_per_model[0])
    hamming_dist_matrix = np.zeros((args.num_models, args.num_models))

    for i in range(args.num_models):
        for j in range(i, args.num_models):
            disagreement_count = torch.sum(y_labels_per_model[i] != y_labels_per_model[j]).item()
            normalized_dist = disagreement_count / total_samples
            hamming_dist_matrix[i, j] = normalized_dist
            hamming_dist_matrix[j, i] = normalized_dist
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = hamming_dist_matrix[triu_indices]
    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)

    logger.write("\nNormalized Hamming Distance of Class Predictions (Y) [Disagreement Rate]:")
    df_ham = pd.DataFrame(hamming_dist_matrix, index=[f"Model {i}" for i in range(args.num_models)], columns=[f"Model {i}" for i in range(args.num_models)])
    logger.write(df_ham.to_string(float_format="%.4f"))
    logger.write(f"Mean nNormalized Hamming Distance of Class Predictions (Y) across different models: {mean_cross:.4f} ± {std_cross:.4f}")

    if args.exp in ['X2C', 'ConvParX2C', 'PartialX2C', 'ConvAda','Lora', 'Dropout', 'Revolver','Sparse_Concept']:
        # --- Concept Prediction (C) Cosine Similarity ---
        logger.write("\nAverage Cosine Similarity of Concept Predictions (C):")
        c_sim_matrix = np.ones((args.num_models, args.num_models))
        for (i, j), total_sim in c_similarities.items():
            avg_sim = total_sim / batch_count
            c_sim_matrix[i, j] = avg_sim
            c_sim_matrix[j, i] = avg_sim
        triu_indices = np.triu_indices(args.num_models, k=1)
        cross_values = c_sim_matrix[triu_indices]
        mean_cross = np.mean(cross_values)
        std_cross = np.std(cross_values)

        df_c_cos = pd.DataFrame(c_sim_matrix, index=[f"Model {i}" for i in range(args.num_models)], columns=[f"Model {i}" for i in range(args.num_models)])
        logger.write(df_c_cos.to_string(float_format="%.4f"))
        logger.write(f"\nMean Cosine Similarity of Concept Predictions (C) across different models: {mean_cross:.4f} ± {std_cross:.4f}")

        # --- Concept Prediction (C) Central Kernal Alignment ---
        c_preds_per_model = [torch.cat(preds) for preds in c_preds_for_cka]

        cka_matrix_linear = np.zeros((args.num_models, args.num_models))
        cka_matrix_rbf = np.zeros((args.num_models, args.num_models))

        for i in range(args.num_models):
            for j in range(i, args.num_models):
                # CKA with Linear Kernel
                cka_linear_val = cka(c_preds_per_model[i], c_preds_per_model[j], kernel=gram_linear)
                cka_matrix_linear[i, j] = cka_matrix_linear[j, i] = cka_linear_val
        triu_indices = np.triu_indices(args.num_models, k=1)
        cross_values = cka_matrix_linear[triu_indices]
        mean_cross = np.mean(cross_values)
        std_cross = np.std(cross_values)
        
        logger.write("\nCentered Kernel Alignment (CKA) with Linear Kernel [Concept Representations]:")
        df_cka_linear = pd.DataFrame(cka_matrix_linear, index=[f"Model {i}" for i in range(args.num_models)], columns=[f"Model {i}" for i in range(args.num_models)])
        logger.write(df_cka_linear.to_string(float_format="%.4f"))
        logger.write(f"\nMean Centered Kernel Alignment (CKA) with Linear Kernel across different models: {mean_cross:.4f} ± {std_cross:.4f}")
        
        for i in range(args.num_models):
            for j in range(i, args.num_models):
                # CKA with RBF Kernel
                cka_rbf_val = cka(c_preds_per_model[i], c_preds_per_model[j], kernel=gram_rbf)
                cka_matrix_rbf[i, j] = cka_matrix_rbf[j, i] = cka_rbf_val
        triu_indices = np.triu_indices(args.num_models, k=1)
        cross_values = cka_matrix_rbf[triu_indices]
        mean_cross = np.mean(cross_values)
        std_cross = np.std(cross_values)
        
        logger.write("\nCentered Kernel Alignment (CKA) with RBF Kernel [Concept Representations]:")
        df_cka_rbf = pd.DataFrame(cka_matrix_rbf, index=[f"Model {i}" for i in range(args.num_models)], columns=[f"Model {i}" for i in range(args.num_models)])
        logger.write(df_cka_rbf.to_string(float_format="%.4f"))
        logger.write(f"\nMean nCentered Kernel Alignment (CKA) with RBF Kernel across different models: {mean_cross:.4f} ± {std_cross:.4f}")

    logger.write("\nAnalysis complete.")
    logger.close()

def analyze_independent_model_similarity(loader, args):
    """
    A self-contained function to analyze the similarity of models trained
    under the 'Independent' mode. It handles model loading, inference, and
    all similarity calculations.

    Args:
        loader (DataLoader): The DataLoader for the test set.
        args (argparse.Namespace): Script arguments, must include log_dir, dataname, etc.
    """
    output_file_path = os.path.join(args.log_dir, "independent_similarity_analysis.txt")
    logger = Logger(output_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.write(f"Starting similarity analysis for independently trained models.")
    logger.write(f"Results will be saved to {output_file_path}")

    model_path = os.path.join(args.log_dir, 'best_model.pth')
    logger.write(f"Loading list of independent models from: {model_path}")
    
    try:
        list_of_state_dicts = torch.load(model_path, map_location=device)
        if not isinstance(list_of_state_dicts, list):
            raise TypeError("Model file is not a list of state_dicts.")
    except Exception as e:
        logger.write(f"Error loading model file: {e}")
        logger.write("Please ensure the path is correct and the file was saved from an 'Independent' training run.")
        return

    child_models = []
    for i, state_dict in enumerate(list_of_state_dicts):
        model = SingleE2EBranch(
            n_class_attr=args.n_class_attr, pretrained=False, freeze=False,
            num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=CONCEPT_DIM,
            expand_dim=args.expand_dim, encoder=args.encoder
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        child_models.append(model)
    
    num_models = len(child_models)
    logger.write(f"Successfully loaded and reconstructed {num_models} independent models.")

    all_y_logits = [[] for _ in range(num_models)]
    all_y_labels = [[] for _ in range(num_models)]
    all_c_vectors = [[] for _ in range(num_models)]

    logger.write("Running inference on the test set to collect all predictions...")
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if args.dataname in ['cifar10', 'Awa2', 'CelebA','CUB']:
                inputs = data['img']
            
            inputs_var = inputs.to(device)

            for i, model in enumerate(child_models):
                y_logits, c_pred_list = model(inputs_var, is_training=False)
                all_y_logits[i].append(y_logits.cpu())
                all_y_labels[i].append(torch.argmax(y_logits, dim=1).cpu())
                all_c_vectors[i].append(torch.cat(c_pred_list, dim=1).cpu())
            
            if (batch_idx + 1) % 50 == 0:
                logger.write(f"  Processed {batch_idx + 1} batches...")

    logger.write("Inference complete. Calculating similarity metrics...")
    y_logits_per_model = [torch.cat(preds) for preds in all_y_logits]
    y_labels_per_model = [torch.cat(preds) for preds in all_y_labels]
    c_vectors_per_model = [torch.cat(preds) for preds in all_c_vectors]
    total_samples = len(y_labels_per_model[0])
    
    logger.write("\n--- Similarity Analysis Results ---")
    
    # Class Prediction (Y) - Cosine Similarity
    logger.write("\nAverage Cosine Similarity of Class Predictions (Y Logits):")
    y_sim_matrix = np.ones((num_models, num_models))
    for i, j in itertools.combinations(range(num_models), 2):
        avg_sim = F.cosine_similarity(y_logits_per_model[i], y_logits_per_model[j]).mean().item()
        y_sim_matrix[i, j] = y_sim_matrix[j, i] = avg_sim
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = y_sim_matrix[triu_indices]
    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)
    df_y_cos = pd.DataFrame(y_sim_matrix, index=[f"Model {i}" for i in range(num_models)], columns=[f"Model {i}" for i in range(num_models)])
    logger.write(df_y_cos.to_string(float_format="%.4f"))
    logger.write(f"Mean Cosine Similarity (Y) across different models: {mean_cross:.4f} ± {std_cross:.4f}")

    # Class Prediction (Y) - Normalized Hamming Distance
    logger.write("\nNormalized Hamming Distance of Class Predictions (Y Labels) [Disagreement Rate]:")

    hamming_dist_matrix = np.zeros((num_models, num_models))
    for i, j in itertools.combinations(range(num_models), 2):
        disagreement_count = torch.sum(y_labels_per_model[i] != y_labels_per_model[j]).item()
        normalized_dist = disagreement_count / total_samples
        hamming_dist_matrix[i, j] = hamming_dist_matrix[j, i] = normalized_dist
    df_ham = pd.DataFrame(hamming_dist_matrix, index=[f"Model {i}" for i in range(num_models)], columns=[f"Model {i}" for i in range(num_models)])
    logger.write(df_ham.to_string(float_format="%.4f"))
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = hamming_dist_matrix[triu_indices]
    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)
    logger.write(f"Mean nNormalized Hamming Distance of Class Predictions (Y) across different models: {mean_cross:.4f} ± {std_cross:.4f}")

    # Concept Prediction (C) - Cosine Similarity
    logger.write("\nAverage Cosine Similarity of Concept Representations (C):")

    
    c_sim_matrix = np.ones((num_models, num_models))
    for i, j in itertools.combinations(range(num_models), 2):
        avg_sim = F.cosine_similarity(c_vectors_per_model[i], c_vectors_per_model[j]).mean().item()
        c_sim_matrix[i, j] = c_sim_matrix[j, i] = avg_sim
    df_c_cos = pd.DataFrame(c_sim_matrix, index=[f"Model {i}" for i in range(num_models)], columns=[f"Model {i}" for i in range(num_models)])
    logger.write(df_c_cos.to_string(float_format="%.4f"))
    triu_indices = np.triu_indices(args.num_models, k=1)    
    cross_values = c_sim_matrix[triu_indices]
    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)

    logger.write(f"\nMean Cosine Similarity of Concept Predictions (C) across different models: {mean_cross:.4f} ± {std_cross:.4f}")

    # Concept Prediction (C) - CKA (Linear Kernel)
    logger.write("\nCentered Kernel Alignment (CKA) with Linear Kernel [Concept Representations]:")
    cka_matrix_linear = np.ones((num_models, num_models))
    for i, j in itertools.combinations(range(num_models), 2):
        cka_val = cka(c_vectors_per_model[i], c_vectors_per_model[j], kernel=gram_linear)
        cka_matrix_linear[i, j] = cka_matrix_linear[j, i] = cka_val
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = cka_matrix_linear[triu_indices]
    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)

    df_cka_linear = pd.DataFrame(cka_matrix_linear, index=[f"Model {i}" for i in range(num_models)], columns=[f"Model {i}" for i in range(num_models)])
    logger.write(df_cka_linear.to_string(float_format="%.4f"))
    logger.write(f"\nMean Centered Kernel Alignment (CKA) with Linear Kernel across different models: {mean_cross:.4f} ± {std_cross:.4f}")

    # Concept Prediction (C) - CKA (RBF Kernel)
    logger.write("\nCentered Kernel Alignment (CKA) with RBF Kernel [Concept Representations]:")
    cka_matrix_rbf = np.ones((num_models, num_models))
    for i, j in itertools.combinations(range(num_models), 2):
        cka_val = cka(c_vectors_per_model[i], c_vectors_per_model[j], kernel=gram_rbf)
        cka_matrix_rbf[i, j] = cka_matrix_rbf[j, i] = cka_val
    df_cka_rbf = pd.DataFrame(cka_matrix_rbf, index=[f"Model {i}" for i in range(num_models)], columns=[f"Model {i}" for i in range(num_models)])
    logger.write(df_cka_rbf.to_string(float_format="%.4f"))
    triu_indices = np.triu_indices(args.num_models, k=1)
    cross_values = cka_matrix_rbf[triu_indices]
    mean_cross = np.mean(cross_values)
    std_cross = np.std(cross_values)
    logger.write(f"\nMean nCentered Kernel Alignment (CKA) with RBF Kernel across different models: {mean_cross:.4f} ± {std_cross:.4f}")

    logger.write("\nAnalysis complete.")
    logger.close()
def analyze_independent_test_acc(train_loader, val_loader, test_loader, args):
    '''
    Analyze accuracy for independent (random init) models on train/val/test datasets.
    Similar to analyze_model_test_acc but handles list of independent models.
    '''
    print('entering analyze_independent_test_acc')
    folds = args.folds
    args.n_attributes = CONCEPT_DIM
    args.lambda_c_acc = 1.0
    
    # Split test dataset into k folds
    test_dataset = test_loader.dataset
    n = len(test_dataset)
    lengths = [n // folds] * folds
    for i in range(n % folds):
        lengths[i] += 1
    
    g = torch.Generator().manual_seed(42)
    subsets = random_split(test_dataset, lengths, generator=g)
    test_loaders = [
        DataLoader(subset, batch_size=test_loader.batch_size, shuffle=False)
        for subset in subsets
    ]

    output_file_path = os.path.join(args.log_dir, "dataset_accuracy.txt")
    logger = Logger(output_file_path)
    logger.write(f"Starting multi-dataset accuracy analysis with {folds}-fold validation on test set")
    logger.write(f"Results will be saved to {output_file_path}")
    logger.write(f"Successfully split test dataset into {folds} folds")
    logger.write("Will also evaluate on full train and eval datasets")

    logger.write("Instantiating models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(args.log_dir, 'best_model.pth')
    logger.write(f"Loading list of independent models from: {model_path}")
    
    try:
        list_of_state_dicts = torch.load(model_path, map_location=device)
        if not isinstance(list_of_state_dicts, list):
            raise TypeError("Model file is not a list of state_dicts.")
    except Exception as e:
        logger.write(f"Error loading model file: {e}")
        logger.write("Please ensure the path is correct and the file was saved from an 'Independent' training run.")
        return

    child_models = []
    for i, state_dict in enumerate(list_of_state_dicts):
        model = SingleE2EBranch(
            n_class_attr=args.n_class_attr, pretrained=False, freeze=False,
            num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=CONCEPT_DIM,
            expand_dim=args.expand_dim, encoder=args.encoder
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        child_models.append(model)
    
    num_models = len(child_models)
    logger.write(f"Successfully loaded and reconstructed {num_models} independent models.")
    
    epoch_runner_func = run_epoch_independent
    y_criterion = torch.nn.CrossEntropyLoss()
    c_criterion = [torch.nn.BCEWithLogitsLoss() for _ in range(CONCEPT_DIM)]
    val_optimizer = None
    
    # Store results: train and eval single evaluation, test multiple folds
    train_y_acc = None
    train_c_acc = None
    train_ensemble_acc_val = None
    eval_y_acc = None
    eval_c_acc = None
    eval_ensemble_acc_val = None
    test_y_acc_list = []
    test_c_acc_list = []
    test_ensemble_acc_list = []
    
    with torch.no_grad():
        # Evaluate on train dataset
        logger.write("\n=== Evaluating on TRAIN dataset ===")
        train_total_loss, train_ensemble_acc, train_detailed_meters = epoch_runner_func(
            child_models, val_optimizer, train_loader, y_criterion, c_criterion, args, is_training=False
        )
        train_y_acc = [train_detailed_meters['y_acc'][m].avg for m in range(args.num_models)]
        train_c_acc = [train_detailed_meters['c_acc'][m].avg for m in range(args.num_models)]
        train_ensemble_acc_val = train_ensemble_acc.avg
        logger.write(f"Train ensemble accuracy: {train_ensemble_acc_val:.4f}")
        logger.write(f"Train label accuracy per model: {train_y_acc}")
        logger.write(f"Train concept accuracy per model: {train_c_acc}")
        
        # Evaluate on eval (validation) dataset
        logger.write("\n=== Evaluating on EVAL dataset ===")
        eval_total_loss, eval_ensemble_acc, eval_detailed_meters = epoch_runner_func(
            child_models, val_optimizer, val_loader, y_criterion, c_criterion, args, is_training=False
        )
        eval_y_acc = [eval_detailed_meters['y_acc'][m].avg for m in range(args.num_models)]
        eval_c_acc = [eval_detailed_meters['c_acc'][m].avg for m in range(args.num_models)]
        eval_ensemble_acc_val = eval_ensemble_acc.avg
        logger.write(f"Eval ensemble accuracy: {eval_ensemble_acc_val:.4f}")
        logger.write(f"Eval label accuracy per model: {eval_y_acc}")
        logger.write(f"Eval concept accuracy per model: {eval_c_acc}")
        
        # Evaluate on test dataset with k-fold
        logger.write(f"\n=== Evaluating on TEST dataset ({folds} folds) ===")
        for fold_idx, fold_loader in enumerate(test_loaders):
            logger.write(f"\nFold {fold_idx + 1}/{folds}:")
            test_total_loss, test_ensemble_acc, test_detailed_meters = epoch_runner_func(
                child_models, val_optimizer, fold_loader, y_criterion, c_criterion, args, is_training=False
            )
            test_y_acc_fold = [test_detailed_meters['y_acc'][m].avg for m in range(args.num_models)]
            test_c_acc_fold = [test_detailed_meters['c_acc'][m].avg for m in range(args.num_models)]
            
            test_y_acc_list.append(test_y_acc_fold)
            test_c_acc_list.append(test_c_acc_fold)
            test_ensemble_acc_list.append(test_ensemble_acc.avg)
            
            logger.write(f"  Fold {fold_idx + 1} ensemble accuracy: {test_ensemble_acc.avg:.4f}")
            logger.write(f"  Fold {fold_idx + 1} label accuracy per model: {test_y_acc_fold}")
            logger.write(f"  Fold {fold_idx + 1} concept accuracy per model: {test_c_acc_fold}")
    
    # Calculate mean and std across folds for test dataset
    test_y_acc_array = np.array(test_y_acc_list)  # shape: (folds, num_models)
    test_c_acc_array = np.array(test_c_acc_list)  # shape: (folds, num_models)
    test_ensemble_acc_array = np.array(test_ensemble_acc_list)  # shape: (folds,)
    
    test_y_acc_mean = np.mean(test_y_acc_array, axis=0)  # shape: (num_models,)
    test_y_acc_std = np.std(test_y_acc_array, axis=0, ddof=1)
    test_c_acc_mean = np.mean(test_c_acc_array, axis=0)  # shape: (num_models,)
    test_c_acc_std = np.std(test_c_acc_array, axis=0, ddof=1)
    
    test_ensemble_acc_mean = np.mean(test_ensemble_acc_array)
    test_ensemble_acc_std = np.std(test_ensemble_acc_array, ddof=1)
    
    logger.write(f"\n=== Test Dataset Summary (across {folds} folds) ===")
    logger.write(f"Ensemble accuracy: {test_ensemble_acc_mean:.4f} ± {test_ensemble_acc_std:.4f}")
    logger.write(f"Mean label accuracy per model: {test_y_acc_mean}")
    logger.write(f"Std label accuracy per model: {test_y_acc_std}")
    logger.write(f"Mean concept accuracy per model: {test_c_acc_mean}")
    logger.write(f"Std concept accuracy per model: {test_c_acc_std}")
    
    # Save CSV files
    logger.write("\n=== Saving CSV files ===")
    
    # Generate CSV files with mean and std for test accuracy
    csv_label_path = os.path.join(args.log_dir, "accuracy_label.csv")
    csv_concept_path = os.path.join(args.log_dir, "accuracy_concept.csv")
    csv_ensemble_path = os.path.join(args.log_dir, "accuracy_ensemble.csv")
    
    # Separate CSV files for std
    csv_label_std_path = os.path.join(args.log_dir, "accuracy_label_std.csv")
    csv_concept_std_path = os.path.join(args.log_dir, "accuracy_concept_std.csv")
    csv_ensemble_std_path = os.path.join(args.log_dir, "accuracy_ensemble_std.csv")
    
    # Create DataFrames for per-model accuracy (mean values only)
    label_data = {
        'model_id': [f"{i}" for i in range(args.num_models)],
        'train': train_y_acc,
        'eval': eval_y_acc,
        'test': test_y_acc_mean.tolist()
    }
    
    concept_data = {
        'model_id': [f"{i}" for i in range(args.num_models)],
        'train': train_c_acc,
        'eval': eval_c_acc,
        'test': test_c_acc_mean.tolist()
    }
    
    # Create DataFrames for per-model std (only test has std)
    label_std_data = {
        'model_id': [f"{i}" for i in range(args.num_models)],
        'test_std': test_y_acc_std.tolist()
    }
    
    concept_std_data = {
        'model_id': [f"{i}" for i in range(args.num_models)],
        'test_std': test_c_acc_std.tolist()
    }
    
    # Create DataFrame for ensemble accuracy
    ensemble_data = {
        'dataset': ['train', 'eval', 'test'],
        'ensemble_accuracy': [train_ensemble_acc_val, eval_ensemble_acc_val, test_ensemble_acc_mean]
    }
    
    # Create DataFrame for ensemble std (only test has std)
    ensemble_std_data = {
        'dataset': ['test'],
        'ensemble_std': [test_ensemble_acc_std]
    }
    
    df_label = pd.DataFrame(label_data)
    df_concept = pd.DataFrame(concept_data)
    df_ensemble = pd.DataFrame(ensemble_data)
    
    df_label_std = pd.DataFrame(label_std_data)
    df_concept_std = pd.DataFrame(concept_std_data)
    df_ensemble_std = pd.DataFrame(ensemble_std_data)
    
    # Save CSVs
    df_label.to_csv(csv_label_path, index=False)
    df_concept.to_csv(csv_concept_path, index=False)
    df_ensemble.to_csv(csv_ensemble_path, index=False)
    
    df_label_std.to_csv(csv_label_std_path, index=False)
    df_concept_std.to_csv(csv_concept_std_path, index=False)
    df_ensemble_std.to_csv(csv_ensemble_std_path, index=False)
    
    logger.write(f"\n{'='*60}")
    logger.write("CSV FILES GENERATED:")
    logger.write(f"Label accuracy: {csv_label_path}")
    logger.write(f"Label std: {csv_label_std_path}")
    logger.write(f"Concept accuracy: {csv_concept_path}")
    logger.write(f"Concept std: {csv_concept_std_path}")
    logger.write(f"Ensemble accuracy: {csv_ensemble_path}")
    logger.write(f"Ensemble std: {csv_ensemble_std_path}")
    
    logger.write("\nAll analysis done!")
    logger.close()
    return
def analyze_model_test_acc(train_loader, val_loader, test_loader, args):
    print('entering analyze_model_test_acc')
    folds = args.folds
    args.n_attributes=CONCEPT_DIM
    args.lambda_c_acc=1.0
    
    # Split test dataset into k folds for robust evaluation
    test_dataset = test_loader.dataset
    n = len(test_dataset)
    
    # Calculate lengths for each fold
    lengths = [n // folds] * folds
    for i in range(n % folds):
        lengths[i] += 1
    
    g = torch.Generator().manual_seed(42)
    
    # Create reproducible splits
    subsets = random_split(test_dataset, lengths, generator=g)
    
    # Create DataLoaders for each fold
    test_loaders = [
        DataLoader(subset, batch_size=test_loader.batch_size, shuffle=False)
        for subset in subsets
    ]

    output_file_path = os.path.join(args.log_dir, "dataset_accuracy.txt")
    print("output_file_path:",output_file_path)
    logger = Logger(output_file_path)
    logger.write(f"Starting multi-dataset accuracy analysis with {folds}-fold validation on test set")
    logger.write(f"Results will be saved to {output_file_path}")

    logger.write(f"Successfully split test dataset into {folds} folds")
    logger.write("Will also evaluate on full train and eval datasets")


    logger.write("Instantiating model...")

    if args.exp == 'X2C':
        model = EnsembleXtoCtoY(
            num_models=args.num_models,
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=N_CLASSES,
            use_aux=args.use_aux,
            n_attributes=CONCEPT_DIM,
            expand_dim=args.expand_dim,
            encoder=args.encoder
        )
        epoch_runner_func = run_epoch_ensemble_e2e

    elif args.exp == 'DivEns':
        model = EnsembleWithSharedXtoC(
            num_models=args.num_models,
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=N_CLASSES,
            use_aux=args.use_aux,
            n_attributes=CONCEPT_DIM,
            expand_dim=args.expand_dim,
            encoder=args.encoder
        )
        epoch_runner_func = run_ensembleDivEns

    elif args.exp == 'PartialX2C':
        model = EnsembleWithPartialSharing(
            num_models=args.num_models,
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=N_CLASSES,
            use_aux=args.use_aux,
            n_attributes=CONCEPT_DIM,
            expand_dim=args.expand_dim,
            encoder=args.encoder
        )
        epoch_runner_func = run_epoch_partial_ensemble

    elif args.exp == 'ConvParX2C':
        model = ConvParEnsemble(
            num_models=args.num_models,
            num_classes=N_CLASSES,
            n_attributes=CONCEPT_DIM,
            encoder=args.encoder,
            bottleneck_dim=args.bottleneck_dim,
            expand_dim=args.expand_dim
        )
        epoch_runner_func = run_epoch_convpar

    elif args.exp == 'ConvAda':
        if args.encoder != "resnet18":
            raise ValueError("--exp ConvAda requires --encoder resnet18.")
        model = SConvParEnsemble(
            num_models=args.num_models,
            num_classes=N_CLASSES,
            n_attributes=CONCEPT_DIM,
            encoder=args.encoder,
            bottleneck_dim=args.bottleneck_dim,
            expand_dim=args.expand_dim,
            share_mask=args.share_mask,
            use_pretrained=True
        )
        epoch_runner_func = run_epoch_sconvpar

    elif args.exp=='Lora'or args.exp=='Sparse_Concept':
        print(args.share_mask,args.share_mask=='0'*len(args.share_mask))
        if set(args.share_mask)>set(['0','1']) or len(args.share_mask)!=12:
            raise ValueError("share_mask should be a string of 0 and 1 with length 12 for vit small and base model, e.g., '110011001100'")
        if args.share_mask == '0'*len(args.share_mask):
            print('none of the lora layers are shared')
            model=LoraEnsemble(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        elif args.share_mask == '1'*len(args.share_mask):
            print('all of the lora layers are shared')
            model = LoraEnsembleshared(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        else:
            print('some of the lora layers are shared')
            model = LoraEnsemblePshared(
                num_models=args.num_models,
                num_classes=N_CLASSES,
                n_attributes=CONCEPT_DIM,
                encoder=args.encoder,
                expand_dim=args.expand_dim,
                lora_block_mask=args.share_mask,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        if args.encoder not in ('vit', 'medical_vit'):
            raise NotImplementedError("exp='Lora' only supports encoder in {'vit','medical_vit'}.")
        epoch_runner_func = run_epoch_lora_par

    elif args.exp == "Revolver":
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

        epoch_runner_func = run_epoch_revolver
    
    elif args.exp == "Dropout":
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
                # TODO: Check with the google/vit-base-patch16-224?
                baseline_model = AutoModel.from_pretrained("timm/vit_small_patch16_224.augreg_in21k_ft_in1k")
                n_features = baseline_model.timm_model.embed_dim

                targets = [
                    f"timm_model.blocks.{i}" for i in range(11)
                ]
        else:
            baseline_model = SingleE2EBranch(
                n_class_attr=args.n_class_attr, 
                pretrained=True, 
                freeze=False,
                num_classes=N_CLASSES, 
                use_aux=False, 
                n_attributes=args.n_attributes,
                expand_dim=0, 
                encoder=args.encoder
            ).cuda()
            
            if args.encoder == "resnet18":
                targets = [
                    "model_x_to_c.model.layer1",
                    "model_x_to_c.model.layer2",
                    "model_x_to_c.model.layer3",
                    # "model_x_to_c.layer4"
                ]

            elif args.encoder == "vit":
                targets = [
                    f"model_x_to_c.model.timm_model.blocks.{i}" for i in range(11)
                ]
            
            n_features = 0

        # NOTE: We could force num_models = 1 as this doesn't effect training
        model = DropoutEnsemble(
            n_models=args.num_models,
            n_classes=N_CLASSES,
            n_attributes=args.n_attributes,
            n_features=n_features,
            baseline=baseline_model,
            targets=targets,
            dropouts=args.dropout,
            encoder=args.encoder,
            passthrough=args.passthrough
        ).cuda()

        epoch_runner_func = run_epoch_dropout

    model_path = os.path.join(args.log_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        logger.write(f"Error: Model file not found at {model_path}")
        exit()
    
    logger.write(f"Loading saved model from: {model_path}")
    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.cuda()
    y_criterion = torch.nn.CrossEntropyLoss()
    c_criterion = [torch.nn.BCEWithLogitsLoss() for _ in range(CONCEPT_DIM)]
    val_optimizer = None
    
    # Store results: train and eval single evaluation, test multiple folds
    train_y_acc = None
    train_c_acc = None
    train_ensemble_acc_val = None
    eval_y_acc = None
    eval_c_acc = None
    eval_ensemble_acc_val = None
    test_y_acc_list = []  # List of arrays, one per fold
    test_c_acc_list = []  # List of arrays, one per fold
    test_ensemble_acc_list = []  # List of ensemble accuracies, one per fold
    
    with torch.no_grad():
        # Evaluate on train dataset
        logger.write("\n=== Evaluating on TRAIN dataset ===")
        train_total_loss, train_ensemble_acc, train_detailed_meters = epoch_runner_func(
            model, val_optimizer, train_loader, y_criterion, c_criterion, args, is_training=False
        )
        train_y_acc = [train_detailed_meters['y_acc'][m].avg for m in range(args.num_models)]
        train_c_acc = [train_detailed_meters['c_acc'][m].avg for m in range(args.num_models)]
        train_ensemble_acc_val = train_ensemble_acc.avg
        logger.write(f"Train ensemble accuracy: {train_ensemble_acc_val:.4f}")
        logger.write(f"Train label accuracy per model: {train_y_acc}")
        logger.write(f"Train concept accuracy per model: {train_c_acc}")
        
        # Evaluate on eval (validation) dataset
        logger.write("\n=== Evaluating on EVAL dataset ===")
        eval_total_loss, eval_ensemble_acc, eval_detailed_meters = epoch_runner_func(
            model, val_optimizer, val_loader, y_criterion, c_criterion, args, is_training=False
        )
        eval_y_acc = [eval_detailed_meters['y_acc'][m].avg for m in range(args.num_models)]
        eval_c_acc = [eval_detailed_meters['c_acc'][m].avg for m in range(args.num_models)]
        eval_ensemble_acc_val = eval_ensemble_acc.avg
        logger.write(f"Eval ensemble accuracy: {eval_ensemble_acc_val:.4f}")
        logger.write(f"Eval label accuracy per model: {eval_y_acc}")
        logger.write(f"Eval concept accuracy per model: {eval_c_acc}")
        
        # Evaluate on test dataset with k-fold
        logger.write(f"\n=== Evaluating on TEST dataset ({folds} folds) ===")
        for fold_idx, fold_loader in enumerate(test_loaders):
            logger.write(f"\nFold {fold_idx + 1}/{folds}:")
            test_total_loss, test_ensemble_acc, test_detailed_meters = epoch_runner_func(
                model, val_optimizer, fold_loader, y_criterion, c_criterion, args, is_training=False
            )
            test_y_acc_fold = [test_detailed_meters['y_acc'][m].avg for m in range(args.num_models)]
            test_c_acc_fold = [test_detailed_meters['c_acc'][m].avg for m in range(args.num_models)]
            
            test_y_acc_list.append(test_y_acc_fold)
            test_c_acc_list.append(test_c_acc_fold)
            test_ensemble_acc_list.append(test_ensemble_acc.avg)
            
            logger.write(f"  Fold {fold_idx + 1} ensemble accuracy: {test_ensemble_acc.avg:.4f}")
            logger.write(f"  Fold {fold_idx + 1} label accuracy per model: {test_y_acc_fold}")
            logger.write(f"  Fold {fold_idx + 1} concept accuracy per model: {test_c_acc_fold}")
    
    # Calculate mean and std across folds for test dataset
    test_y_acc_array = np.array(test_y_acc_list)  # shape: (folds, num_models)
    test_c_acc_array = np.array(test_c_acc_list)  # shape: (folds, num_models)
    test_ensemble_acc_array = np.array(test_ensemble_acc_list)  # shape: (folds,)
    
    test_y_acc_mean = np.mean(test_y_acc_array, axis=0)  # shape: (num_models,)
    test_y_acc_std = np.std(test_y_acc_array, axis=0, ddof=1)  # shape: (num_models,)
    test_c_acc_mean = np.mean(test_c_acc_array, axis=0)  # shape: (num_models,)
    test_c_acc_std = np.std(test_c_acc_array, axis=0, ddof=1)  # shape: (num_models,)
    test_ensemble_acc_mean = np.mean(test_ensemble_acc_array)  # scalar
    test_ensemble_acc_std = np.std(test_ensemble_acc_array, ddof=1)  # scalar
    
    # Generate CSV files with mean and std for test accuracy
    csv_label_path = os.path.join(args.log_dir, "accuracy_label.csv")
    csv_concept_path = os.path.join(args.log_dir, "accuracy_concept.csv")
    csv_ensemble_path = os.path.join(args.log_dir, "accuracy_ensemble.csv")
    
    # Separate CSV files for std
    csv_label_std_path = os.path.join(args.log_dir, "accuracy_label_std.csv")
    csv_concept_std_path = os.path.join(args.log_dir, "accuracy_concept_std.csv")
    csv_ensemble_std_path = os.path.join(args.log_dir, "accuracy_ensemble_std.csv")
    
    # Create DataFrames for per-model accuracy (mean values only)
    label_data = {
        'model_id': [f"{i}" for i in range(args.num_models)],
        'train': train_y_acc,
        'eval': eval_y_acc,
        'test': test_y_acc_mean.tolist()
    }
    
    concept_data = {
        'model_id': [f"{i}" for i in range(args.num_models)],
        'train': train_c_acc,
        'eval': eval_c_acc,
        'test': test_c_acc_mean.tolist()
    }
    
    # Create DataFrames for per-model std (only test has std)
    label_std_data = {
        'model_id': [f"{i}" for i in range(args.num_models)],
        'test_std': test_y_acc_std.tolist()
    }
    
    concept_std_data = {
        'model_id': [f"{i}" for i in range(args.num_models)],
        'test_std': test_c_acc_std.tolist()
    }
    
    # Create DataFrame for ensemble accuracy
    ensemble_data = {
        'dataset': ['train', 'eval', 'test'],
        'ensemble_accuracy': [train_ensemble_acc_val, eval_ensemble_acc_val, test_ensemble_acc_mean]
    }
    
    # Create DataFrame for ensemble std (only test has std)
    ensemble_std_data = {
        'dataset': ['test'],
        'ensemble_std': [test_ensemble_acc_std]
    }
    
    df_label = pd.DataFrame(label_data)
    df_concept = pd.DataFrame(concept_data)
    df_ensemble = pd.DataFrame(ensemble_data)
    
    df_label_std = pd.DataFrame(label_std_data)
    df_concept_std = pd.DataFrame(concept_std_data)
    df_ensemble_std = pd.DataFrame(ensemble_std_data)
    
    # Save CSVs
    df_label.to_csv(csv_label_path, index=False)
    df_concept.to_csv(csv_concept_path, index=False)
    df_ensemble.to_csv(csv_ensemble_path, index=False)
    
    df_label_std.to_csv(csv_label_std_path, index=False)
    df_concept_std.to_csv(csv_concept_std_path, index=False)
    df_ensemble_std.to_csv(csv_ensemble_std_path, index=False)
    
    logger.write(f"\n{'='*60}")
    logger.write("CSV FILES GENERATED:")
    logger.write(f"Label accuracy: {csv_label_path}")
    logger.write(f"Label std: {csv_label_std_path}")
    logger.write(f"Concept accuracy: {csv_concept_path}")
    logger.write(f"Concept std: {csv_concept_std_path}")
    logger.write(f"Ensemble accuracy: {csv_ensemble_path}")
    logger.write(f"Ensemble std: {csv_ensemble_std_path}")
    logger.write(f"{'='*60}")

    
    # Print summary statistics
    logger.write("\n=== SUMMARY STATISTICS ===")
    logger.write("\nTRAIN dataset:")
    logger.write(f"  Ensemble accuracy: {train_ensemble_acc_val:.4f}")
    logger.write(f"  Label accuracy (mean): {np.mean(train_y_acc):.4f} ± {np.std(train_y_acc):.4f}")
    logger.write(f"  Concept accuracy (mean): {np.mean(train_c_acc):.4f} ± {np.std(train_c_acc):.4f}")
    
    logger.write("\nEVAL dataset:")
    logger.write(f"  Ensemble accuracy: {eval_ensemble_acc_val:.4f}")
    logger.write(f"  Label accuracy (mean): {np.mean(eval_y_acc):.4f} ± {np.std(eval_y_acc):.4f}")
    logger.write(f"  Concept accuracy (mean): {np.mean(eval_c_acc):.4f} ± {np.std(eval_c_acc):.4f}")
    
    logger.write(f"\nTEST dataset (averaged over {folds} folds):")
    logger.write(f"  Ensemble accuracy: {test_ensemble_acc_mean:.4f} ± {test_ensemble_acc_std:.4f}")
    logger.write(f"  Label accuracy (mean): {np.mean(test_y_acc_mean):.4f} ± {np.mean(test_y_acc_std):.4f}")
    logger.write(f"  Concept accuracy (mean): {np.mean(test_c_acc_mean):.4f} ± {np.mean(test_c_acc_std):.4f}")
    
    logger.write("\nPer-model TEST accuracy with std across folds:")
    for i in range(args.num_models):
        logger.write(f"  Model {i}: label={test_y_acc_mean[i]:.4f}±{test_y_acc_std[i]:.4f}, "
                    f"concept={test_c_acc_mean[i]:.4f}±{test_c_acc_std[i]:.4f}")

    print("all analysis done!")
    logger.close()

    return


def parse_arguments_for_analysis():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='SHAP', help='Evaluation task', choices=['SHAP', 'similarity', 'task_acc'])
    parser.add_argument('--exp', type=str, default='Lora', choices=['DivEns', 'random', 'ConvAda', 'Lora', 'Dropout'], help='Model type')
    parser.add_argument('--log_dir', type=str, default='EnsembleX2C/log_cifar10_1', help='Directory where the trained model (best_model.pth) is saved.')
    parser.add_argument('--data_dir', default='', help='Directory for the CUB dataset pickle files.')
    parser.add_argument('--image_dir', default='', help='Root directory for CUB images.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Mini-batch size for inference.')

    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for the dropout ensemble')
    parser.add_argument('--passthrough', action="store_true", help="directly use a random init instance for dropout")

    parser.add_argument('--num_models', type=int, default=3, help='Number of models in the ensemble.')
    parser.add_argument('--encoder', type=str, default='vit', choices=['resnet18', 'vit', 'medical_vit'], help='Encoder architecture used.')
    parser.add_argument('--n_attributes', type=int, help='Number of attributes used.')
    parser.add_argument('--expand_dim', type=int, default=0, help='Dimension of the hidden layer in the bottleneck.')
    parser.add_argument('--n_class_attr', type=int, default=CUB_N_CLASSES, help='Output dimensions for attribute prediction.')
    parser.add_argument('--use_aux', action='store_true', help='Whether the model was trained with auxiliary logits.')
    parser.add_argument('--dataname', type=str, default='cifar10', choices=['CUB', 'cifar10','Awa2','CelebA'], help='Dataset name for loading the correct data.')
    parser.add_argument('--bottleneck_dim', type=int, default=64, help='Bottleneck dimension for adapter-based models.')
    parser.add_argument('--share_mask', type=str, default="11110")
    parser.add_argument('--folds', type=int, default=3, help='Number of folds for test accuracy.')
    parser.add_argument("--acc_earlystop",action="store_true", help="whether to use accuracy for early stopping instead of loss")
    parser.add_argument("--reduced_alpha",action="store_true", help="reduce alpha")
    parser.add_argument('--reduce_alpha', type=float, default=0.1,
                        help="to how much reduce alpha")
    parser.add_argument('--lora_r', type=int, default=8,
                        help="lora_rank")
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help="lora_alpha")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments_for_analysis()
    # Prepare the test data
    if args.dataname == 'CUB':
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val_data_path = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test_data_path = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')
        train_loader = load_CUB_data(train_data_path, batch_size=args.batch_size,is_training=False)
        val_loader = load_CUB_data(val_data_path, batch_size=args.batch_size, is_training=False)
        test_loader = load_CUB_data(test_data_path,  batch_size=args.batch_size, is_training=False)
        N_CLASSES = CUB_N_CLASSES
        CONCEPT_DIM = 112

    elif args.dataname == 'Awa2':
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val_data_path = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test_data_path = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')
        train_loader = load_awa2_data(train_data_path, batch_size=args.batch_size, is_training=False)
        val_loader = load_awa2_data(val_data_path, batch_size=args.batch_size, is_training=False)
        test_loader = load_awa2_data(test_data_path, batch_size=args.batch_size, is_training=False)
        N_CLASSES = AWA2_N_CLASSES
        CONCEPT_DIM = 85

    elif args.dataname == 'cifar10':
        N_CLASSES = CIFAR10_N_CLASSES
        setup_cifar10_dataset(os.path.join(BASE_DIR, args.data_dir), os.path.join(BASE_DIR, args.data_dir))
        train = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')

        train_loader = load_cifar10_data(train, batch_size=args.batch_size, is_training=False)
        val_loader = load_cifar10_data(val, batch_size=args.batch_size, is_training=False)
        test_loader = load_cifar10_data(test, batch_size=args.batch_size, is_training=False)
        N_CLASSES = CIFAR10_N_CLASSES
        CONCEPT_DIM = 143

    elif args.dataname == 'CelebA':
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val_data_path = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test_data_path = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')
        train_loader = load_celeba_data(train_data_path, batch_size=args.batch_size, is_training=False)
        val_loader = load_celeba_data(val_data_path, batch_size=args.batch_size, is_training=False)
        test_loader = load_celeba_data(test_data_path, batch_size=args.batch_size, is_training=False)
        N_CLASSES = CELEBA_N_CLASSES
        CONCEPT_DIM = 6

    if args.task == 'SHAP':
        if args.exp == 'Lora' or args.exp == 'Sparse_Concept':
            shap_analysis_lora(test_loader, args)
        else:
            shap_analysis(test_loader, args)
    elif args.task == 'similarity':
        if args.exp == 'random':
            analyze_independent_model_similarity(test_loader, args)
        else:    
            analyze_model_similarity(test_loader, args)
    elif args.task == 'task_acc':
        if args.exp == 'random':
            analyze_independent_test_acc(train_loader, val_loader, test_loader, args)
        else:    
            analyze_model_test_acc(train_loader, val_loader, test_loader, args)
