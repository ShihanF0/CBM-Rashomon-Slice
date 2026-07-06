from copy import deepcopy
from template_model import MLP, FC
import torch
import torch.nn as nn
import torch.nn.functional as F
from template_model import resnet18, vit, MLP, Adapter
from torchvision import models
import copy
import os
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, ViTModel, ViTConfig, AutoModel, CLIPVisionConfig, CLIPVisionModel
from peft import get_peft_model, LoraConfig, PeftMixedModel, LoraModel, PeftModel, AutoPeftModel
try:
    import bitsandbytes as bnb
except Exception:
    bnb = None
from typing import Dict, List
import json
from safetensors.torch import load_file

# PubMedCLIP provides a CLIP ViT-B/32 image encoder
VIT_DEFAULT_MODEL_NAME = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
MEDICAL_VIT_MODEL_NAME = "flaviagiammarino/pubmed-clip-vit-base-patch32"

# A single, complete X -> C -> Y branch
class SingleE2EBranch(nn.Module):
    def __init__(self, n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, encoder):
        super(SingleE2EBranch, self).__init__()

        model_name = MEDICAL_VIT_MODEL_NAME if encoder == "medical_vit" else VIT_DEFAULT_MODEL_NAME
        model_args = {
            'pretrained': pretrained, 'freeze': freeze, 'num_classes': num_classes, 
            'aux_logits': use_aux, 'n_attributes': n_attributes, 'bottleneck': True, 
            'expand_dim': expand_dim, 'three_class': (n_class_attr == 3), 'encoder': encoder,'model_name': model_name,
        }
        # Part 1: X -> C model
        if encoder == 'resnet18':
            self.model_x_to_c = resnet18(**model_args)
        elif encoder in ('vit', 'medical_vit'):
            self.model_x_to_c = vit(**model_args)
        else:
            raise ValueError(f"Unknown encoder specified: {encoder}")
        
        # Part 2: C -> Y model
        mlp_input_dim = n_attributes * 1
        self.model_c_to_y = MLP(
            input_dim=mlp_input_dim, num_classes=num_classes, expand_dim=expand_dim
        )
    
    def forward(self, x, is_training: bool = True):
        if is_training and getattr(self.model_x_to_c, "aux_logits", False):
            predicted_concepts, _ = self.model_x_to_c(x)
        else:
            predicted_concepts = self.model_x_to_c(x)

        concatenated = torch.cat(predicted_concepts, dim=1)
        final_prediction = self.model_c_to_y(concatenated)
        return final_prediction, predicted_concepts

# Ensemble on XtoC
class EnsembleXtoCtoY(nn.Module):
    def __init__(self, num_models, n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, encoder):
        super(EnsembleXtoCtoY, self).__init__()
        self.num_models = num_models
        self.branches = nn.ModuleList([
            SingleE2EBranch(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, encoder)
            for _ in range(num_models)
        ])

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        all_y_preds = []
        all_c_preds = []

        for i in range(self.num_models):
            y_pred, c_pred = self.branches[i](x, self.training)
            all_y_preds.append(y_pred)
            all_c_preds.append(c_pred)
            
        return all_y_preds, all_c_preds

# Ensemble on CtoY Branches
class BranchCtoY(nn.Module):
    """Single C -> Y branch for the ensemble"""
    def __init__(self, n_class_attr, n_attributes, num_classes, expand_dim):
        super(BranchCtoY, self).__init__()
        if n_class_attr == 3:
            mlp_input_dim = n_attributes * n_class_attr
        else:
            mlp_input_dim = n_attributes
        self.model_c_to_y = MLP(
            input_dim=mlp_input_dim, num_classes=num_classes, expand_dim=expand_dim
        )
    
    def forward(self, concatenated_concepts):
        return self.model_c_to_y(concatenated_concepts)

# Ensemble on CtoY
class EnsembleWithSharedXtoC(nn.Module):
    """Shared X -> C model with multiple C -> Y branches"""
    def __init__(self, num_models, n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, encoder):
        super(EnsembleWithSharedXtoC, self).__init__()
        self.num_models = num_models
        model_name = MEDICAL_VIT_MODEL_NAME if encoder == "medical_vit" else VIT_DEFAULT_MODEL_NAME
        model_args = {
            'pretrained': pretrained, 'freeze': freeze, 'num_classes': num_classes, 
            'aux_logits': use_aux, 'n_attributes': n_attributes, 'bottleneck': True, 
            'expand_dim': expand_dim, 'three_class': (n_class_attr == 3),'model_name': model_name,
        }
        
        if encoder == 'resnet18':
            self.model_x_to_c = resnet18(**model_args)
        elif encoder in ('vit', 'medical_vit'):
            self.model_x_to_c = vit(**model_args)
        else:
            raise ValueError(f"Unknown encoder specified: {encoder}")

        self.branches_c_to_y = nn.ModuleList([
            BranchCtoY(n_class_attr, n_attributes, num_classes, expand_dim)
            for _ in range(num_models)
        ])
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.training and self.model_x_to_c.aux_logits:
            predicted_concepts, _ = self.model_x_to_c(x)
        else:
            predicted_concepts = self.model_x_to_c(x)
        concatenated_concepts = torch.cat(predicted_concepts, dim=1)

        all_y_preds = [branch(concatenated_concepts) for branch in self.branches_c_to_y]
        return all_y_preds, predicted_concepts

class EnsembleWithPartialSharing(nn.Module):
    def __init__(self, num_models, num_classes, n_attributes, encoder,
                 expand_dim=0, split_point='layer4', bottleneck_dim=64, **kwargs):
        super(EnsembleWithPartialSharing, self).__init__()
        
        self.num_models = num_models
        
        # Initialization
        if encoder == 'resnet18':
            resnet18_channel_map = {
                'layer1': 64,
                'layer2': 128,
                'layer3': 256,
                'layer4': 512
            }
            if split_point not in resnet18_channel_map:
                raise ValueError(f"Invalid split_point '{split_point}' for resnet18. "
                                 f"Valid options are: {list(resnet18_channel_map.keys())}")
            
            original_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            
            children = list(original_model.named_children())
            split_idx = [name for name, _ in children].index(split_point)
            
            self.trunk = nn.Sequential(*[module for name, module in children[:split_idx + 1]])
            
            branch_template_layers = [copy.deepcopy(module) for name, module in children[split_idx + 1:] if name not in ['avgpool', 'fc']]
            self.branch_template = nn.Sequential(*branch_template_layers)
            
            self.adapter_in_channels = resnet18_channel_map[split_point]
            self.feature_dim = original_model.fc.in_features
        else:
            raise NotImplementedError(f"Encoder '{encoder}' not implemented.")
            
        # Create individual branches
        self.branches = nn.ModuleList()
        for _ in range(num_models):
            branch = nn.ModuleDict({
                'deep_layers': copy.deepcopy(self.branch_template),
                'concept_head': self._create_concept_head(self.feature_dim, n_attributes, expand_dim),
                'classifier': MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
            })
            self.branches.append(branch)

        # Fine tune strategy
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.branches.parameters():
            param.requires_grad = True

        for param in self.trunk.parameters():
            param.requires_grad = True
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.alpha.requires_grad = True
        
    def _create_concept_head(self, input_dim, n_attributes, expand_dim):
        head = nn.ModuleList()
        for _ in range(n_attributes):
            head.append(FC(input_dim, 1, expand_dim))
        return head
    
    def forward(self, x):
        shared_features = self.trunk(x)
        
        all_y_preds = []
        all_c_preds = []

        for branch in self.branches:
            branch_deep_features = branch['deep_layers'](shared_features)
            if hasattr(branch, 'adapter'):
                adapter_module = getattr(branch, 'adapter')
                if adapter_module is not None:
                    adapted_features = adapter_module(branch_deep_features)
                    pooled_features = F.adaptive_avg_pool2d(adapted_features, (1, 1))
                else:
                    ValueError("Adapter module is None, but expected to be present.")
            else:
                pooled_features = F.adaptive_avg_pool2d(branch_deep_features, (1, 1))
                
            flattened_features = torch.flatten(pooled_features, 1)

            concept_preds = [fc(flattened_features) for fc in branch['concept_head']]
            all_c_preds.append(concept_preds)

            concatenated_concepts = torch.cat(concept_preds, dim=1)
            y_pred = branch['classifier'](concatenated_concepts)
            all_y_preds.append(y_pred)
            
        return all_y_preds, all_c_preds

class ConvParEnsemble(nn.Module):
    def __init__(self, num_models, num_classes, n_attributes, encoder,
                 bottleneck_dim=64, expand_dim=0, **kwargs):
        super(ConvParEnsemble, self).__init__()
        if encoder != 'resnet18':
            raise NotImplementedError("This model is optimized for 'resnet18'.")
        self.num_models = num_models
        self.shared_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = self.shared_backbone.fc.in_features
        self.shared_backbone.fc = nn.Identity()
        self.shared_backbone.requires_grad_(False)
        self.shared_backbone.eval()
        self.adapter_sets = nn.ModuleList([
            self._create_adapter_set(self.shared_backbone, bottleneck_dim) 
            for _ in range(num_models)
        ])
        self.final_heads = nn.ModuleList([
            nn.ModuleDict({
                'concept_head': self._create_concept_head(feature_dim, n_attributes, expand_dim),
                'classifier': MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
            }) for _ in range(num_models)
        ])
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self._freeze_all_bn_eval()

    def _create_adapter_set(self, model, bottleneck_dim):
        adapters = nn.ModuleDict()
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name == 'conv1' or name.endswith('.conv1') or name.endswith('.conv2'):
                    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                    adapters[name.replace('.', '_')] = Adapter(
                        module.in_channels, module.out_channels, bottleneck_dim, stride
                    )
        return adapters

    def _create_concept_head(self, input_dim, n_attributes, expand_dim):
        head = nn.ModuleList()
        for _ in range(n_attributes):
            head.append(MLP(input_dim, 1, expand_dim))
        return head

    def _reconstruct_forward_with_adapters(self, x, backbone, adapters, use_checkpoint: bool=True):
        def get_name(layer_name): return layer_name.replace('.', '_')

        out = backbone.conv1(x) + adapters[get_name('conv1')](x)
        out = backbone.relu(backbone.bn1(out))
        out = backbone.maxpool(out)

        for layer_idx in range(1, 5):
            layer = getattr(backbone, f'layer{layer_idx}')
            for block_idx, block in enumerate(layer):
                def block_forward(inp, li=layer_idx, bi=block_idx, blk=block):
                    identity = inp
                    if blk.downsample is not None:
                        identity = blk.downsample(identity)

                    res = blk.conv1(inp) + adapters[get_name(f'layer{li}.{bi}.conv1')](inp)
                    res = blk.relu(blk.bn1(res))

                    res = blk.conv2(res) + adapters[get_name(f'layer{li}.{bi}.conv2')](res)
                    res = blk.bn2(res)

                    return blk.relu(res + identity)

                out = checkpoint(block_forward, out) if use_checkpoint else block_forward(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def forward(self, x):
        self.shared_backbone.eval()

        all_y_preds, all_c_preds = [], []
        for i in range(self.num_models):
            features = self._reconstruct_forward_with_adapters(
                x, self.shared_backbone, self.adapter_sets[i], use_checkpoint=True
            )

            head = self.final_heads[i]
            concept_preds = [fc(features) for fc in head['concept_head']]
            all_c_preds.append(concept_preds)

            concatenated_concepts = torch.cat(concept_preds, dim=1)
            y_pred = head['classifier'](concatenated_concepts)
            all_y_preds.append(y_pred)

        return all_y_preds, all_c_preds
    
    def _freeze_all_bn_eval(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
                m.eval()
                if m.affine:
                    if m.weight is not None: m.weight.requires_grad_(False)
                    if m.bias   is not None: m.bias.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.shared_backbone is not None:
            self.shared_backbone.eval()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
                m.eval()
        return self


class SConvParEnsemble(nn.Module):
    def __init__(
        self,
        num_models: int,
        num_classes: int,
        n_attributes: int,
        encoder: str = "resnet18",
        bottleneck_dim: int = 64,
        expand_dim: int = 0,
        share_mask: str = "1|11|11|00|00",   # For block-level control use "1|11|11|00|00"; for stage level control, use "11100"
        use_pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        assert encoder == "resnet18", "This implementation is optimized for 'resnet18'."
        self.num_models = int(num_models)
        self.n_attributes = int(n_attributes)
        self.share_mask = share_mask

        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        backbone = models.resnet18(weights=weights)

        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        # stem = conv1/bn1/relu/maxpool
        self.stem = nn.ModuleDict(dict(
            conv1   = backbone.conv1,
            bn1     = backbone.bn1,
            relu    = backbone.relu,
            maxpool = backbone.maxpool,
        ))
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        for p in backbone.parameters():
            p.requires_grad_(False)
        self._freeze_backbone_bn_eval(backbone)

        self._share_block = self._parse_share_mask(self.share_mask)
        self._last_shared_stage = self._find_last_shared_prefix_stage(self._share_block)

        self.shared_adapters, self.branch_adapters = self._build_adapters_by_mask(
            bottleneck_dim=bottleneck_dim,
            share_block=self._share_block
        )

        self.final_heads = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(self.feature_dim),
                'concept_head': self._create_concept_head(self.feature_dim, self.n_attributes, expand_dim),
                'classifier': MLP(input_dim=self.n_attributes, num_classes=num_classes, expand_dim=expand_dim)
            }) for _ in range(self.num_models)
        ])

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.eval()

    def _create_concept_head(self, input_dim, n_attributes, expand_dim):
        head = nn.ModuleList()
        for _ in range(n_attributes):
            head.append(MLP(input_dim, 1, expand_dim))
        return head

    # parsing share mask
    def _parse_share_mask(self, mask: str) -> Dict[int, Dict[int, bool]]:
        if '|' in mask:
            segs = mask.split('|')
            assert len(segs) == 5 and len(segs[0]) == 1 and all(len(s) == 2 for s in segs[1:]), \
                "share_mask must be like '1|11|11|00|00' with segment lengths 1|2|2|2|2"
            bits = [segs[0], *segs[1:]]
        else:
            assert len(mask) == 5, "stage-level mask must have 5 chars like '11100'"
            bits = [mask[0], mask[1]*2, mask[2]*2, mask[3]*2, mask[4]*2]

        share_block = {0: {0: bits[0] == '1'}}
        for li in (1, 2, 3, 4):
            share_block[li] = {bi: (bits[li][bi] == '1') for bi in (0, 1)}
        return share_block

    # find last shared stage
    def _find_last_shared_prefix_stage(self, share_block) -> int:
        def stage_all_shared(li: int) -> bool:
            if li == 0:  # stem
                return share_block[0][0]
            return all(share_block[li][bi] for bi in (0, 1))
        last = -1
        for li in (0, 1, 2, 3, 4):
            if stage_all_shared(li):
                last = li
            else:
                break
        return last

    # Building adapters
    def _build_adapters_by_mask(self, bottleneck_dim: int, share_block):
        shared = nn.ModuleDict()
        branches = [nn.ModuleDict() for _ in range(self.num_models)]

        # stem
        conv = self.stem['conv1']
        stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
        if share_block[0][0]:
            shared['conv1'] = Adapter(conv.in_channels, conv.out_channels, bottleneck_dim, stride)
        else:
            for i in range(self.num_models):
                branches[i]['conv1'] = Adapter(conv.in_channels, conv.out_channels, bottleneck_dim, stride)

        # layer1-4
        for li, layer in ((1, self.layer1), (2, self.layer2), (3, self.layer3), (4, self.layer4)):
            for bi, blk in enumerate(layer):
                for cname in ('conv1', 'conv2'):
                    conv = getattr(blk, cname)
                    stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
                    key = f'layer{li}.{bi}.{cname}'.replace('.', '_')
                    if share_block[li][bi]:
                        shared[key] = Adapter(conv.in_channels, conv.out_channels, bottleneck_dim, stride)
                    else:
                        for i in range(self.num_models):
                            branches[i][key] = Adapter(conv.in_channels, conv.out_channels, bottleneck_dim, stride)
        return shared, nn.ModuleList(branches)

    # foward for shared parts
    def _forward_shared_prefix_masked(self, x):
        out = x
        did_any = False

        if self._last_shared_stage >= 0:
            # stem
            s = self.stem['conv1'](out)
            a = self.shared_adapters['conv1'](out)
            out = self.stem['relu'](self.stem['bn1'](s + a))
            out = self.stem['maxpool'](out)
            did_any = True

            # layer1-last_shared_stage
            for li in (1, 2, 3, 4):
                if li > self._last_shared_stage:
                    break
                layer = getattr(self, f'layer{li}')
                for bi, blk in enumerate(layer):
                    identity = out
                    if blk.downsample is not None:
                        identity = blk.downsample(identity)
                    k1 = f'layer{li}.{bi}.conv1'.replace('.', '_')
                    k2 = f'layer{li}.{bi}.conv2'.replace('.', '_')
                    res = blk.conv1(out) + self.shared_adapters[k1](out)
                    res = blk.relu(blk.bn1(res))
                    res = blk.conv2(res) + self.shared_adapters[k2](res)
                    res = blk.bn2(res)
                    out = blk.relu(res + identity)
        return out, did_any

    # forward unshared parts
    def _forward_suffix_batched_masked(self, x_in, did_shared_prefix: bool, use_checkpoint: bool = False):
        N = self.num_models

        # Prepare per-branch stem outputs
        if did_shared_prefix:
            x_list = [x_in for _ in range(N)]
        else:
            s = self.stem['conv1'](x_in)
            if self._share_block[0][0]:
                a_shared = self.shared_adapters['conv1'](x_in)
                stem_out = self.stem['relu'](self.stem['bn1'](s + a_shared))
                stem_out = self.stem['maxpool'](stem_out)
                x_list = [stem_out for _ in range(N)]
            else:
                x_list = []
                for i in range(N):
                    a_i = self.branch_adapters[i]['conv1'](x_in)
                    out_i = self.stem['relu'](self.stem['bn1'](s + a_i))
                    out_i = self.stem['maxpool'](out_i)
                    x_list.append(out_i)

        start_stage = max(1, self._last_shared_stage + 1) if did_shared_prefix else 1

        def run_one_adapter(x, adapter_id):
            for li in (1, 2, 3, 4):
                if li < start_stage:
                    continue
                layer = getattr(self, f'layer{li}')
                for bi, blk in enumerate(layer):
                    identity = x
                    if blk.downsample is not None:
                        identity = blk.downsample(x)

                    # conv1
                    s1 = blk.conv1(x)
                    key1 = f'layer{li}.{bi}.conv1'.replace('.', '_')
                    if self._share_block[li][bi]:
                        a1 = self.shared_adapters[key1](x)
                    else:
                        a1 = self.branch_adapters[adapter_id][key1](x) 
                    res = blk.relu(blk.bn1(s1 + a1))

                    # conv2
                    s2 = blk.conv2(res)
                    key2 = f'layer{li}.{bi}.conv2'.replace('.', '_')
                    if self._share_block[li][bi]:
                        a2 = self.shared_adapters[key2](res)
                    else:
                        a2 = self.branch_adapters[adapter_id][key2](res)
                    res = blk.bn2(s2 + a2)
                    x = blk.relu(res + identity)
            return x

        outputs = []
        for i in range(N):
            x_start = x_list[i]
            if use_checkpoint:
                outputs.append(checkpoint(lambda z: run_one_adapter(z, i), x_start))
            else:
                outputs.append(run_one_adapter(x_start, i))
        return outputs

    # Combined forward
    def forward(self, x, use_checkpoint: bool = False):
        # shared parts
        h_shared, did_shared_prefix = self._forward_shared_prefix_masked(x)

        # unshared parts
        feats_list = self._forward_suffix_batched_masked(
            h_shared if did_shared_prefix else x,
            did_shared_prefix,
            use_checkpoint=use_checkpoint
        )

        # heads
        all_y_preds, all_c_preds = [], []
        for i in range(self.num_models):
            pooled = F.adaptive_avg_pool2d(feats_list[i], (1, 1))
            flat   = torch.flatten(pooled, 1)
            head   = self.final_heads[i]

            flat   = head['norm'](flat)

            concept_preds = [fc(flat) for fc in head['concept_head']]
            concatenated  = torch.cat(concept_preds, dim=1) 
            y_pred        = head['classifier'](concatenated)

            all_c_preds.append(concept_preds)
            all_y_preds.append(y_pred)

        return all_y_preds, all_c_preds

    def _freeze_backbone_bn_eval(self, backbone):
        backbone.eval()
        for m in backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
                m.eval()
                if m.affine:
                    if m.weight is not None: m.weight.requires_grad_(False)
                    if m.bias   is not None: m.bias.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
                m.eval()
        return self

class LoraEnsembleshared(nn.Module):
    def _load_backbone_and_lora_targets(self, vit_model_name: str, *, backbone_pretrained: bool = True):
        if vit_model_name == "timm/vit_small_patch16_224.augreg_in21k_ft_in1k":
            if backbone_pretrained:
                try:
                    backbone = AutoModel.from_pretrained(vit_model_name, torch_dtype="auto", local_files_only=True)
                except TypeError:
                    backbone = AutoModel.from_pretrained(vit_model_name, torch_dtype="auto")
                except Exception:
                    backbone = AutoModel.from_pretrained(vit_model_name, torch_dtype="auto")
            else:
                try:
                    cfg = AutoConfig.from_pretrained(vit_model_name, local_files_only=True)
                except Exception:
                    cfg = AutoConfig.from_pretrained(vit_model_name)
                backbone = AutoModel.from_config(cfg)
            feature_dim = backbone.timm_model.embed_dim
            target_modules = ["qkv", "proj"]
            arch = "timm_vit"
            num_blocks = len(backbone.timm_model.blocks)
            return backbone, feature_dim, target_modules, arch, num_blocks

        if vit_model_name == "google/vit-base-patch16-224":
            if backbone_pretrained:
                try:
                    backbone = ViTModel.from_pretrained(vit_model_name, local_files_only=True)
                except TypeError:
                    backbone = ViTModel.from_pretrained(vit_model_name)
                except Exception:
                    backbone = ViTModel.from_pretrained(vit_model_name)
            else:
                try:
                    cfg = ViTConfig.from_pretrained(vit_model_name, local_files_only=True)
                except Exception:
                    cfg = ViTConfig.from_pretrained(vit_model_name)
                backbone = ViTModel(cfg)
            feature_dim = backbone.config.hidden_size
            target_modules = ["query", "key", "value", "out"]  # "out" matches "output.*"
            arch = "hf_vit"
            num_blocks = len(backbone.encoder.layer)
            return backbone, feature_dim, target_modules, arch, num_blocks

        # Two low-friction cases we support:
        #  - A ViTModel checkpoint (ViTModel.from_pretrained works)
        #  - A CLIP vision checkpoint (CLIPVisionModel.from_pretrained works)
        name_l = str(vit_model_name).lower()
        if "clip" in name_l:
            if backbone_pretrained:
                try:
                    backbone = CLIPVisionModel.from_pretrained(vit_model_name, local_files_only=True)
                except TypeError:
                    backbone = CLIPVisionModel.from_pretrained(vit_model_name)
                except Exception:
                    backbone = CLIPVisionModel.from_pretrained(vit_model_name)
            else:
                try:
                    cfg = CLIPVisionConfig.from_pretrained(vit_model_name, local_files_only=True)
                except Exception:
                    cfg = CLIPVisionConfig.from_pretrained(vit_model_name)
                backbone = CLIPVisionModel(cfg)
            feature_dim = backbone.config.hidden_size
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
            arch = "hf_clip_vision"
            num_blocks = len(backbone.vision_model.encoder.layers)
            return backbone, feature_dim, target_modules, arch, num_blocks

        if backbone_pretrained:
            try:
                backbone = ViTModel.from_pretrained(vit_model_name, local_files_only=True)
            except TypeError:
                backbone = ViTModel.from_pretrained(vit_model_name)
            except Exception:
                backbone = ViTModel.from_pretrained(vit_model_name)
        else:
            try:
                cfg = ViTConfig.from_pretrained(vit_model_name, local_files_only=True)
            except Exception:
                cfg = ViTConfig.from_pretrained(vit_model_name)
            backbone = ViTModel(cfg)
        feature_dim = backbone.config.hidden_size
        target_modules = ["query", "key", "value", "out"]
        arch = "hf_vit"
        num_blocks = len(backbone.encoder.layer)
        return backbone, feature_dim, target_modules, arch, num_blocks

    def __init__(
        self,
        num_models,
        num_classes,
        n_attributes,
        encoder,
        expand_dim=0,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_block_mask=None,
        backbone_pretrained: bool = True,
        **kwargs,
    ):
        #
        #google/vit-base-patch16-224
        if lora_block_mask!='111111111111':
            raise NotImplementedError("This model is not for unshared lora.")

        super(LoraEnsembleshared, self).__init__()
        if encoder not in ('vit', 'medical_vit'):
            raise NotImplementedError("This model is optimized for encoder in {'vit','medical_vit'}.")
        self.num_models = num_models  # number of loras
        model_name = MEDICAL_VIT_MODEL_NAME if encoder == "medical_vit" else VIT_DEFAULT_MODEL_NAME
        self.vit_model_name = model_name
        self.shared_backbone, feature_dim, base_target_modules, _arch, _num_blocks = (
            self._load_backbone_and_lora_targets(model_name, backbone_pretrained=bool(backbone_pretrained))
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=base_target_modules,
        )

        self.lora_models=None
        if num_models<2:
            raise  NotImplementedError("This model is not for single lora.")
        self.lora_models = LoraModel(self.shared_backbone, lora_config, 'lora_shared')
            
                
        print(f"Created and loaded lora_shared")

        self.lora_models.set_adapter('lora_shared')
        self.final_heads = nn.ModuleList([
            nn.ModuleDict({
                'concept_head': self._create_concept_head(feature_dim, n_attributes, expand_dim),
                'classifier': MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
            }) for _ in range(num_models)
        ])
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _create_concept_head(self, input_dim, n_attributes, expand_dim):
        head = nn.ModuleList()
        for _ in range(n_attributes):
            head.append(MLP(input_dim, 1, expand_dim))
        return head

    def forward(self, x):
        all_y_preds, all_c_preds = [], []
        outputs = self.lora_models(x)
        features = outputs.last_hidden_state[:,0]
        for i in range(self.num_models):
            head = self.final_heads[i]
            concept_preds = [fc(features) for fc in head['concept_head']]
            all_c_preds.append(concept_preds)
            concatenated_concepts = torch.cat(concept_preds, dim=1)
            y_pred = head['classifier'](concatenated_concepts)
            all_y_preds.append(y_pred)
        return all_y_preds, all_c_preds

    def train(self, mode=True):
        super().train(mode)
        if self.shared_backbone is not None:
            self.shared_backbone.eval()
        return self
    def model_x_to_c(self,x,i):
        outputs = self.lora_models(x)
        features = outputs.last_hidden_state[:,0]
        head = self.final_heads[i]
        concept_preds = [fc(features) for fc in head['concept_head']]
        return concept_preds
    def model_c_to_y(self,x,i):
        head = self.final_heads[i]
        y_pred = head['classifier'](x)
        return y_pred
    def save_shared_adapter(self,out_dir):
        self.lora_models.set_adapter('lora_shared')
        self.lora_models.save_pretrained(os.path.join(out_dir,'lora_shared'))
        return
    def save_adapter_cbm(self,i,out_dir):
        os.makedirs(os.path.join(out_dir, f'lora_{i}'), exist_ok=True)
        torch.save(self.final_heads[i].state_dict(), os.path.join(out_dir, f'lora_{i}','final_head.pth'))
        return

class LoraEnsemble(nn.Module):
    def __init__(
        self,
        num_models,
        num_classes,
        n_attributes,
        encoder,
        expand_dim=0,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_block_mask=None,
        backbone_pretrained: bool = True,
        **kwargs,
    ):
        #google/vit-base-patch16-224
        if lora_block_mask!='000000000000':
            raise NotImplementedError("This model is not for shared lora.")
        super(LoraEnsemble, self).__init__()
        if encoder not in ('vit', 'medical_vit'):
            raise NotImplementedError("This model is optimized for encoder in {'vit','medical_vit'}.")
        self.num_models = num_models  # number of loras
        model_name = MEDICAL_VIT_MODEL_NAME if encoder == "medical_vit" else VIT_DEFAULT_MODEL_NAME
        self.vit_model_name = model_name
        self.shared_backbone, feature_dim, base_target_modules, _arch, _num_blocks = (
            LoraEnsembleshared._load_backbone_and_lora_targets(
                self, model_name, backbone_pretrained=bool(backbone_pretrained)
            )
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=base_target_modules,
        )

        self.lora_models=None
        if num_models<2:
            raise  NotImplementedError("This model is not for single lora.")
        self.adapter_set=[f"lora_{i}"for i in range(num_models)]
  
        for i in range(num_models):
            adapter_name = self.adapter_set[i]
            
            if self.lora_models is None:
                # self.lora_models = PeftMixedModel(self.shared_backbone, lora_config, adapter_name=adapter_name)
                self.lora_models = LoraModel(self.shared_backbone, lora_config, adapter_name)
            
            else:
                self.lora_models.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
                
            print(f"Created and loaded {adapter_name}")

        self.lora_models.set_adapter(self.adapter_set[0])
        self.final_heads = nn.ModuleList([
            nn.ModuleDict({
                'concept_head': self._create_concept_head(feature_dim, n_attributes, expand_dim),
                'classifier': MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
            }) for _ in range(num_models)
        ])
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _create_concept_head(self, input_dim, n_attributes, expand_dim):
        head = nn.ModuleList()
        for _ in range(n_attributes):
            head.append(MLP(input_dim, 1, expand_dim))
        return head


    def forward(self, x):
        all_y_preds, all_c_preds = [], []
        for i in range(self.num_models):
            self.lora_models.set_adapter(self.adapter_set[i])
            outputs = self.lora_models(x)
            features = outputs.last_hidden_state[:,0]
            head = self.final_heads[i]
            concept_preds = [fc(features) for fc in head['concept_head']]
            all_c_preds.append(concept_preds)
            concatenated_concepts = torch.cat(concept_preds, dim=1)
            y_pred = head['classifier'](concatenated_concepts)
            all_y_preds.append(y_pred)
        self.lora_models.set_adapter(self.adapter_set[0])
        if self.training:
            for param in self.lora_models.parameters():
                if not param.requires_grad and param.ndim > 0:
                    param.requires_grad = True
        return all_y_preds, all_c_preds


    def train(self, mode=True):
        super().train(mode)
        if self.shared_backbone is not None:
            self.shared_backbone.eval()
        return self
    def model_x_to_c(self,x,i):
        outputs = self.lora_models(x)
        features = outputs.last_hidden_state[:,0]
        head = self.final_heads[i]
        concept_preds = [fc(features) for fc in head['concept_head']]
        return concept_preds
    def model_c_to_y(self,x,i):
        head = self.final_heads[i]
        y_pred = head['classifier'](x)
        return y_pred
    def save_adapter_cbm(self,i,out_dir):
        self.lora_models.set_adapter(f"lora_{i}")
        self.lora_models.save_pretrained(os.path.join(out_dir,f"lora_{i}"))
        torch.save(self.final_heads[i].state_dict(), os.path.join(out_dir, f'lora_{i}','final_head.pth'))
        return

class LoraEnsemble_gc(LoraEnsemble):
    def forward(self, x):
        def _run_single_adapter(x_input, adapter_idx):
            self.lora_models.set_adapter(self.adapter_set[adapter_idx])

            outputs = self.lora_models(x_input)
            features = outputs.last_hidden_state[:, 0]

            head = self.final_heads[adapter_idx]
            concept_preds = [fc(features) for fc in head['concept_head']]
            concatenated_concepts = torch.cat(concept_preds, dim=1)
            y_pred = head['classifier'](concatenated_concepts)

            self.lora_models.set_adapter(self.adapter_set[0])
            return y_pred, concept_preds

        all_y_preds, all_c_preds = [], []
        for i in range(self.num_models):
            y_pred, concept_preds = checkpoint(
                _run_single_adapter, x, i, use_reentrant=False
            )
            
            all_y_preds.append(y_pred)
            all_c_preds.append(concept_preds)
        
        self.lora_models.set_adapter(self.adapter_set[0])
        
        return all_y_preds, all_c_preds

class LoraEnsemblePshared(nn.Module):
    def __init__(
        self,
        num_models,
        num_classes,
        n_attributes,
        encoder,
        expand_dim=0,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_block_mask=None,
        backbone_pretrained: bool = True,
        **kwargs,
    ):
        #google/vit-base-patch16-224
        super(LoraEnsemblePshared, self).__init__()
        if encoder not in ('vit', 'medical_vit'):
            raise NotImplementedError("This model is optimized for encoder in {'vit','medical_vit'}.")
        self.num_models = num_models  # number of loras
        model_name = MEDICAL_VIT_MODEL_NAME if encoder == "medical_vit" else VIT_DEFAULT_MODEL_NAME
        self.vit_model_name = model_name
        if lora_block_mask is not None:
            lora_target_blocks = [i for i, bit in enumerate(lora_block_mask) if bit == '0']
            lora_shared_blckes = [i for i, bit in enumerate(lora_block_mask) if bit == '1']
            #0 add lora 1 don't add lora 
            print(f"Parsed target block index {lora_target_blocks} from mask '{lora_block_mask}'")
        else:
            raise ValueError("lora_block_mask must be provided, e.g., '1110000000' for vit-small.")
        if len(lora_shared_blckes)==0 or len(lora_target_blocks)==0:
            raise ValueError("lora_block_mask must contain at least one '1' and one '0'. If you want to share all blocks or have no shared blocks, consider using LoraEnsemble or LoraEnsembleshared instead.")
        
        shared_modules = [] # which modules to share LoRA across all models
        target_modules = [] # which modules to apply LoRA to
        self.shared_backbone, feature_dim, _base_target_modules, arch, num_blocks = (
            LoraEnsembleshared._load_backbone_and_lora_targets(
                self, model_name, backbone_pretrained=bool(backbone_pretrained)
            )
        )

        def _extend_block_modules(dst: list, i: int):
            if arch == "timm_vit":
                dst.extend([f"timm_model.blocks.{i}.attn.qkv", f"timm_model.blocks.{i}.attn.proj"])
                return
            if arch == "hf_vit":
                dst.extend([
                    f"encoder.layer.{i}.attention.attention.query",
                    f"encoder.layer.{i}.attention.attention.key",
                    f"encoder.layer.{i}.attention.attention.value",
                    f"encoder.layer.{i}.attention.output.dense",
                ])
                return
            if arch == "hf_clip_vision":
                dst.extend([
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj",
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj",
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj",
                    f"vision_model.encoder.layers.{i}.self_attn.out_proj",
                ])
                return
            raise NotImplementedError(f"Unsupported backbone arch for partial-sharing LoRA: {arch}")

        for i in lora_target_blocks:
            if i >= num_blocks:
                raise ValueError(f"Block index {i} is out of range. Model has {num_blocks} blocks.")
            _extend_block_modules(target_modules, i)

        for i in lora_shared_blckes:
            if i >= num_blocks:
                raise ValueError(f"Block index {i} is out of range. Model has {num_blocks} blocks.")
            _extend_block_modules(shared_modules, i)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        lora_config_shared = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=shared_modules,
        )

        self.lora_models=None
        if num_models<2:
            raise  NotImplementedError("This model is not for single lora.")
        self.adapter_set=[f"lora_{i}"for i in range(num_models)]
        self.adapter_set.append("lora_shared")
        #add a shared lora
        for i in range(num_models):
            adapter_name = self.adapter_set[i]
            
            if self.lora_models is None:
                # self.lora_models = PeftMixedModel(self.shared_backbone, lora_config, adapter_name=adapter_name)
                self.lora_models = LoraModel(self.shared_backbone, lora_config, adapter_name)
            
            else:
                self.lora_models.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
                
            print(f"Created and loaded {adapter_name}")
        self.lora_models.add_adapter(adapter_config=lora_config_shared, adapter_name='lora_shared')
        self.lora_models.set_adapter(self.adapter_set)
        self.final_heads = nn.ModuleList([
            nn.ModuleDict({
                'concept_head': self._create_concept_head(feature_dim, n_attributes, expand_dim),
                'classifier': MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
            }) for _ in range(num_models)
        ])
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _create_concept_head(self, input_dim, n_attributes, expand_dim):
        head = nn.ModuleList()
        for _ in range(n_attributes):
            head.append(MLP(input_dim, 1, expand_dim))
        return head

    def forward(self, x):
        all_y_preds, all_c_preds = [], []
        for i in range(self.num_models):
            self.lora_models.set_adapter([self.adapter_set[i]]+['lora_shared'])
            outputs = self.lora_models(x)
            features = outputs.last_hidden_state[:,0]
            head = self.final_heads[i]
            concept_preds = [fc(features) for fc in head['concept_head']]
            all_c_preds.append(concept_preds)
            concatenated_concepts = torch.cat(concept_preds, dim=1)
            y_pred = head['classifier'](concatenated_concepts)
            all_y_preds.append(y_pred)
        self.lora_models.set_adapter(self.adapter_set)
        return all_y_preds, all_c_preds
    
    def train(self, mode=True):
        super().train(mode)
        if self.shared_backbone is not None:
            self.shared_backbone.eval()
        return self
    def model_x_to_c(self,x,i):
        outputs = self.lora_models(x)
        features = outputs.last_hidden_state[:,0]
        head = self.final_heads[i]
        concept_preds = [fc(features) for fc in head['concept_head']]
        return concept_preds
    def model_c_to_y(self,x,i):
        head = self.final_heads[i]
        y_pred = head['classifier'](x)
        return y_pred
    def save_adapter_cbm(self,i,out_dir):
        self.lora_models.set_adapter(f"lora_{i}")
        self.lora_models.save_pretrained(os.path.join(out_dir,f"lora_{i}"))
        torch.save(self.final_heads[i].state_dict(), os.path.join(out_dir, f'lora_{i}','final_head.pth'))
        return
    def save_shared_adapter(self,out_dir):
        self.lora_models.set_adapter('lora_shared')
        self.lora_models.save_pretrained(os.path.join(out_dir,'lora_shared'))
        return

class random_vit(nn.Module):
    def __init__(self, num_classes, n_attributes, encoder='vit',
                 vit_model_name="google/vit-base-patch16-224",
                 expand_dim=0, concept_dropout=0.0):
        super().__init__()
        self.vit_model_name = vit_model_name

        if vit_model_name == "timm/vit_small_patch16_224.augreg_in21k_ft_in1k":
            self.backbone = AutoModel.from_pretrained(vit_model_name)
            feature_dim = self.backbone.timm_model.embed_dim
        elif vit_model_name == "google/vit-base-patch16-224":
            self.backbone = ViTModel.from_pretrained(vit_model_name)
            feature_dim = self.backbone.config.hidden_size
        else:
            raise NotImplementedError("Only enable vit-small(timm) and vit-base(google)")

        self.concept_head = nn.ModuleList([
            MLP(feature_dim, 1, expand_dim, concept_dropout)
            for _ in range(n_attributes)
        ])
        self.classifier = MLP(n_attributes, num_classes, expand_dim, 0.0)

        self.enable_grad_ckpt = False

    def set_gradient_checkpointing(self, enabled: bool = True):
        self.enable_grad_ckpt = enabled
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            if enabled:
                self.backbone.gradient_checkpointing_enable()
            else:
                self.backbone.gradient_checkpointing_disable()
        if hasattr(self.backbone, "timm_model") and hasattr(self.backbone.timm_model, "set_grad_checkpointing"):
            self.backbone.timm_model.set_grad_checkpointing(enabled)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        features = outputs.last_hidden_state[:, 0]
        c_preds = [fc(features) for fc in self.concept_head]
        c_vec = torch.cat(c_preds, dim=1)
        y_pred = self.classifier(c_vec)
        return y_pred, c_preds

    def save_model(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "model.pt"))
        meta = {
            "vit_model_name": self.vit_model_name,
            "num_concepts": len(self.concept_head)
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_model(out_dir: str, num_classes, n_attributes,
                   encoder='vit', expand_dim=0, concept_dropout=0.0, map_location=None):
        with open(os.path.join(out_dir, "meta.json")) as f:
            meta = json.load(f)
        m = random_vit(num_classes=num_classes,
                               n_attributes=n_attributes,
                               encoder=encoder,
                               vit_model_name=meta["vit_model_name"],
                               expand_dim=expand_dim,
                               concept_dropout=concept_dropout)
        sd = torch.load(os.path.join(out_dir, "model.pt"), map_location=map_location)
        m.load_state_dict(sd, strict=True)
        return m

class RevolverWrapper(torch.nn.Module):
    def __init__(self,
                 baseline: torch.nn.Module,
                 targets: List[str],
                 cylinders: int,
                 donor: torch.nn.Module | None = None):
        super().__init__()

        self.model = baseline
        self.targets = targets

        self.cylinders = cylinders
        self.current_cylinder = 0

        self.donor = donor if donor else baseline

        self._generate_cylinders()

    def _generate_cylinders(self):
        for parameter in self.donor.parameters():
            parameter.requires_grad = False

        for name, submodule in self.donor.named_modules():
            if name in self.targets:
                for parameter in submodule.parameters():
                    parameter.requires_grad = True

        if type(self.cylinders) is not int:
            return self.cylinders
        else:
            assert type(self.cylinders) is int

            cylinders = []

            # Optimization to not copy a first cylinder
            first_cylinder = {}
            for target in self.targets:
                first_cylinder[target] = self.donor.get_submodule(target)
            cylinders.append(first_cylinder)

            for _ in range(self.cylinders - 1):
                cylinder = {}
                for target in self.targets:
                    # TODO: Check this segment is correctly creating separate, detached copies
                    cylinder[target] = deepcopy(self.donor.get_submodule(target))
                    for parameter in cylinder[target].parameters():
                        parameter.detach()
                        parameter.requires_grad = True

                cylinders.append(cylinder)

            self.cylinders = cylinders

    def _generate_trunk(self):
        """
        TODO: Implement (either user-specified or generate by tracing a forward pass, record when the frontier hits 
        something in the targets list), generally just a speed-up, difficult to do in general
        """
        pass

    def forward(self, *args, cylinder_index: int | None = None, **kwargs):
        if cylinder_index != self.current_cylinder:
            self.cycle(cylinder_index)

        return self.model(*args, **kwargs)
        

    def cycle(self, index: int | None = None):
        if index not in range(len(self.cylinders)):
            return
        
        if index == self.current_cylinder:
            return
        
        for target in self.targets:
            self.model.set_submodule(target, self.cylinders[index][target])

        self.current_cylinder = index

class RevolverEnsemble(torch.nn.Module):
    def __init__(self,
                 n_models: int,
                 n_classes: int,
                 n_attributes: int,
                 n_features: int,
                 baseline: torch.nn.Module,
                 targets: List[str],
                 donor: torch.nn.Module | None = None,
                 expand_dim: int = 0,
                 encoder: str = "resnet18"):
        """
        TODO: Add a mode for using MLP (expanding/constricting layer heads, as well as unshared-concept heads)
        """

        super().__init__()

        self.n_models = n_models
        self.n_classes = n_classes
        self.n_attributes = n_attributes
        self.n_features = n_features

        assert encoder in ["resnet18", "vit", "medical_vit"], f"Unsupported encoder: {encoder}"
        self.encoder = encoder

        self.ensemble = RevolverWrapper(
            baseline,
            targets,
            cylinders=n_models,
            donor=donor
        )

        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "concept_head": self._create_concept_head(self.n_features, self.n_attributes, expand_dim),
                "class_head": MLP(self.n_attributes, self.n_classes, expand_dim)
            }) for _ in range(self.n_models)
        ])

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.eval()

    def _create_concept_head(self, input_dim, n_attributes, expand_dim):
        head = nn.ModuleList()
        for _ in range(n_attributes):
            head.append(MLP(input_dim, 1, expand_dim))
        return head
    
    def forward(self, x, use_checkpoint: bool = False):
        """
        TODO: Add checkpointing support?
        """

        checkpoint_dummy = torch.nn.Identity() if not use_checkpoint else lambda x : checkpoint(x)

        all_y_preds, all_c_preds = [], []
        for i in range(self.n_models):
            raw_outputs = checkpoint_dummy(self.ensemble(x, cylinder_index=i))
            if self.encoder in ("vit", "medical_vit"):
                raw_outputs = raw_outputs.last_hidden_state[:, 0]

            head = self.heads[i]

            concept_preds = [fc(raw_outputs) for fc in head["concept_head"]]
            y_pred = head["class_head"](torch.cat(concept_preds, dim=1))

            all_c_preds.append(concept_preds)
            all_y_preds.append(y_pred)

        return all_y_preds, all_c_preds
    
    # TODO: Not entirely sure this (or _freeze_backbone_bn_eval) is particularly necessary?
    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
                m.eval()
        return self
    
class DropoutWrapper(torch.nn.Module):
    def __init__(
        self,
        baseline: torch.nn.Module,
        targets: List[str],
        dropouts: float | Dict[str, float] | None = 1e-1,
    ):
        super().__init__()

        self.baseline = baseline
        self.targets = targets
        self.dropouts = 1e-1 if dropouts is None else dropouts

        self.dropping = True
        self.add_hooks()

    def add_hooks(self):
        self.dropping = True
        self.dropout_hooks = {}

        for target in self.targets:
            if isinstance(self.dropouts, (float, int)):
                target_dropout = float(self.dropouts)
            elif isinstance(self.dropouts, dict):
                if target not in self.dropouts:
                    raise KeyError(
                        f"Missing dropout prob for target '{target}'. "
                        f"Got keys: {sorted(self.dropouts.keys())}"
                    )
                target_dropout = float(self.dropouts[target])
            else:
                raise TypeError(
                    f"dropouts must be float|int|dict|None, got {type(self.dropouts).__name__}"
                )

            if not (0.0 <= target_dropout <= 1.0):
                raise ValueError(f"dropout prob must be in [0,1], got {target_dropout} for target '{target}'")

            def BernoulliDropout(layer, input, output, p=target_dropout):
                # Some HF blocks (e.g., CLIP encoder layers) return tuples like
                # (hidden_states, attn_weights?). Apply dropout to hidden_states only.
                if torch.is_tensor(output):
                    return torch.nn.functional.dropout(output, p=p)
                if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                    return (torch.nn.functional.dropout(output[0], p=p),) + tuple(output[1:])
                if isinstance(output, list) and output and torch.is_tensor(output[0]):
                    output = list(output)
                    output[0] = torch.nn.functional.dropout(output[0], p=p)
                    return output
                return output

            # NOTE: Keeping support for soft dropouts, not used for now
            def GaussianDropout(layer, input, output, p=target_dropout):
                if torch.is_tensor(output):
                    return output * (torch.randn_like(output) * p + 1)
                if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                    first = output[0] * (torch.randn_like(output[0]) * p + 1)
                    return (first,) + tuple(output[1:])
                if isinstance(output, list) and output and torch.is_tensor(output[0]):
                    output = list(output)
                    output[0] = output[0] * (torch.randn_like(output[0]) * p + 1)
                    return output
                return output

            self.dropout_hooks[target] = self.baseline.get_submodule(target).register_forward_hook(BernoulliDropout)

    def remove_hooks(self):
        for hook in self.dropout_hooks.values():
            hook.remove()
        self.dropping = False
        self.dropout_hooks = {}

    def is_dropping(self):
        return self.dropping

    def forward(self, *args, **kwargs):
        return self.baseline(*args, **kwargs)


class DropoutEnsemble(torch.nn.Module):
    def __init__(self,
                 n_models: int,
                 n_classes: int,
                 n_attributes: int,
                 n_features: int,
                 baseline: torch.nn.Module,
                 targets: List[str],
                 dropouts: float | Dict[str, float] | None = 1e-1,
                 expand_dim: int = 0,
                 encoder: str = "resnet18",
                 passthrough: bool = False):

        super().__init__()

        self.n_models = n_models
        self.n_classes = n_classes
        self.n_attributes = n_attributes
        self.n_features = n_features

        assert encoder in ["resnet18", "vit", "medical_vit"], f"Unsupported encoder: {encoder}"
        self.encoder = encoder

        self.targets = targets
        self.dropouts = 1e-1 if dropouts is None else dropouts
        self.passthrough = passthrough

        self.ensemble = DropoutWrapper(
            baseline,
            targets,
            dropouts=dropouts
        )

        self.masking = False
        self.masks = [{} for _ in range(self.n_models)]
        self.mask_hooks = [{} for _ in range(self.n_models)]

        self.curr_cylinder = None

        if self.passthrough:
            self.heads = nn.ModuleDict()
        else:
            self.heads = nn.ModuleDict({
                "concept_head": self._create_concept_head(self.n_features, self.n_attributes, expand_dim),
                "class_head": MLP(self.n_attributes, self.n_classes, expand_dim),
            })

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.eval()

    def _create_concept_head(self, input_dim, n_attributes, expand_dim):
        head = nn.ModuleList()
        for _ in range(n_attributes):
            head.append(MLP(input_dim, 1, expand_dim))
        return head

    def add_masks(self, cylinder: int = 0):
        assert cylinder >= 0
        assert cylinder < self.n_models

        if cylinder == self.curr_cylinder:
            return

        for target in self.targets:
            if isinstance(self.dropouts, (float, int)):
                target_dropout = float(self.dropouts)
            elif isinstance(self.dropouts, dict):
                if target not in self.dropouts:
                    raise KeyError(
                        f"Missing dropout prob for target '{target}'. "
                        f"Got keys: {sorted(self.dropouts.keys())}"
                    )
                target_dropout = float(self.dropouts[target])
            else:
                raise TypeError(
                    f"dropouts must be float|int|dict|None, got {type(self.dropouts).__name__}"
                )

            if not (0.0 <= target_dropout <= 1.0):
                raise ValueError(f"dropout prob must be in [0,1], got {target_dropout} for target '{target}'")

            def BernoulliMask(layer, input, output, i, curr_target, p: float):
                if torch.is_tensor(output):
                    hidden = output
                    wrapper = None
                elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                    hidden = output[0]
                    wrapper = ("tuple", output[1:])
                elif isinstance(output, list) and output and torch.is_tensor(output[0]):
                    hidden = output[0]
                    wrapper = ("list", None)
                else:
                    return output

                if curr_target not in self.masks[i].keys():
                    mask_shape = hidden.shape[1:] if hidden.dim() > 1 else hidden.shape
                    renormalization = 1.0 / (1.0 - p)
                    self.masks[i][curr_target] = (torch.rand(size=mask_shape, device=hidden.device) > p).type_as(hidden) * renormalization

                masked = hidden * self.masks[i][curr_target]
                if wrapper is None:
                    return masked
                if wrapper[0] == "tuple":
                    return (masked,) + tuple(wrapper[1])
                if wrapper[0] == "list":
                    out_list = list(output)
                    out_list[0] = masked
                    return out_list
                return output

            def GaussianMask(layer, input, output, i, curr_target, p: float):
                if torch.is_tensor(output):
                    hidden = output
                    wrapper = None
                elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                    hidden = output[0]
                    wrapper = ("tuple", output[1:])
                elif isinstance(output, list) and output and torch.is_tensor(output[0]):
                    hidden = output[0]
                    wrapper = ("list", None)
                else:
                    return output

                if curr_target not in self.masks[i].keys():
                    mask_shape = hidden.shape[1:] if hidden.dim() > 1 else hidden.shape
                    self.masks[i][curr_target] = torch.randn(size=mask_shape, device=hidden.device).type_as(hidden)

                masked = hidden * (self.masks[i][curr_target] * p + 1)
                if wrapper is None:
                    return masked
                if wrapper[0] == "tuple":
                    return (masked,) + tuple(wrapper[1])
                if wrapper[0] == "list":
                    out_list = list(output)
                    out_list[0] = masked
                    return out_list
                return output
            
            def make_bernoulli(i, curr_target, p: float):
                return lambda layer, input, output, p=p: BernoulliMask(layer, input, output, i, curr_target, p)
            
            def make_gaussian(i, curr_target, p: float):
                return lambda layer, input, output, p=p: GaussianMask(layer, input, output, i, curr_target, p)
            
            self.mask_hooks[cylinder][target] = \
                self.ensemble.baseline.get_submodule(target).register_forward_hook(
                    make_bernoulli(cylinder, target, target_dropout)
                )

        self.curr_cylinder = cylinder
        self.masking = True

    def remove_masks(self):
        for mask_hook_set in self.mask_hooks:
            for hook in mask_hook_set.values():
                hook.remove()
        self.masking = False
        self.mask_hooks = [{} for _ in range(self.n_models)]

        self.curr_cylinder = None
    
    def forward(self, x, use_checkpoint: bool = False):

        checkpoint_dummy = torch.nn.Identity() if not use_checkpoint else lambda x : checkpoint(x)

        if self.passthrough:
            if not self.masking:
                y_pred, concept_preds = checkpoint_dummy(self.ensemble(x, is_training=self.training))
                return y_pred, concept_preds

            with torch.no_grad():
                all_y_preds, all_c_preds = [], []
                for i in range(self.n_models):
                    self.remove_masks()
                    self.add_masks(i)
                    y_pred, concept_preds = checkpoint_dummy(self.ensemble(x, is_training=self.training))
                    all_y_preds.append(y_pred)
                    all_c_preds.append(concept_preds)
            return all_y_preds, all_c_preds

        if not self.masking:
            raw_outputs = checkpoint_dummy(self.ensemble(x))
            if self.encoder in ("vit", "medical_vit"):
                raw_outputs = raw_outputs.last_hidden_state[:, 0]

            concept_preds = [fc(raw_outputs) for fc in self.heads["concept_head"]]
            y_pred = self.heads["class_head"](torch.cat(concept_preds, dim=1))

            return y_pred, concept_preds

        else:
            with torch.no_grad():
                all_y_preds, all_c_preds = [], []
                for i in range(self.n_models):
                    # self.ensemble.remove_hooks()
                    self.remove_masks()
                    self.add_masks(i)

                    raw_outputs = checkpoint_dummy(self.ensemble(x))
                    if self.encoder in ("vit", "medical_vit"):
                        raw_outputs = raw_outputs.last_hidden_state[:, 0]

                    concept_preds = [fc(raw_outputs) for fc in self.heads["concept_head"]]
                    y_pred = self.heads["class_head"](torch.cat(concept_preds, dim=1))

                    all_c_preds.append(concept_preds)
                    all_y_preds.append(y_pred)

            return all_y_preds, all_c_preds
    
    def train(self, mode: bool = True):
        if mode:
            if not self.ensemble.is_dropping():
                self.ensemble.add_hooks()
            if self.masking:
                self.remove_masks()
        else:
            if self.ensemble.is_dropping():
                self.ensemble.remove_hooks()
            self.masking = True

        super().train(mode)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
                m.eval()
        return self 
