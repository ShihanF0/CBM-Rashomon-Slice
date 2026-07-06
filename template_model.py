import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoConfig, AutoModel, ViTConfig, ViTModel, CLIPVisionConfig, CLIPVisionModel

__all__ = ['MLP', 'FC', 'Adapter', 'resnet18', 'vit']

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation = torch.nn.ReLU()
            self.linear2 = nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation(x)
            x = self.linear2(x)
        return x

class FC(nn.Module):
    def __init__(self, input_dim, output_dim, expand_dim, stddev=None):
        """
        Extend standard Torch Linear layer to include the option of expanding into 2 Linear layers
        """
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = nn.ReLU()
            self.fc_new = nn.Linear(input_dim, expand_dim)
            self.fc = nn.Linear(expand_dim, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x
    
class Adapter(nn.Module):
    """
    A small adapter module with 2 1*1 convolutions.
    """
    def __init__(self, in_channels, out_channels, bottleneck_dim, stride=1):
        super(Adapter, self).__init__()

        self.adapter_downsample = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

        self.adapter_path = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, bias=False)
        )

        nn.init.constant_(self.adapter_path[-1].weight, 0)
        if self.adapter_path[-1].bias is not None:
             nn.init.constant_(self.adapter_path[-1].bias, 0)

    def forward(self, x):
        x_downsampled = self.adapter_downsample(x)
        return self.adapter_path(x_downsampled)
    
class ResNet18(nn.Module):
    def __init__(self, n_attributes, bottleneck=True, expand_dim=0, **kwargs):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        feature_dim = self.model.fc.in_features
        
        self.model.fc = nn.Identity()
        
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.aux_logits = False

        self.all_fc = nn.ModuleList()
        if self.bottleneck:
            for _ in range(self.n_attributes):
                self.all_fc.append(FC(input_dim=feature_dim, output_dim=1, expand_dim=expand_dim))

    def forward(self, x):
        features = self.model(x)
        out = []
        for fc in self.all_fc:
            out.append(fc(features))
          
        return out

def resnet18(pretrained, freeze, **kwargs):
    """
    Args:
        pretrained (bool): if true, load weights on pretrained resnet18
        freeze (bool): of true, freeze weights of the backbone and only train the newly-added FC layers
    """
    if not pretrained:
        print("Warning: This implementation defaults to using pretrained weights for ResNet18.")

    model = ResNet18(**kwargs)
    
    if freeze:
        for param in model.model.parameters():
            param.requires_grad = False
    return model

# --- ViT Backbone ---
class ViTBasePatch16_224(nn.Module):
    def __init__(self, n_attributes, bottleneck=True, expand_dim=0, pretrained: bool = True, **kwargs):
        super(ViTBasePatch16_224, self).__init__()
        if pretrained:
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        else:
            cfg = ViTConfig.from_pretrained("google/vit-base-patch16-224")
            self.model = ViTModel(cfg)
        feature_dim = self.model.config.hidden_size

        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.aux_logits = False

        self.all_fc = nn.ModuleList()
        if self.bottleneck:
            for _ in range(self.n_attributes):
                self.all_fc.append(FC(input_dim=feature_dim, output_dim=1, expand_dim=expand_dim))

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0]

        out = []
        for fc in self.all_fc:
            out.append(fc(features))
        return out

class ViTSmallPatch16_224(nn.Module):
    def __init__(self, n_attributes, bottleneck=True, expand_dim=0, pretrained: bool = True, **kwargs):
        super(ViTSmallPatch16_224, self).__init__()
        name = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
        # TimmWrapperModel doesn't accept dtype kwargs in some versions; keep defaults.
        if pretrained:
            self.model = AutoModel.from_pretrained(name)
        else:
            cfg = AutoConfig.from_pretrained(name)
            self.model = AutoModel.from_config(cfg)
        feature_dim = self.model.timm_model.embed_dim
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.aux_logits = False

        self.all_fc = nn.ModuleList()
        if self.bottleneck:
            for _ in range(self.n_attributes):
                self.all_fc.append(FC(input_dim=feature_dim, output_dim=1, expand_dim=expand_dim))

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0]

        out = []
        for fc in self.all_fc:
            out.append(fc(features))
        return out


class CLIPViTBasePatch32(nn.Module):
    def __init__(self, n_attributes, bottleneck=True, expand_dim=0, model_name: str = "openai/clip-vit-base-patch32", pretrained: bool = True, **kwargs):
        super(CLIPViTBasePatch32, self).__init__()
        if pretrained:
            self.model = CLIPVisionModel.from_pretrained(model_name)
        else:
            cfg = CLIPVisionConfig.from_pretrained(model_name)
            self.model = CLIPVisionModel(cfg)
        feature_dim = self.model.config.hidden_size

        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.aux_logits = False

        self.all_fc = nn.ModuleList()
        if self.bottleneck:
            for _ in range(self.n_attributes):
                self.all_fc.append(FC(input_dim=feature_dim, output_dim=1, expand_dim=expand_dim))

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]
        out = []
        for fc in self.all_fc:
            out.append(fc(features))
        return out


def vit(pretrained, freeze, **kwargs):
    model_name = kwargs.get('model_name', '')
    if 'clip' in str(model_name).lower():
        model = CLIPViTBasePatch32(pretrained=bool(pretrained), **kwargs)
    elif model_name == 'timm/vit_small_patch16_224.augreg_in21k_ft_in1k':
        model = ViTSmallPatch16_224(pretrained=bool(pretrained), **kwargs)
    elif model_name == 'google/vit-base-patch16-224':
        model = ViTBasePatch16_224(pretrained=bool(pretrained), **kwargs)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    if freeze:
        for p in model.model.parameters():
            p.requires_grad = False
    return model

