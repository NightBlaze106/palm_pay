"""
MobileNetV3-Small Palm Embedder Architecture.
Emits 512-D L2-normalized float32 embeddings matching Phase-2 DeepPalmModel specification.
Can run under PyTorch when available, or load/export to ONNX.
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = object


if TORCH_AVAILABLE:
    class MobileNetV3PalmEmbedder(nn.Module):
        """
        Lightweight MobileNetV3-Small backbone adapted for 128x128 1-channel or 3-channel
        palm ROI input, projecting onto a 512-D unit-normalized hypersphere.
        """

        def __init__(
            self,
            feature_dim: int = 512,
            in_channels: int = 3,
            pretrained: bool = True,
        ):
            super().__init__()
            self.feature_dim = feature_dim
            self.in_channels = in_channels

            # Load MobileNetV3-Small backbone
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)

            # Adapt first conv layer if 1-channel grayscale input
            if in_channels == 1:
                first_conv = base.features[0][0]
                new_conv = nn.Conv2d(
                    1,
                    first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=False,
                )
                if pretrained:
                    # Average original RGB weights across channel dim
                    with torch.no_grad():
                        new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                base.features[0][0] = new_conv

            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            # Projection head: 576 -> 512 L2-norm
            in_features = 576
            self.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feature_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Input: (B, C, H, W) normalized to [-1, 1]
            Output: (B, feature_dim) unit-normalized
            """
            feat = self.features(x)
            pooled = self.avgpool(feat)
            flattened = torch.flatten(pooled, 1)
            raw_emb = self.head(flattened)
            # L2 Normalization onto unit hypersphere
            normalized = F.normalize(raw_emb, p=2, dim=1)
            return normalized

    class ArcFaceLoss(nn.Module):
        """
        ArcFace (Additive Angular Margin Loss) for Palm Biometric Metric Learning.
        Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
        """

        def __init__(self, num_classes: int, embedding_dim: int = 512, s: float = 30.0, m: float = 0.50):
            super().__init__()
            self.num_classes = num_classes
            self.embedding_dim = embedding_dim
            self.s = s
            self.m = m
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
            nn.init.xavier_uniform_(self.weight)

            self.cos_m = np.cos(m)
            self.sin_m = np.sin(m)
            self.th = np.cos(np.pi - m)
            self.mm = np.sin(np.pi - m) * m

        def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # Cosine similarity between normalized embeddings and normalized class centers
            cosine = F.linear(embeddings, F.normalize(self.weight, p=2, dim=1))
            sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(), device=embeddings.device)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            return F.cross_entropy(output, labels)

else:
    class MobileNetV3PalmEmbedder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MobileNetV3PalmEmbedder. Install requirements-train.txt.")
