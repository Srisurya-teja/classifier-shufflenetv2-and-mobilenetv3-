"""
mobilenetv3_model.py

Trainable MobileNetV3 Small backbone fine-tuned end-to-end with two
classification heads:
  - face_head : Face (1) vs Non-Face (0)
  - mask_head : Mask (1) vs No-Mask (0)

MobileNetV3 Small from torchvision, adapted for 112x112 face crops:
  - Removes the 1000-class ImageNet classifier
  - Adds a 576 → embedding_dim linear projection + BN
  - Works with 112x112 input (features → 576 × 4 × 4 → AdaptiveAvgPool → 576)

Compared to ShuffleNetV2 x0.5:
  - ~2× more parameters (1.0M vs 473K backbone)
  - Similar latency (~0.89ms vs ~0.83ms)
  - Potentially better accuracy due to SE blocks and H-Swish activations
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Backbone — Trainable MobileNetV3 Small (from torchvision)
# --------------------------------------------------------------------------- #

class MobileNetV3SmallBackbone(nn.Module):
    """
    MobileNetV3 Small adapted for 112x112 face crops.

    Input : (B, 3, 112, 112) float32 [-1, 1]
    Output: (B, embedding_dim) float32

    Architecture with 112x112 input:
        features → (B, 576, 4, 4)
        avgpool  → (B, 576, 1, 1)
        flatten  → (B, 576)
        project  → (B, embedding_dim)
    """

    FEATURE_DIM = 576   # MobileNetV3 Small output channels before classifier

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        self.embedding_dim = embedding_dim

        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            base = models.mobilenet_v3_small(weights=weights)
            print(f"[backbone] MobileNetV3 Small — ImageNet pretrained")
        else:
            base = models.mobilenet_v3_small(weights=None)
            print(f"[backbone] MobileNetV3 Small — random init")

        # Keep feature extraction layers and adaptive pool
        self.features = base.features    # conv layers + inverted residuals + SE blocks
        self.avgpool  = base.avgpool     # AdaptiveAvgPool2d((1, 1))

        # Replace ImageNet classifier with embedding projection
        # 576 (after avgpool + flatten) → embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[backbone] Params   : {total:,}")
        print(f"[backbone] Embed dim: {embedding_dim}")
        print(f"[backbone] Input    : (B, 3, 112, 112)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # → (B, 576, 4, 4)
        x = self.avgpool(x)        # → (B, 576, 1, 1)
        x = x.flatten(1)           # → (B, 576)
        x = self.projection(x)     # → (B, embedding_dim)
        return x


# --------------------------------------------------------------------------- #
#  Dual-head classifier
# --------------------------------------------------------------------------- #

class DualHeadClassifier(nn.Module):
    """
    Two independent classification heads on top of the backbone embedding.

    face_head : 0 = non-face,  1 = face
    mask_head : 0 = no-mask,   1 = mask
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        hidden = max(embedding_dim // 4, 32)

        self.face_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.PReLU(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

        self.mask_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.PReLU(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def forward(self, embeddings: torch.Tensor):
        return self.face_head(embeddings), self.mask_head(embeddings)


# --------------------------------------------------------------------------- #
#  End-to-end model (backbone + heads)
# --------------------------------------------------------------------------- #

class EndToEndClassifier(nn.Module):
    """
    Full pipeline: MobileNetV3 Small backbone → dual heads.

    Input : (B, 3, 112, 112) float32
    Output: face_logits (B, 2), mask_logits (B, 2)
    """

    def __init__(self, backbone: MobileNetV3SmallBackbone,
                 heads: DualHeadClassifier):
        super().__init__()
        self.backbone = backbone
        self.heads    = heads

    def forward(self, x: torch.Tensor):
        emb = self.backbone(x)
        return self.heads(emb)

    def export_backbone_onnx(self, output_path: str):
        """Export just the backbone to ONNX for production use."""
        import os
        self.backbone.eval()
        device = next(self.backbone.parameters()).device
        dummy = torch.randn(1, 3, 112, 112, device=device)

        torch.onnx.export(
            self.backbone, dummy, output_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={
                "input":     {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=18,
            dynamo=False,
        )
        mb = os.path.getsize(output_path) / 1e6
        print(f"[export] Backbone → {output_path}  ({mb:.2f} MB)")
        print(f"[export] Input  : (B, 3, 112, 112)")
        print(f"[export] Output : (B, {self.backbone.embedding_dim})")

    def export_heads_onnx(self, output_path: str):
        """Export just the heads to ONNX."""
        import os
        self.heads.eval()
        device = next(self.heads.parameters()).device
        dummy = torch.randn(1, self.backbone.embedding_dim, device=device)

        torch.onnx.export(
            self.heads, dummy, output_path,
            input_names=["embedding"],
            output_names=["face_logits", "mask_logits"],
            dynamic_axes={"embedding": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
        mb = os.path.getsize(output_path) / 1e6
        print(f"[export] Heads   → {output_path}  ({mb:.2f} MB)")