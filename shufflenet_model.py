"""
model.py

Trainable ShuffleNetV2 x0.5 backbone fine-tuned end-to-end with two
classification heads:
  - face_head : Face (1) vs Non-Face (0)
  - mask_head : Mask (1) vs No-Mask (0)
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Backbone — Trainable ShuffleNetV2 x0.5 (from torchvision)
# --------------------------------------------------------------------------- #

class ShuffleNetV2Backbone(nn.Module):
    """
    ShuffleNetV2 x0.5 adapted for 112x112 face crops.

    Modifications from vanilla torchvision model:
      - Removes the 1000-class ImageNet FC
      - Adds a 1024 → embedding_dim linear projection + BN
      - Works with 112x112 input (spatial: 56→28→14→7→4→4→GAP→1)

    Input : (B, 3, 112, 112) float32 [-1, 1]
    Output: (B, embedding_dim) float32
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        self.embedding_dim = embedding_dim

        if pretrained:
            weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
            base = models.shufflenet_v2_x0_5(weights=weights)
            print(f"[backbone] ShuffleNetV2 x0.5 — ImageNet pretrained")
        else:
            base = models.shufflenet_v2_x0_5(weights=None)
            print(f"[backbone] ShuffleNetV2 x0.5 — random init")

        # Keep the feature extraction layers
        self.conv1   = base.conv1
        self.maxpool = base.maxpool
        self.stage2  = base.stage2
        self.stage3  = base.stage3
        self.stage4  = base.stage4
        self.conv5   = base.conv5    # outputs 1024 channels

        # Replace ImageNet FC with embedding projection
        # 1024 (after GAP) → embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(1024, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        # Param count
        total = sum(p.numel() for p in self.parameters())
        print(f"[backbone] Params   : {total:,}")
        print(f"[backbone] Embed dim: {embedding_dim}")
        print(f"[backbone] Input    : (B, 3, 112, 112)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)        # → (B, 24, 56, 56)
        x = self.maxpool(x)      # → (B, 24, 28, 28)
        x = self.stage2(x)       # → (B, 48, 14, 14)
        x = self.stage3(x)       # → (B, 96, 7, 7)
        x = self.stage4(x)       # → (B, 192, 4, 4)
        x = self.conv5(x)        # → (B, 1024, 4, 4)
        x = x.mean([2, 3])       # GAP → (B, 1024)
        x = self.projection(x)   # → (B, embedding_dim)
        return x


# --------------------------------------------------------------------------- #
#  Dual-head classifier (unchanged — works with any embedding dim)
# --------------------------------------------------------------------------- #

class DualHeadClassifier(nn.Module):
    """
    Two independent classification heads on top of the backbone embedding.

    face_head : 0 = non-face,  1 = face
    mask_head : 0 = no-mask,   1 = mask
    """

    def __init__(self, embedding_dim: int = 512):
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
#  End-to-end model (backbone + heads, for fine-tuning)
# --------------------------------------------------------------------------- #

class EndToEndClassifier(nn.Module):
    """
    Full pipeline: ShuffleNetV2 backbone → dual heads.
    Used during fine-tuning when the backbone is trainable PyTorch.

    Input : (B, 3, 112, 112) float32
    Output: face_logits (B, 2), mask_logits (B, 2)
    """

    def __init__(self, backbone: ShuffleNetV2Backbone,
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