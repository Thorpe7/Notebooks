"""
EfficientNet-B0 model for medical image classification.

EfficientNet-B0 is more parameter-efficient than ResNet-50 (~5M vs ~25M parameters),
making it better suited for small datasets and reducing overfitting risk.
"""
import torch
import torch.nn as nn
from torchvision import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with proper regularization for medical image classification.

    Features:
    - Pre-trained ImageNet weights for transfer learning
    - Dropout before final classification layer
    - Optional feature extraction mode (frozen backbone)
    """

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        """
        Initialize EfficientNet-B0 classifier.

        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability before final layer
            freeze_backbone: If True, freeze all backbone weights for feature extraction
        """
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Get the number of features from the backbone's classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with custom head including dropout
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        # Optionally freeze backbone for initial training
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all backbone parameters (features), keep classifier trainable."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_efficientnet_b0(
    num_classes: int,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> EfficientNetB0Classifier:
    """
    Factory function to create EfficientNet-B0 classifier.

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability (default 0.3)
        freeze_backbone: Whether to freeze backbone initially

    Returns:
        EfficientNetB0Classifier model moved to appropriate device
    """
    model = EfficientNetB0Classifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    return model.to(device)


# For backwards compatibility and simple usage
class EfficientNet:
    """Wrapper class matching ResNet interface for drop-in replacement."""

    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.3):
        self.model = create_efficientnet_b0(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=False,
        )

    def get_model(self) -> EfficientNetB0Classifier:
        return self.model
