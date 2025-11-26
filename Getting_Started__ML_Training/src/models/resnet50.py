import torchvision.models as models
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Following methodology from paper, ResNet50 pre-trained on ImageNet1k used
# With last two layers ("top two layers") removed as to only output feature maps.
class ResNet:
    def __init__(self):
        # Load ResNet50 model with IMAGENET1K_V2 weights
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the top layers (global pooling and dense layer)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model.to(device)

    def get_model(self):
        # Return the modified model
        return self.model

    @staticmethod
    def global_average_pooling(features):
        """
        Perform global average pooling on the feature maps.
        Args:
            features (torch.Tensor): Feature maps of shape [batch_size, channels, height, width].
        Returns:
            torch.Tensor: Pooled features of shape [batch_size, channels].
        """
        return features.mean(dim=[2, 3])
