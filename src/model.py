import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class SpectrogramCNN(nn.Module):
    """Simple CNN architecture for spectrogram classification."""
    
    def __init__(self, num_classes=2):
        super(SpectrogramCNN, self).__init__()
        
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size after convolution and pooling layers
        # Input: 224x224 -> 4 pooling layers with stride 2 -> 224/(2^4)=14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, 1, 224, 224]
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResNetTransfer(nn.Module):
    """Transfer learning model using pre-trained ResNet."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNetTransfer, self).__init__()
        
        # Load pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first convolution layer to accept grayscale input (1 channel)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class EfficientNetTransfer(nn.Module):
    """Transfer learning model using pre-trained EfficientNet."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetTransfer, self).__init__()
        
        # Use timm library for EfficientNet
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        
        # Modify the first convolution layer to accept grayscale input
        original_conv = self.model.conv_stem
        self.model.conv_stem = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Replace classifier
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def get_model(model_name='cnn', num_classes=2, pretrained=True):
    """Factory function to get the specified model."""
    
    if model_name == 'cnn':
        return SpectrogramCNN(num_classes=num_classes)
    elif model_name == 'resnet':
        return ResNetTransfer(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet':
        return EfficientNetTransfer(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported.")

if __name__ == "__main__":
    # Test the model
    model_names = ['cnn', 'resnet', 'efficientnet']
    
    for model_name in model_names:
        try:
            print(f"Testing {model_name} model...")
            model = get_model(model_name)
            print(model)
            
            # Create a dummy input
            x = torch.randn(4, 1, 224, 224)  # [batch_size, channels, height, width]
            
            # Forward pass
            output = model(x)
            print(f"Output shape: {output.shape}")
            print("-" * 50)
        except Exception as e:
            print(f"Error with {model_name}: {e}")