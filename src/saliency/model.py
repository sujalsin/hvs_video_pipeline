import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SaliencyDetector(nn.Module):
    """Lightweight saliency detection model."""
    
    def __init__(self):
        super(SaliencyDetector, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Decoder
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x1_pool = self.pool(x1)
        
        x2 = self.relu(self.conv2(x1_pool))
        x2_pool = self.pool(x2)
        
        x3 = self.relu(self.conv3(x2_pool))
        
        # Decoder
        x4 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = self.relu(self.conv4(x4))
        
        x5 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = self.relu(self.conv5(x5))
        
        x6 = self.conv6(x5)
        
        return torch.sigmoid(x6)


class VideoSaliencyPredictor:
    """Wrapper class for video saliency prediction."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SaliencyDetector().to(device)
        self.model.eval()
    
    def predict_frame(self, frame):
        """Predict saliency map for a single frame."""
        with torch.no_grad():
            # Preprocess frame
            if not isinstance(frame, torch.Tensor):
                frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                frame = frame.unsqueeze(0)  # Add batch dimension
                frame = frame.to(self.device)
            
            # Normalize
            frame = frame / 255.0
            
            # Predict saliency
            saliency_map = self.model(frame)
            
            # Convert to numpy
            saliency_map = saliency_map.cpu().numpy().squeeze()
            
            return saliency_map
    
    def predict_video(self, frames):
        """Predict saliency maps for a sequence of frames."""
        return [self.predict_frame(frame) for frame in frames]
