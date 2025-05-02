import torch
import torch.nn as nn
from torchvision import models

class ArtStyleClassifier(nn.Module):
    def __init__(self, num_styles=10):
        super(ArtStyleClassifier, self).__init__()
        
        # Load pretrained model
        self.base_model = models.resnet50(pretrained=True)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Replace final layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_styles)
        )
        
    def forward(self, x):
        return self.base_model(x)

# Common art styles
ART_STYLES = [
    "Impressionism",
    "Expressionism",
    "Cubism",
    "Surrealism",
    "Abstract Art",
    "Realism",
    "Pop Art",
    "Contemporary Art",
    "Classical Art",
    "Digital Art"
]

def get_style_prediction(model, image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_style = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][predicted_style].item()
        
    return ART_STYLES[predicted_style.item()], confidence 