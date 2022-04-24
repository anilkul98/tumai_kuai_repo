import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_model(model_name, num_classes):
    model = models.efficientnet_b3(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1536, out_features=num_classes)
    return model