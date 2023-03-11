import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights


def initialize_vision_model(model_name, num_classes, feature_extract, weights, use_pretrained=True):
    model_ft = None
    input_size = 0
    
    if model_name == "inception":
        """
        InceptionV3
        Image size should be (299, 299)
        Has auxiliary output
        """
        model_ft = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        print("Initializing InceptionV3 . . .")
        
    else:
        print("Invalid Model Name!")
        exit()
        
    return model_ft, input_size