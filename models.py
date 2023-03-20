import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_vision_model(model_name, num_classes, feature_extract, use_pretrained=True):
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
        print("Initializing InceptionV3 with weights=Inception_V3_Weights.DEFAULT...")
        print(f'Input size = {input_size}')
        print(f'out_features = {num_classes}')

    elif model_name == "efficientnet_b7":
        ######################
        #EfficientNet B7 model
        #Image size (600, 600)
        ######################

        #Set model with weights and if we're freezing the model
        model_ft = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)

        #Replace the existing linear classifer with 1000 output classes with
        #number of classifiers in garbage-classification-project.
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 600
        print("Initializing EfficientNetB7 with weights=EfficientNet_B7_Weights.DEFAULT...")
        print(f'Input size = {input_size}')
        print(f'out_features = {num_classes}')


    else:
        print("Invalid Model Name!")
        exit()
        
    return model_ft, input_size