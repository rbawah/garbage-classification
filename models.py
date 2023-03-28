import torch
import torch.nn as nn
from torchvision.models import (inception_v3, 
                                Inception_V3_Weights,
                                efficientnet_b7, 
                                EfficientNet_B7_Weights)
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_vision_model(model_name, num_classes, feature_extract, multimodal=True):
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
        out_features = model_ft.fc.out_features
        if multimodal:
            model_ft.fc = nn.Identity()
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        print("Initializing InceptionV3 with weights=Inception_V3_Weights.DEFAULT...")
        print(f'Input size = {input_size}')
        # print(f'out_features = {out_features}')

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
        out_features = model_ft.classifier[1].out_features
        #Leave the classification to the joint classifier, so multiply by identity.
        #Otherwise, run the fully connected layer as usual.
        if multimodal:
            model_ft.classifier[1] = nn.Identity()
        else:
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 600
        print("Initializing EfficientNetB7 with weights=EfficientNet_B7_Weights.DEFAULT...")
        print(f'Input size = {input_size}')
        # print(f'out_features = {out_features}')
    
    elif model_name == "mobilenet_v2":
        """
        MobileNetV2 model
        Image size (224, 224)
        """

        #Set model with weights and if we're freezing the model
        model_ft = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)

        #Replace the existing linear classifer with 1000 output classes with
        #number of classifiers in garbage-classification-project.
        num_ftrs = model_ft.classifier[1].in_features
        out_features = model_ft.classifier[1].out_features
        #Leave the classification to the joint classifier, so multiply by identity.
        #Otherwise, run the fully connected layer as usual.
        if multimodal:
            model_ft.classifier[1] = nn.Identity()
        else:
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        print("Initializing MobileNetV2 with weights=MobileNet_V2_Weights.DEFAULT ...")
        print(f'Input size = {input_size}')
        # print(f'out_features = {out_features}')


    else:
        print("Invalid Model Name!")
        exit()
        
    return model_ft, input_size


def initialize_language_model(model_name, num_classes, multimodal=False):
    model = None
    tokenizer = None
    """
    initialize and return the language model.
    """
    if model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if multimodal:
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes, 
                                                            output_hidden_states=False, output_attentions=False)
        else:
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
        print("\nInitializing Bert-Base-Uncased...")
    else:
        print("Invalid language model name.")
        exit()

    return model, tokenizer