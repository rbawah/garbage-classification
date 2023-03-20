import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from transformers import BertTokenizer, BertForSequenceClassification



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_vision_model(model_name, num_classes, feature_extract, use_pretrained=True, multimodal=True):
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

    else:
        print("Invalid Model Name!")
        exit()
        
    return model_ft, input_size, out_features


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