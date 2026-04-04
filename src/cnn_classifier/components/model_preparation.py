import torch
import torch.nn as nn
import torch.optim as optim

from src.cnn_classifier.models.model_factory import get_model

class ModelPreparation :
    def __init__(self, config):
        self.config = config
    
    def initiate_model_preparation(self):
        model = get_model(
            model_name = self.config.model_name,
            num_classes = self.config.num_classes
        )

        device = torch.device(self.config.device)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        if self.config.optimizer_name.lower() == "adam":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay    
            )
        else : 
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")
        
        scheduler = None
        if self.config.scheduler_name:
            if self.config.scheduler_name.lower() == "steplr":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size = self.config.step_size,
                    gamma = self.config.gamma
                )
            else : 
                raise ValueError(f"Unsupported scheduler: {self.config.scheduler_name}")
        
        return model, criterion, optimizer, scheduler, device
    
    def prepare_model(self, model_name: str):
        model = get_model(
            model_name=model_name,
            num_classes=self.config.num_classes
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = None

        return model, criterion, optimizer, scheduler
