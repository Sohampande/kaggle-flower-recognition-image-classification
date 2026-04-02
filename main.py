from src.cnn_classifier.components.model_preparation import ModelPreparation
from src.cnn_classifier.config.configuration import ConfigurationManager

configman = ConfigurationManager() 
config = configman.get_model_preparation_config()
prep = ModelPreparation(config)
model, criterion, optimizer, scheduler, device = prep.initiate_model_preparation()

print(model)
print(criterion)
print(optimizer)
print(scheduler)
print(device)