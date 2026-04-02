# PROJECT SUMMARY : 

In this project I am making a simple image classification model with PyTorch to identify different types of flowers nameley: daisy, dandelion, rose, sunflower, tulip. Thus, the goal of this project is to build a flower image classifier.

I have copied a lot of the project structure and formatting from the [GOAT](https://github.com/krishnaik06). Spcifically, I am using [this project](https://www.youtube.com/watch?v=86BKEv0X2xU&t=501s) to follow the templates.

I am using the dataset sourced from this [site](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data). For now, I have downloaded the dataset fromt he source and then using it.

## Some Important Notes : 
- I have not added the dataset here because it takes up a lot of space int the repo. But, by downloading it from the given link above it should be okay.
- I am also not adding the enviornment files here, (ie venv folder). But any changes made to the enviorment, I will be tracking here.
- ** Managing Data ** : Through data ingestion, the pipeline should be able to download the data the local file for training. I am downloading the data from the kaggle source that I have mentioned above.
- config.yaml : is a file that only stores paths and artifact locations such as : artifact root, raw data location, saved model paths, metrics path, plots paths, etc. 
- params.yaml : we put the experiment settings here like the image size, batch size epochs, etc. Have a look at the file for the comprehensive list. 

## How does it work : 
For each step :
      1. read config.yaml In the config.yaml file, I have all the necessary data stored. There we also have the location from which data needs to be downloaded and where it needs to be downloaded. 

      2. **Data Ingestion** : reads the data from artifacts/data_ingestion/flowers using CustomImageDataset. Then it gets the names from folder names and splits the data for train/eval/test. Note, we use augmentation only on training data so that the model prevents overfitting. By the end of this stage, we output the train_loader, val_loader, test_loader, class_names, and dataset_sizes. 

## Different Models : 
      1. In this project, I have developed several different types of models. These models have been tested for the training data, and the best one has been selected. All the models I have made are in src/cnn_classifier/models/ . Here, I have given a general list of all the models that I have created. For further details, have a look into the models. The list of models is : 
         
         A. Multilayer Perceptron 
         B. 

# WorkFlow : 
   1. Update config.yaml
   2. Update secrets.yaml
   3. Update params.yaml
   4. Update the entity
   5. Update teh configuration Manager in src config
   6. Update the components
   7. Update the pipeline
   8. Update main.py
   9. Update the dvc.yaml
   10. app.py

# Order For Creation : 
1. data ingestion 
2. data transformation 
3. model preparation 
4. model training
5. model evaluation 
6. model selection 
7. prediction pipeline 
8. deployment. 

## How to get started : 
1. Clone the Repo :
   ```bash
   https://github.com/Sohampande/kaggle-flower-recognition-image-classification
   ```

2. Install the requirements :
   ```bash
   pip install -r requirements.txt
   ```
   
   ```bash
   # To run the file, use this command :
   python app.py
   ```
