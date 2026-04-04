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
- Im using a MacbookPro, where I can run the code on gpu(aka mps). If this is a feature not available, then simply go to config/config.yaml and set model_preparation.device = cpu.

## How does it work : 
For each step :
   1. **Data Ingestion** : read config.yaml In the config.yaml file, I have all the necessary data stored. There we also have the location from which data needs to be downloaded and where it needs to be downloaded. 

   2. **Data Transformation** : reads the data from artifacts/data_ingestion/flowers using CustomImageDataset. Then it gets the names from folder names and splits the data for train/eval/test. Note, we use augmentation only on training data so that the model prevents overfitting. By the end of this stage, we output the train_loader, val_loader, test_loader, class_names, and dataset_sizes. 

   3. **Model Preparation** : In this step, using the components from the previous step we, we prepare our model for training. This means : (1) we define critical variables such as learning_rate, no of epochs, weight_decay, etc. (2) In this step we also define a very important file called the model_factory.py which improves model retirval and makes for cleaner code. 
   In this step, we make many important design choices, these are : 
      - weight_decay : 0.0001 because L2 regularization is another layer in the code that should not be hard-coded.

   4. **Model Training** : In this section, (1) we load the dataloaders, (2) get the model/criterion/etc from the model preparation, (3) run epoch loop, (4) compute train/val metrics, (5) save the best model. After this stage the following files are stored : best_model.pth, last_model.pth, metrics.json, history.json . Moreover, we store the parameters/settings for each of the models in artifacts/model_training.
   
   5. **Model Evaluation** : In model evaluation, we test out the model all the models that we have trained. We are using the metrics : (1) accuracy, (2) precision_macro, (3) recall_macro, (4) f1_macro, (5) confusion_macro, and (6) loss. Each evaluation metric tells us something important. For example, accuracy tells us the overall performance of the model, where as precision tells us how clean our predictions are. Also, the confusion matrix tells us where the model fails. 

   At the end of this stage, the folder artifacts/model_evaluation is created, in which all the above metrics for a model are stored.

   6. **Model Selection** : Using the data from model evaluation, the following graph can be generated : 

   ![Alt Text][/Users/Soham/Documents/Projects/kaggle-flower-recognition-image-classification/artifacts/model_selection/model_comparison.png]

   From the above image, we can see that the best performing models are dropout and bacthnorm. The models baseline and width overfit the data, and hence perfom poorly in testing. By a small margin, dropout beats batchnorm.

   Hence, dropout is the best model to use.

   7. deployment.

## Different Models : 
   1. In this project, I have developed several different types of models. These models have been tested for the training data, and the best one has been selected. All the models I have made are in src/cnn_classifier/models/ . Here, I have given a general list of all the models that I have created. For further details, have a look into the models. The list of models is : 
      - Multilayer Perceptron 

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
