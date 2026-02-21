# PROJECT SUMMARY : 

In this project I am making a simple image classification model with PyTorch to identify different types of flowers nameley: daisy, dandelion, rose, sunflower, tulip. Thus, the goal of this project is to build a flower image classifier.

I have copied a lot of the project structure and formatting from the [GOAT](https://github.com/krishnaik06). Spcifically, I am using [this project](https://www.youtube.com/watch?v=86BKEv0X2xU&t=501s) to follow the templates.

I am using the dataset sourced from this [site](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data). For now, I have downloaded the dataset fromt he source and then using it.

## Some Important Notes : 
- I have not added the dataset here because it takes up a lot of space int the repo. But, by downloading it from the given link above it should be okay.
- I am also not adding the enviornment files here, (ie venv folder). But any changes made to the enviorment, I will be tracking here.
- ** Managing Data ** : Through data ingestion, the pipeline should be able to download the data the local file for training. I am downloading the data from the kaggle source that I have mentioned above.

## How does it work : 
For each step :
      1. read config.yaml In the config.yaml file, I have all the necessary data stored. There we also have the location from which data needs to be downloaded and where it needs to be downloaded. 

      2. Update params.yaml :

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
