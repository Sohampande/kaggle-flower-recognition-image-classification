# This file is primarly made when we want to make our project into a package
#       through this file, anybody can setup the project and run it.

import setuptools

with open('README.md', "r", encoding='utf-8') as f:
    long_discription = f.read()

__version__ = "0.0.0"

REPO_NAME = "kaggle-flower-recognition-image-classification"
AUTHOR_USER_NAME = "Sohampande"
SRC_REPO = "kaggle-flower-recognition-image-classification"
AUTHOR_EMAIL = "sohampande58@gmail.com"

# One of the key tools of this file is the setup class. 
#       through this class, it is accessible for anyone to set up 
#       the project on their local machines as long as the meet 
#       the system requirements.

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_context="text/markdown",
    url=f"https://github.com/Sohampande/kaggle-flower-recognition-image-classification"
    project_url={
        "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")
)