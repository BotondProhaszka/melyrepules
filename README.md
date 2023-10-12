# melyrepules

## Participants

* Antal Péter  ZE4SG8
* Liska Tamás  IWGB4I
* Prohászka Botond Bendegúz  DG1647

## Description

Participate in https://www.kaggle.com/competitions/birdclef-2023 and achive as good result as possible.

### The competition  

Birds are excellent indicators of biodiversity change since they are highly mobile and have diverse habitat requirements. Changes in species assemblage and the number of birds can thus indicate the success or failure of a restoration project. However, frequently conducting traditional observer-based bird biodiversity surveys over large areas is expensive and logistically challenging. In comparison, passive acoustic monitoring (PAM) combined with new analytical tools based on machine learning allows conservationists to sample much greater spatial scales with higher temporal resolution and explore the relationship between restoration interventions and biodiversity in depth.

For this competition, you'll use your machine-learning skills to identify Eastern African bird species by sound. Specifically, you'll develop computational solutions to process continuous audio data and recognize the species by their calls. The best entries will be able to train reliable classifiers with limited training data. If successful, you'll help advance ongoing efforts to protect avian biodiversity in Africa, including those led by the Kenyan conservation organization NATURAL STATE.

## Function of the files in the repository

### data_prep/data_preparation.py

We downloaded the datapack from the website of the competiton and integrated into the data preparation docker container.

In data_preparation.py file we implemented a basic data generator class in order to make the dataset available to other containers (e.g. models). The data generator class can resample a wave to a desired rate. We create three instances of the generator class: for the training, validation and test datasets.

Related works:


### data_prep/Dockerfile

Creates and runs a docker container.

### data_prep/requirements

Describes each library version required in the project.

### .gitignore

Contains the files and librarys we do not intend to upload to github.

## Related works
* The [dataset](https://www.kaggle.com/competitions/birdclef-2023/data)
* [Opening and resampling the waves](https://www.kaggle.com/code/philculliton/inferring-birds-with-kaggle-models)
* For the ```resampling``` function we used [Google Bard](bard.google.com/) 


## How to run

* Install python3 and pip
* Todo
