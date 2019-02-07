# csc2541_project

## Description
Build an ML classifier to match procedure at each site to a ground truth ontology defined by imaging repository (radlex/LOINC)

Using Python 3.6

## Directory Structure
**input_data/** contains the input data as csv (not uploaded due to privacy concerns/restrictions)

**output/** should be where all output is stored

**data_reader.py** contains DataReader which is a class that reads csv's from input_data and presents them as a pandas dataframe

**data_manipulator.py** is a library that has functions for manipulating data, such as splitting the dataframe into train/test

**Models/simple_logistic_regression.py** is trivial example of a model wrapper, made to work well with our data.

## Development
To get started, download the .csv files from the Google Drive and paste them into input_data. From there, you can do the following:

  To create a new model, create a new subclass of Model in Models. Implement the abstract methods of Models and add any additional needed methods.

  To modify how data is read, add a method to the DataReader class inside data_reader.py

  To transform the data, implement a method inside data_manipulator.py


