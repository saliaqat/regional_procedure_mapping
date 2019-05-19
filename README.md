# Sharing is Caring: Medical image consolidation using machine learning

## Description
This work started off as a course project, so the code acts as an archive of methods we explored.

main.py contains the code to generate the evaluation in the paper. A majority of the code was run on a server with a 14 core/28 thread Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz. The server has 128Gb of ram and Nvidia Quadro P4000 (8Gb). On this server, training took anywhere from under a minute up to 12 hours depending on the representation/model used.
 

## Directory Structure
**input_data/** contains the input data as csv (not uploaded due to privacy concerns/restrictions)

**output/** should be where all output is stored

**data_reader.py** contains DataReader which is a class that reads csv's from input_data and presents them as a pandas dataframe

**data_manipulator.py** is a library that has functions for manipulating data, such as splitting the dataframe into train/test

**Models/** Contains all the models we tried

**main.py** shows the code used to generate results shown in the paper.

## Development
To get started, download the .csv files from the Google Drive and paste them into input_data. From there, you can do the following:

    * To create a new model, create a new subclass of Model in Models. Implement the abstract methods of Models and add any additional needed methods.

    * To modify how data is read, add a method to the DataReader class inside data_reader.py

    * To transform the data, implement a method inside data_manipulator.py
        
    * Run main.py to see results


