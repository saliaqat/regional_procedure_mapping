# csc2541_project

Build an ML classifier to match procedure at each site to a ground truth ontology defined by imaging repository (radlex/LOINC)

Using Python 3.6

input_data contains the input data as csv (not uploaded due to privacy concerns/restrictions)
data_reader contains DataReader which is a class that reads csv's from input_data and presents them as a pandas
dataframe
data_manipulator is a library that has functions for manipulating data, such as splitting the dataframe into train/test
simple_logistic_regression is trivial example of a model wrapper, made to work well with our data.


