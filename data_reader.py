import os
import pandas as pd
from sklearn.model_selection import train_test_split
# Data Reader class. Pass in data path and the class with read all the csv files inside the directory.
# Provides methods to get certain data
class DataReader:
    def __init__(self, data_path='input_data/'):
        self.data_path = data_path
        self._read_data_directory()


    def _read_data_directory(self):
        self.files = [(self.data_path + f) for f in os.listdir(self.data_path) if os.path.isfile(self.data_path + f) and
                      (self.data_path + f).endswith('.csv')]

        self.df_list = ([], [])
        for file in self.files:
            df = pd.read_csv(file, index_col=None, header=0)
            df = df.drop('Unnamed: 9', axis=1)
            df['src_file'] = file
            self.df_list[0].append(df)
            self.df_list[1].append(file)

        self.df = pd.concat(self.df_list[0], axis=0, ignore_index=True)

        df = pd.read_csv('input_data/region_labels/regional_labels.csv', index_col=None, header=0, sep=',')
        self.regional_df = df

        self.east_dir = pd.read_csv('input_data/east_dir/DICS Maps_GTAW_HDIRS_20190306.csv', sep=',', header=0)


    # PUBLIC INTERFACE BELOW #

    # Returns all csv's from datapath merged into a single dataframe
    def get_all_data(self):
        return self.df

    # Returns a list containing the the full file names of all the files in the data path
    def get_files(self):
        return self.files

    # Takes a file path, and returns a dataframe containing all the data.
    def get_file_data(self, file):
        for i in range(len(self.df_list[0])):
            if self.df_list[1][i] == file:
                return self.df_list[0][i]

        raise FileNotFoundError

    def get_region_labels(self):
        return self.regional_df

    def get_east_dir(self):
        return self.east_dir