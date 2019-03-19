import numpy as np
from data_reader import DataReader
import warnings
warnings.filterwarnings("ignore")


# get all the labelled data
reader = DataReader()
data = reader.get_all_data()

print('\n================== Before removing missing values ==================\n')
print('No. of samples: {}'.format(len(data)))
print('No. of classes: {}'.format(data['ON WG IDENTIFIER'].nunique()))
counts = data.groupby(['ON WG IDENTIFIER']).size().to_frame(name='counts') \
                        .sort_values(['counts']).values
print('Max no. of samples for a class: {}'.format(counts[-1][-1]))
print('Min no. of samples for a class: {}'.format(counts[0][0]))
print('Avg no. of samples for a class: {}'.format(round(np.mean(counts), 2)))
print('\n===================================================================\n')

print('================== After removing missing values ====================\n')
# drop rows with missing values
dataNoNan = data.dropna()
print('No. of samples: {}'.format(len(dataNoNan)))
print('No. of classes: {}'
      .format(dataNoNan['ON WG IDENTIFIER'].nunique()))

counts = dataNoNan.groupby(['ON WG IDENTIFIER']).size().to_frame(name='counts') \
                        .sort_values(['counts']).values
print('Max no. of samples for a class: {}'.format(counts[-1][-1]))
print('Min no. of samples for a class: {}'.format(counts[0][0]))
print('Avg no. of samples for a class: {}'.format(round(np.mean(counts), 2)))
print('\n====================================================================')