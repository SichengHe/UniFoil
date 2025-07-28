# Data Generation Launcher
# Ben Melanson
# July 28th, 2025

# Description
# There are many logistical problems when dealing with a data set of this size.
# It would be convienient to simply save the dataset in a native .keras format,
# However it is impossible to load all the data at once.
# This script will generate a series of child scripts to help gather all the data.
# Modify DataGenerationUtility.py to alter the parameters of the process

import os
import subprocess
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.layers import Dense, Input, Conv2DTranspose, UpSampling2D, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model

returnCode = 1 # Set this to 0 to skip generation
index = 0

while returnCode > 0:
    print("Generating Shard #" + str(index))
    returnCode = subprocess.run(['python','./DataGenerationUtility.py']).returncode
    print(str(returnCode) + ' Remaining Shards')
    index = index + 1

print('All shards compiled!')

print('Testing compilation...')

rootPath = os.path.dirname(__file__)
rootPath = rootPath + '/'

datasetFolder = rootPath + 'datasetFolder/'

allFiles = os.listdir(datasetFolder)

shardCount = len(allFiles)

print('Starting Dataset Compliation with shard + \'' + allFiles[0] + '\'')
dataset = tf.data.Dataset.load(datasetFolder + allFiles[0])

for file in allFiles[1:-1]:
    print('Stitching shard \'' + file + '\' to dataset...')
    dataset = dataset.concatenate(tf.data.Dataset.load(datasetFolder + file))

print('Dataset Stitching done, printing an output...')

for entry in tf.data.Dataset.as_numpy_iterator(dataset.take(1)):
    print(entry)

print('Running an example Neural Network')
# This neural network is just an example to show the input/output pipeline.
# Its actual effectivness has never been tested, though feel free to change that.
network = Sequential([
    Input([292, 2]),
    Reshape([292 * 2]),
    Dense(24528, activation = 'swish'),
    Reshape([292, 84])
])

network.compile(
    optimizer = 'adamax',
    loss = 'mse',
    metrics = ['accuracy']
)

network.summary()

network.fit(
    dataset,
    epochs = 5
)