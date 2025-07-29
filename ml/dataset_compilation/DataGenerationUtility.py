# Data Generation Utility
# Ben Melanson
# July 26th, 2025

# Description
# This example script shows how you can use Pyvista to generate a dataset from the Unifoil set. 
# In this example, a dataset is generated with airfoil surface coordinates as the features and
# the pressure coefficents as the labels. All of the data is extracted directly from the CGNS
# Data, however other sources can also be used.

# Imports

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Used to supress Tensorflow errors.
import tensorflow as tf
import pyvista as pv
import gc
import sys

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disables GPU acceleration

allFiles = os.listdir('./output/')
targetFiles = []

coordinateTensor = tf.TensorSpec(shape = (None, 292, 2), dtype = tf.float64) # This is the Features Format
pressureTensor = tf.TensorSpec(shape = (None, 292, 84), dtype = tf.float64) # This is the Labels Format

datasetTensor = [coordinateTensor, pressureTensor] # The final dataset format, which is made of the two tensors.


def extractBlocks(blockReader, initialBlocks = []): 
# This function extracts the data blocks from the .CGNS files. It is recursive.
    allBlocks = initialBlocks
    for i in range(blockReader.n_blocks):
        block = blockReader[i]
        if isinstance(block, pv.MultiBlock):
            extractBlocks(block)
        elif block is not None:
            allBlocks.append(block)
    return allBlocks

def CGNSExtractor():
# This function is a Tensorflow Dataset Generator. For all entries in targetFiles, it will
# extract the given variables and return them in a tensor format.
    for file in targetFiles:

        print('Generating Data for file:' + file) # Debug printout, shows you that it is actually working :)

        reader = pv.CGNSReader('output/' + file) # Creates a PyVista CGNS Reader pointed at the given file.
    
        reader.load_boundary_patch = False

        cgnsData = reader.read()

        blocks = extractBlocks(cgnsData)

        block = blocks[3] # All data is in block 3. I am not sure why.

        cell_centers = block.cell_centers()

        coords = cell_centers.points

        xCoords = coords[:,0] 
        yCoords = coords[:,1]

        coords = np.vstack([xCoords, yCoords]) # There is probably an easier way to create the x,y coordinate tensor
        coords = np.reshape(coords, [2, 292, 84]) # Ensures that the coords are a stack of x,y in a 292 x 84 grid
        coords = coords[:, :, 0] # Strips all but the first layer of points around the surface
        coords = np.reshape(coords, [1, 292, 2]) # Adds a dummy dimension before the coords to represent Batch Size

        pressureCoefficent = block.cell_data['CoefPressure'] # Extracts the pressure data from the block.

        pressureCoefficent = np.reshape(pressureCoefficent, [1, 292, 84]) # Adds a dummy dimension before the pressures to represent Batch Size

        yield (coords, pressureCoefficent)

# Saving the dataset
# In a perfect world I would be able to use the following command to save the entire dataset.
# tf.data.Dataset.save(dataset, './datasetFolder/data') 
# However this doesn't work for a dataset of this size. It MIGHT work if you are only using a small portion of the dataset.
# Otherwise you will have to use this manual sharding setup that I wrote below

targetDir = './datasetFolder/' # Directory to save all of the shards

targetFilePath = targetDir + 'dataset' # Shard Extension.

filesPerShard = 2000 # Amount of entries to save into each shard. 2000 seems to be a round number with around 400 mb per shard. Decrease for smaller shard sizes.

dataLength = len(allFiles) # Length of all airfoil entries.

shardCount = int(dataLength / filesPerShard) + (1 if dataLength % filesPerShard != 0 else 0) # The amount of shards that will need to be generated.

recordPath = '{}_{}' # This is the naming format for the shards.

currentShards = os.listdir(targetDir) # Lists all the shards that have already been generated.

shard = len(currentShards) # Selects the current shard to run on.

if shard >= shardCount:
    sys.exit(0)

index = int(shard * filesPerShard) # First file that needs to be generated. Always a multiple of filesPerShard

shardPath = recordPath.format(targetFilePath, '%.3d_of_%.3d' % (shard, shardCount - 1))

end = index + filesPerShard if dataLength > (index + filesPerShard) else -1 # Picks the endpoint for the files. Required to prevent overflow errors.

targetFiles = allFiles[index : end] # Picks all files in the given range for processing.

# This runs the generator on the targetFiles property.
dataset = tf.data.Dataset.from_generator(
CGNSExtractor,
output_signature = (coordinateTensor, pressureTensor)
)

# Saves the generated data to the shard path,
dataset.save(shardPath)

tf.keras.backend.clear_session # This was here in an attempt to save memory for large datasets, i am unsure if it is still necessary.

gc.collect() # Runs garbage collection just in case.

remainingShards = shardCount - (shard + 1)

sys.exit(remainingShards) # Returns the amount of remaining shards to the Launcher Script.
