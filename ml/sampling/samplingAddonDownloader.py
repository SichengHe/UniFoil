# Sampling Downloader
# Ben Melanson
# July 28th, 2025

# Description
# This downloads pre-compiled models and addition addons that couldn't
# fit in the Github. For manual instalation visit the following link:
# https://drive.google.com/file/d/1DB7hz72mepTajeWiDImrr_8hFNQes3D9/view?usp=sharing

# The Google Drive Zip File contains some example airfoil data taken from the Unifoil set along with a pre-trained model.
# Manual Download, Automatic Download, or providing the data yourself is required for the Line Plotter and Histogram Plotting scripts.
# Requires Gdown

import os
import gdown
from zipfile import ZipFile

requestTarget = '1DB7hz72mepTajeWiDImrr_8hFNQes3D9'

fileName = 'Sampling_Addons'

rootPath = os.path.dirname(__file__)
rootPath = rootPath + '/'

targetFile = rootPath + fileName + '.zip'

gdown.download(id = requestTarget, output = targetFile)

with ZipFile(targetFile, 'r') as zippedFile:
    zippedFile.extractall(rootPath)

newDir = rootPath + fileName + '/'
for file in os.listdir(newDir):
    os.replace(newDir + file, rootPath + file)

os.rmdir(newDir)
os.remove(targetFile)
