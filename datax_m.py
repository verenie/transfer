# Copyright <2019> Edward Verenich <verenie@clarkson.edu>
# MIT license <https://opensource.org/licenses/MIT>
# The script is used prepare TRAIN/VALIDATE/TEST splits
# for use with generic image dataset classes such as torchvision.datasets.ImageFolder
# Optional ground_truth.csv file can be generated for any split in the format: <filename,class>
# Usage:  after setting up directory structure variables, specify number of FOLDS,
# then run the datax_m.py script.
# Developed with Python 3.7.3


import numpy as np
import sklearn.model_selection as model_selection
import os
import shutil
import csv

# setup directory structures
# because of the way the imagefolder dataset module works, make sure
# that our target class appears first in directory listing
# in our use case we are interested in the target class (aplha) and other
# (beta), although as many classes is supported as needed, we use two
# to make our confusion matrix easy to read durring model development

OUTPUT_DIR = "./data/tel"
# class map keys corresspond to image classes, i.e. dog/cat/monkey
class_map = {'alpha': "./data/raw_tel/tels", 'beta': "./data/raw_tel/not_tels" }
dir_type = ['test','val','train']

# settings below will perform a TRAIN/VAL/TEST split of 60/20/20 percent
# perform two splits using the train_test_split method from sklearn
# first split at 80/20 and second on 75/25 on the remaining train set
TEST_SIZE = 0.2
VAL_SIZE = 0.25
# used in train split to reproduce the splits, any int value works
SEED = 79
# used to create K-fold splits of train/validate while keeping the test set the same
FOLDS = 3



def build_directory(class_key, d_type, names, fold, ground_truth=False):
  # create directory /OUTPUT_DIR/<val or test or train>/<alpha or beta>
  FOLD_DIR = OUTPUT_DIR + str(fold)
  try:
    os.makedirs(os.path.join(FOLD_DIR,d_type,class_key))
  except FileExistsError:
    # already exists
    pass
  # create a ground_truth.csv file if needed and get a writer ref

  if ground_truth:
    gt_file = os.path.join(FOLD_DIR,d_type,'ground_truth.csv')
    with open(gt_file, 'a') as gt:
      writer = csv.writer(gt)
      for f in names:
         shutil.copy(os.path.join(class_map[class_key], f), os.path.join(FOLD_DIR,d_type,class_key,f))
         writer.writerow([f,class_key])
  else:
    for f in names:
      shutil.copy(os.path.join(class_map[class_key], f), os.path.join(FOLD_DIR,d_type,class_key,f))




if __name__ == "__main__":
  for k in range(FOLDS):
    # read class_map keys that corresspond to classes
    for d in class_map:
      # get the files for a given class
      X = os.listdir(class_map[d])
      print(d, " -class- ",len(X))
      # split the first time into train, test
      train, test = model_selection.train_test_split(X, test_size=TEST_SIZE, random_state=SEED)
      print("Test: ", len(test))
      # build the test dir
      build_directory(d, dir_type[0], test, k, ground_truth=True)
      # split the second time from the remaining train with a different seed value
      train, validate = model_selection.train_test_split(train, test_size=VAL_SIZE, random_state=SEED+(k+1))
      # build the training directory, no ground_truth file
      build_directory(d, dir_type[2],train, k)
      # build the val directory
      build_directory(d,dir_type[1], validate, k, ground_truth=True)
      print("Train: ", len(train))
      print("validate: ", len(validate))




