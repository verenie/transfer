# Copyright <2019> Edward Verenich <verenie@clarkson.edu>
# MIT license <https://opensource.org/licenses/MIT>
# The script is used prepare TRAIN/VALIDATE/TEST splits
# for use with generic image dataset classes such as torchvision.datasets.ImageFolder
# Optional ground_truth.csv file can be generated for any split in the format: <filename,class>
# Usage:  after setting up directory structure variables, just run the datax.py script.
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
dir_type = ['val','test','train']

# settings below will perform a TRAIN/VAL/TEST split of 60/20/20 percent
# perform two splits using the train_test_split method from sklearn
# first split at 80/20 and second on 75/25 on the remaining train set
TEST_SIZE = 0.25
VAL_SIZE = 0.2
# used in train split to reproduce the splits, any int value works
SEED = 79



def build_directory(class_key, d_type, names, ground_truth=False):
  # create directory /OUTPUT_DIR/<val or test or train>/<tels or not_tels>
  try:
    os.makedirs(os.path.join(OUTPUT_DIR,d_type,class_key))
  except FileExistsError:
    # already exists
    pass
  # create a ground_truth.csv file if needed and get a writer ref

  if ground_truth:
    gt_file = os.path.join(OUTPUT_DIR,d_type,'ground_truth.csv')
    with open(gt_file, 'a') as gt:
      writer = csv.writer(gt)
      for f in names:
         shutil.copy(os.path.join(class_map[class_key], f), os.path.join(OUTPUT_DIR,d_type,class_key,f))
         writer.writerow([f,class_key])
  else:
    for f in names:
      shutil.copy(os.path.join(class_map[class_key], f), os.path.join(OUTPUT_DIR,d_type,class_key,f))




if __name__ == "__main__":
  # read class_map keys that corresspond to classes
  for d in class_map:
    # get the files for a given class
    X = os.listdir(class_map[d])
    print(d, " -class- ",len(X))
    # split the first time into train, validate
    train, validate = model_selection.train_test_split(X, test_size=VAL_SIZE, random_state=SEED)
    print("Val: ", len(validate))
    # build the validate dir
    build_directory(d, dir_type[0], validate, ground_truth=True)
    # split the second time from the remaining train
    train, test = model_selection.train_test_split(train, test_size=TEST_SIZE, random_state=SEED)
    # build the training directory, no ground_truth file
    build_directory(d, dir_type[2],train)
    # build the test directory
    build_directory(d,dir_type[1], test, ground_truth=True)
    print("Train: ", len(train))
    print("Test: ", len(test))




