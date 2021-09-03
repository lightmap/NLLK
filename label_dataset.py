import os
import glob
import argparse
import random


parser = argparse.ArgumentParser(description='Create train, test and val label files')
parser.add_argument('dataset_dir', type=str,
                    help='directory with dataset')
parser.add_argument('output_directory', type=str,
                    help='output directory to label files')

parser.add_argument('--create_test', type=bool, default=False,
                    help='create test file')


args = vars(parser.parse_args())
dataset_dir = args["dataset_dir"].strip('/')
label_dir = args["output_directory"].strip('/')
create_test = args["create_test"]

files = glob.glob(dataset_dir + '/' + '*.png')

random.shuffle(files)
train_set = files[0: int(len(files) * .80)]
test_set = []
val_set = []
if create_test:
    test_set = files[int(len(files) * .80): int(len(files) * .90)]
    val_set = files[int(len(files) * .90): int(len(files) * 1.0)]
else:
    val_set = files[int(len(files) * .80): int(len(files) * 1.0)]    
    
with open(label_dir + "/train.txt", 'w') as out:
    for file in train_set:
        out.write(os.path.abspath(file) + "\n")

if create_test:
    with open(label_dir + "/test.txt", 'w') as out:
        for file in test_set:
            out.write(os.path.abspath(file) + "\n")


with open(label_dir + "/val.txt", 'w') as out:
    for file in val_set:
        out.write(os.path.abspath(file) + "\n")


