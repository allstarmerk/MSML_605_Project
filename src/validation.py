import numpy as np
from pathlib import Path

#the purpose is to check inputs and outputs for common errors before steps in pipline. without we can get silent errors where it runs fine but produces wrong results no errors.
# can use assert statments for checks. If condition=False it  makes a error with discription of why  it failed
VALID_SPLITS = {"train", "val", "test"}

def validate_pairs(pairs, labels, split_name):  #checks if the pairs and labels are valid for the given split
    #check if split name is valid forst

    # check the types

    #check the shape

    # check the len matches

    #check labels are binary

    #check not empty

def validate_scores(scores, pairs):
    #verify score count matches pair count

    #check for non NAN values or infinite values from numpy math errors

def validate_threshold(threshold):
    #check if threshold is a number

    #check if threshold is in valid range for cosine

def validate_metrics(metrics):


def validate_pair_files(pairs_dir, split_name):
    #check if the pair files exist for the given split

    #
