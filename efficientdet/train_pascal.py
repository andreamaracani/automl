import tensorflow as tf
from absl import app
from absl import flags
import urllib.request
from tqdm import tqdm
import json
import tarfile
import os

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = os.path.realpath(__file__)
print(full_path + "\n")

print("This file directory and name")
path, filename = os.path.split(full_path)
print(path + ' --> ' + filename + "\n")

print("This file directory only")
print(os.path.dirname(full_path))
from urllib.parse import urlparse
import sys
import subprocess


def main(_):

    # fine-tuning training

    subprocess.call([sys.executable,
                     "main.py",
                     "--mode=train_and_eval",
                     "--train_file_pattern=../tmp/pascal/tfrecords/tfrecords*.tfrecord",
                     "--val_file_pattern=../tmp/pascal/tfrecords/tfrecords*.tfrecord",
                     "--model_name=efficientdet-d0",
                     "--model_dir=../tmp/efficientdet/pascal/train_00",
                     "--ckpt=../tmp/efficientdet/coco2/efficientdet-d0/",
                     "--train_batch_size=64",
                     "--eval_batch_size=64 ",
                     "--eval_samples=1024", 
                     "--num_examples_per_epoch=5717", 
                     "--num_epochs=300",  
                     "--hparams=config/pascal/pascal.yaml",
                     "--strategy=gpus"])


if __name__ == '__main__':
  app.run(main)