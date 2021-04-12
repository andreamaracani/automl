import tensorflow as tf
from absl import app
from absl import flags
import json
import os
import subprocess
import sys

FLAGS = flags.FLAGS
CONFIG_ICUBWORLD = "config/icubworld/"
CONFIG_EFFICIENTDET = "config/efficientdet/"
CONFIG_TRAINING = "config/training/"
HPARAMS = CONFIG_ICUBWORLD + "icubw_config.yaml"

# specify the model to evaluate
flags.DEFINE_string('dataset', 'icubw', 'Just icubw for now') #TODO: add coco and pascal
flags.DEFINE_string('model', '', 'Name of the trained model to evaluate')

# specify what to do
flags.DEFINE_bool('save', False, 'True to save the model')
flags.DEFINE_string('eval', 'test', 'Select the folder to evaluate, test/train or None') #TODO: add custom folder

SAVED_MODEL_FOLDER = "saved_model/"
INFERENCE_FOLDER = "outputs/"


def save_model(model_name, ckpt_path, hparams, saved_model_dir):
    subprocess.call([sys.executable,
                     'model_inspect.py',
                     '--runmode=saved_model',
                     '--model_name=' + model_name,
                     '--ckpt_path=' + ckpt_path,
                     '--hparams=' + hparams,
                     '--saved_model_dir=' + saved_model_dir
                     ])

def save_outputs(model_name, saved_model_dir, input_image, output_image_dir):
    subprocess.call([sys.executable,
                     'model_inspect.py',
                     '--runmode=saved_model_infer',
                     '--model_name=' + model_name,
                     '--saved_model_dir=' + saved_model_dir,
                     '--input_image=' + input_image,
                     '--output_image_dir=' + output_image_dir
                     ])


def json2dict(file):
    with open(file) as f:
        data = json.load(f)
    return data

def main(_):

    # Check an existing dataset has been selected
    assert FLAGS.dataset == "icubw", "Dataset not set to icubw!"

    # Check an existing json training file has been selected
    assert os.path.isfile(CONFIG_TRAINING + FLAGS.model + ".json"), "File JSON named " + FLAGS.model + " not found!"

    # Model folder based on dataset name
    if FLAGS.dataset == "icubw":
        model_folder = json2dict(CONFIG_EFFICIENTDET + "folderTree.json")["fine_tuning_icubw"]

    # Model name
    model_name = FLAGS.model

    # Check if model folder exists
    assert os.path.isdir(model_folder + model_name), "Model: " + model_name + " not found!"

    # Backbone
    backbone = json2dict(CONFIG_TRAINING + FLAGS.model + ".json")["model"]

    # Saved Model Folder
    saved_model_folder = model_folder + model_name + "/saved_model/"

    if FLAGS.save:
        if not tf.io.gfile.exists(saved_model_folder):
            tf.io.gfile.makedirs(saved_model_folder)

        print("Saving ", model_name, " in ", saved_model_folder)

        save_model(model_name=backbone,
                   ckpt_path=model_folder + model_name,
                   hparams=model_folder + model_name + "/config.yaml",
                   saved_model_dir=saved_model_folder)

    if FLAGS.eval is not None:
        assert FLAGS.eval == "test" or FLAGS.eval == "train", "Evaluation should be set to train or test."
        assert tf.io.gfile.exists(saved_model_folder), "Saved model folder does not exists!"

        output_folder = model_folder + model_name + "/outputs/"
        if not tf.io.gfile.exists(output_folder):
            tf.io.gfile.makedirs(output_folder)

        print("Saving outputs of " + FLAGS.eval + " set, in " + output_folder)
        IMAGE_PATH = "images/1.jpg"

        print(backbone)
        print(saved_model_folder)
        print(IMAGE_PATH)
        print(output_folder)
        save_outputs(model_name=backbone,
                     saved_model_dir=saved_model_folder,
                     input_image=IMAGE_PATH,
                     output_image_dir=output_folder)

if __name__ == '__main__':

  app.run(main)