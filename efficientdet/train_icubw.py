import tensorflow as tf
from absl import app
from absl import flags
import urllib.request
from tqdm import tqdm
import json
import tarfile
import os
from urllib.parse import urlparse
import subprocess
import sys

from dataset import create_icubw_tfrecord

FLAGS = flags.FLAGS
CONFIG_ICUBWORLD = "config/icubworld/"
CONFIG_EFFICIENTDET = "config/efficientdet/"
CONFIG_TRAINING = "config/training/"
HPARAMS = CONFIG_ICUBWORLD + "icubw_config.yaml"


flags.DEFINE_bool('download_training', False, 'True to download training set.')
flags.DEFINE_bool('download_test', False, 'True to download test set.')
flags.DEFINE_bool('train_tf', False, 'True to craete train TFrecords.')
flags.DEFINE_bool('test_tf', False, 'True to craete test TFrecords.')
flags.DEFINE_string('train_config', "training_03.json", 'The training json config file name.')


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_all(urls, output_path):

    if not tf.io.gfile.exists(output_path):
        tf.io.gfile.makedirs(output_path)

    file_paths = []

    for u in urls:

        # get file name from url
        file_name = os.path.basename(urlparse(u).path)
        print("Downloading " + file_name + "...")
        output_path = output_path + file_name
        download_url(u, output_path)

        file_paths.append(output_path)

    return file_paths


def get_file_name(path):
    file_name = os.path.basename(path)
    return file_name[:file_name.index('.')]


def extract_all(tar_files, output_path, remove_base_folder=True):

    # for all files to be extracted
    for path in tar_files:

        # the base folder name inside archive is the same as the archive name
        base_folder_name = get_file_name(path) + "/"

        with tarfile.open(path) as tar:

            # extract all members
            print("Extracting ", path, "to", output_path)
            for member in tqdm(tar):
                if remove_base_folder:
                    member.path = member.path[len(base_folder_name):]

                tar.extract(member, path=output_path)


def json2dict(file):

    with open(file) as f:
        data = json.load(f)
    return data


def main(_):

    # dataset
    download_urls = json2dict(CONFIG_ICUBWORLD + "download.json")
    folder_paths = json2dict(CONFIG_ICUBWORLD + "folderTree.json")

    # model
    ckpt_urls = json2dict(CONFIG_EFFICIENTDET + "checkpoints.json")
    ckpt_paths = json2dict(CONFIG_EFFICIENTDET + "folderTree.json")

    # training parameters
    training_params = json2dict(CONFIG_TRAINING + FLAGS.train_config)

    if FLAGS.download_training:
        train_images_paths = download_all(urls=download_urls["train_images"], output_path=folder_paths["download"])
        train_labels_paths = download_all(urls=download_urls["train_labels"], output_path=folder_paths["download"])
        extract_all(tar_files=train_images_paths, output_path=folder_paths["training_images"], remove_base_folder=True)
        extract_all(tar_files=train_labels_paths, output_path=folder_paths["training_labels"], remove_base_folder=True)

    if FLAGS.download_test:
        test_images_paths = download_all(urls=download_urls["test_images"], output_path=folder_paths["download"])
        test_labels_paths = download_all(urls=download_urls["test_labels"], output_path=folder_paths["download"])
        extract_all(tar_files=test_images_paths, output_path=folder_paths["test_images"], remove_base_folder=True)
        extract_all(tar_files=test_labels_paths, output_path=folder_paths["test_labels"], remove_base_folder=True)


    # create TFrecords
    if FLAGS.train_tf:
        print("Creating training tfrecords...")
        create_icubw_tfrecord.create_tfrecords(CONFIG_ICUBWORLD + "train.json", folder_paths)

    if FLAGS.test_tf:
        print("Creating test tfrecords...")
        create_icubw_tfrecord.create_tfrecords(CONFIG_ICUBWORLD + "test.json", folder_paths)

    # checkpoints

    # if coco-ckpts folder does not exist, create it.
    if not tf.io.gfile.exists(ckpt_paths["efficientdet_coco"]):
        tf.io.gfile.makedirs(ckpt_paths["efficientdet_coco"])

    # if coco-ckpt of specified network does not exist, download it and unizip it.
    model_name = training_params["model"]
    model_path = ckpt_paths[model_name]
    if not tf.io.gfile.exists(model_path):
        ckpt_tar = download_all(ckpt_urls[model_name], ckpt_paths["efficientdet_coco"])
        extract_all(ckpt_tar, ckpt_paths["efficientdet_coco"], remove_base_folder=False)

    # if output folder does not exists, create it.
    output_folder = ckpt_paths["fine_tuning_icubw"] + training_params["training_name"]
    if not tf.io.gfile.exists(output_folder):
        tf.io.gfile.makedirs(output_folder)

    # fine-tuning training
    subprocess.call([sys.executable,
                     "main.py",
                     "--mode=" + training_params["mode"],
                     "--train_file_pattern=" + folder_paths["training_tfrecords"] + "icubw*.tfrecord",
                     "--val_file_pattern=" + folder_paths["test_tfrecords"] + "icubw*.tfrecord",
                     "--model_name=" + model_name,
                     "--model_dir=" + output_folder,
                     "--ckpt=" + model_path,
                     "--train_batch_size=" + str(training_params["train_batch_size"]),
                     "--eval_batch_size=" + str(training_params["eval_batch_size"]),
                     "--eval_samples=" + str(training_params["eval_samples"]),
                     "--eval_after_train=" + str(bool(training_params["eval_after_train"])),
                     "--num_examples_per_epoch=" + str(training_params["num_examples_per_epoch"]),
                     "--num_epochs=" + str(training_params["num_epochs"]),
                     "--hparams=" + HPARAMS,
                     "--strategy=" + training_params["strategy"]])


    #print(os.getcwd())
    # "python main.py --mode = train " \
    # "--train_file_pattern = tfrecord / pascal *.tfrecord " \
    # "--val_file_pattern = tfrecord / pascal *.tfrecord " \
    # "--model_name = efficientdet - d0 " \
    # "--model_dir = / tmp / efficientdet - d0 - finetune " \
    # "--ckpt = efficientdet - d0 " \
    # "--train_batch_size = 64 " \
    # "--eval_batch_size = 64 - -eval_samples = 1024 " \
    # "--num_examples_per_epoch = 5717 --num_epochs = 50 " \
    # "--hparams = voc_config.yaml" \
    # "--strategy = gpus"

if __name__ == '__main__':

  app.run(main)