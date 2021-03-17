import hashlib
import io
import os
import os.path
import json

from absl import logging

from lxml import etree
import PIL.Image
import tensorflow as tf

from dataset import tfrecord_util


SETS = ['train', 'test']

GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.


def get_image_id(filename):
    """Convert a string to a integer."""
    # Warning: this function is highly specific to pascal filename!!
    # Given filename like '2008_000002', we cannot use id 2008000002 because our
    # code internally will convert the int value to float32 and back to int, which
    # would cause value mismatch int(float32(2008000002)) != int(2008000002).
    # COCO needs int values, here we just use a incremental global_id, but
    # users should customize their own ways to generate filename.
    del filename
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def get_ann_id():
    """Return unique annotation id across images."""
    global GLOBAL_ANN_ID
    GLOBAL_ANN_ID += 1
    return GLOBAL_ANN_ID


def dict_to_tf_example(data,
                       image_path,
                       label_map_dict):
    """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by running
      tfrecord_util.recursive_parse_xml_to_dict)
    images_dir: Path to the directory holding raw images.
    label_map_dict: A map from string label names to integers ids.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
    full_path = os.path.join(image_path)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])
    image_id = get_image_id(data['filename'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    area = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            area.append((xmax[-1] - xmin[-1]) * (ymax[-1] - ymin[-1]))
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                    tfrecord_util.int64_feature(height),
                'image/width':
                    tfrecord_util.int64_feature(width),
                'image/filename':
                    tfrecord_util.bytes_feature(data['filename'].encode('utf8')),
                'image/source_id':
                    tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
                'image/key/sha256':
                    tfrecord_util.bytes_feature(key.encode('utf8')),
                'image/encoded':
                    tfrecord_util.bytes_feature(encoded_jpg),
                'image/format':
                    tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                    tfrecord_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                    tfrecord_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                    tfrecord_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                    tfrecord_util.float_list_feature(ymax),
                'image/object/area':
                    tfrecord_util.float_list_feature(area),
                'image/object/class/text':
                    tfrecord_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                    tfrecord_util.int64_list_feature(classes),
                'image/object/difficult':
                    tfrecord_util.int64_list_feature(difficult_obj),
                'image/object/truncated':
                    tfrecord_util.int64_list_feature(truncated),
                'image/object/view':
                    tfrecord_util.bytes_list_feature(poses),
            }))
    return example


def json2dict(file):

    if type(file) is dict:
        return file

    with open(file) as f:
        data = json.load(f)
    return data


def create_tfrecords(config_set, config_folders):

    # parameters for the creation of tfrecords
    params = json2dict(config_set)

    # folder structure of the project
    folders = json2dict(config_folders)

    # path to the list file of this set
    list_path = folders["sets"] + params["file_list"]

    # path to the image folder of this set
    images_folder = folders[params["set_name"] + "_images"]

    # path to the annotation folder of this set
    labels_folder = folders[params["set_name"] + "_labels"]

    # path of the output folder
    out_folder = folders[params["set_name"] + "_tfrecords"]


    if not tf.io.gfile.exists(out_folder):
        tf.io.gfile.makedirs(out_folder)
    logging.info('Writing to output directory: %s', out_folder)

    writers = [
        tf.io.TFRecordWriter(out_folder + '/icubw-%05d-of-%05d.tfrecord' % (i, params["num_shards"]))
        for i in range(params["num_shards"])
    ]

    examples_list = tfrecord_util.read_examples_list(list_path)

    logging.info('Reading from iCub World dataset.')
    for idx, example in enumerate(examples_list):
        if params["num_images"] and idx >= params["num_images"]:
            break
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(labels_folder, example + '.xml')
        with tf.io.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']

        img_path = os.path.join(images_folder + example + ".jpg")

        if os.path.isfile(img_path):
            tf_example = dict_to_tf_example(
            data,
            img_path,
            params["classes_dict"]
            )
            writers[idx % params["num_shards"]].write(tf_example.SerializeToString())
        else:
            print(img_path)

    for writer in writers:
        writer.close()


