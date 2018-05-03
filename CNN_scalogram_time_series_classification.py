# bl - BaseLine
# rd - Rolling defect
# ir - Inner Raceway
# or - Outer Raceway
# labels are strings and classes as integers

import os
import sys
import math
import scipy.io
import re
import pywt
import numpy as np
from matplotlib import pyplot
import PIL
import glob
import random
import lmdb
import datetime
import imageio
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
from caffe2.python import (
   brew,
   core,
   model_helper,
   net_drawer,
   optimizer,
   visualize,
   workspace,
)


def get_sub_signals(fault_dir_path, sub_signal_length):
    """
    Reads a set of .mat files. Each .mat file in a <fault_dir_path> has a signal corresponding to a type of fault.
    Creates sub-signals of length <sub_signal_length> from each signal in each .mat file. Ignores <n_ignore_end> number
    of data points of the signal.
    :param fault_dir_path: string - fully qualified name of a directory
    :param sub_signal_length: integer - no of data points in the sub-signal
    :return sub_signals: list of 1-D arrays - Each 1-D array is a sub-signal
    """
    f_names = os.listdir(fault_dir_path)
    sub_signls = list()
    for f_name in f_names:
        if f_name.endswith('.mat'):
            f_path = os.path.join(fault_dir_path, f_name)
            mtlb_f_content = scipy.io.loadmat(f_path)  # load the content of a matlab file
            for key, value in mtlb_f_content.items():
                if re.match('X..._DE_time', key):
                    full_signal = value
                    full_signal_len = full_signal.shape[0]
                    n_sub_signals = full_signal_len // sub_signal_length
                    n_ignore_end = full_signal_len % sub_signal_length
                    truncated_signal = full_signal[: -n_ignore_end]  # remove the last *n_ignore_end* data points
                    sub_signls.extend(np.split(truncated_signal, n_sub_signals))
                    break  # Sometimes, a key-value pair occurs more than once
    return sub_signls


def get_scalogram(signal):
    """
    Create a scalogram using morlet wavelet. Scales in continuous wavelet transform(cwt) range from 1 to length of signal.
    Square of cwt coefficients is the scalogram
    :param signal: 1-D array - time series
    :return scalogram: 2-D array - shape is [len_of_signal, len_of_signal]
    """
    # To-Do: generalize the input <signal> to be any iterable (Series, list, tuple)
    signal_length = signal.shape[0]
    # convert a (n, 1) array into a (n,) array
    signal = signal.reshape(signal_length)
    scales = np.arange(1, signal_length + 1)
    coefs, freqs = pywt.cwt(data=signal, scales=scales, wavelet='morl')
    scalogram = np.multiply(coefs, coefs)
    return scalogram


def get_scaled_data(data, new_min=0.0, new_max=1.0):
    """
    Scales <data> to be between <new_min> and <new_max>
    :param data: numpy array. Works with other iterables like Series, List, Tuple
    :param new_min: float
    :param new_max: float
    :return scaled_data: Same data-type as <data>
    """
    old_min = np.min(data)
    old_max = np.max(data)
    if (old_min != old_max) and (new_min != new_max):
        old_range = old_max - old_min
        new_range = new_max - new_min
        scaled_data = (((data - old_min) * new_range) / old_range) + new_min
    else:
        scaled_data = (new_max + new_min) / 2.0
    return scaled_data


# def check_scalo_img(data_path):
#     img_f_names = glob.glob(data_path + '/*.tiff')
#     for img_f_name in img_f_names:
#         img = Image.open(img_f_name)
#         img_arr = np.array(img.getdata()).reshape(img.size[0], img.size[1])
#         print(img_arr.shape)
#         print(img_arr[0][:5])
#         sys.exit()


def get_labels_to_classes_map(labels_path):
    """
    Reads a file with labels and returns a map whose keys are labels(strings) and values are classes(integers).
    The file <labels_path> has one label on each line.
    Labels are strings and classes are integers. If 'yes' and 'no' are labels. 0 and 1 classes
    :param labels_path: string - fully qualified name of the file
    :return labels_to_classes_map: map
    """
    labels_fo = open(labels_path, "r")
    labels_to_classes_map = dict()
    class_ = 0
    labels = labels_fo.readlines()
    for label in sorted(labels):
        label = label.rstrip()
        labels_to_classes_map[label] = class_
        class_ += 1
    labels_fo.close()
    return labels_to_classes_map


def create_img_to_class_files(data_path, lbls_to_cls_map):
    """
    Creates three .txt files. One each for training, testing and validation. Each line of these files have fully
    qualified names of image files and the corresponding class
    Assumes that image files have are named as <index>_<label>.tiff (eg. 25_baseline.tiff)
    Training, validation and testing ratio is 75%, 15% and 15% respectively
    :param data_path:  string - fully qualified name of the directory with all the data
    :param lbls_to_cls_map: map
        Keys are labels(strings) and values are classes(integers)
    :return img_to_class_path: map
        Keys are 'train', 'val' and 'test'. Values are fully qualified file names.
    """
    img_to_class_paths = {
        'train': os.path.join(data_path, 'train_img_classes.txt'),
        'val': os.path.join(data_path, 'val_img_classes.txt'),
        'test': os.path.join(data_path, 'test_img_classes.txt')
    }
    train_class_map_fo = open(img_to_class_paths['train'], "w")
    val_class_map_fo = open(img_to_class_paths['val'], "w")
    test_class_map_fo = open(img_to_class_paths['test'], "w")
    img_f_names = glob.glob(data_path + '/*.tiff')
    random.shuffle(img_f_names)
    n_samples = len(img_f_names)
    train_ratio = 0.7
    val_ratio = 0.15
    # test_ratio = 0.15
    n_train_samples = n_samples * train_ratio
    n_val_samples = n_samples * val_ratio
    # n_test_samples = n_samples - (n_train_samples + n_val_samples)
    idx = 0
    for img_f_name in img_f_names:
        idx += 1
        if idx <= n_train_samples:
            img_label = img_f_name.split('_')[-1].split('.')[0]
            class_ = str(lbls_to_cls_map[img_label])
            train_class_map_fo.write(img_f_name + ' ' + class_ + '\n')
        elif idx <= (n_train_samples + n_val_samples):
            img_label = img_f_name.split('_')[-1].split('.')[0]
            class_ = str(lbls_to_cls_map[img_label])
            val_class_map_fo.write(img_f_name + ' ' + class_ + '\n')
        else:
            img_label = img_f_name.split('_')[-1].split('.')[0]
            class_ = str(lbls_to_cls_map[img_label])
            test_class_map_fo.write(img_f_name + ' ' + class_ + '\n')
    train_class_map_fo.close()
    val_class_map_fo.close()
    test_class_map_fo.close()
    return img_to_class_paths


def write_lmdb_files(data_path, img_to_class_paths):
    """
    Writes Lightning Memory-Mapped Database(lmdb) files. One file each for training, testing and validation
    Converts [n, n] image array to [n, n, 1] array
    Converts from height-width-channel(HWC) to channel-height-width (CHW)
    :param data_path: string - fully qualified name of the directory with all data files
    :param img_to_class_paths: map
          Keys are 'train', 'val' and 'test'. Values are fully qualified file names. Each line of these files have fully
          qualified names of image files and the corresponding class
    :return lmdb_paths: map
          Keys are 'train', 'val' and 'test'. Values are fully qualified directory names. Each directory has lmdb for
          training, validation and testing data
    """
    def write_lmdb(img_to_class_path, lmdb_path):
        img_to_class_fo = open(img_to_class_path, "r")
        lmdb_map_size = 1 << 40
        env = lmdb.Environment(lmdb_path, map_size=lmdb_map_size)
        with env.begin(write=True) as txn:  # txn is Transaction object
            count = 0
            for line in img_to_class_fo.readlines():
                line = line.rstrip()
                img_path = line.split()[0]
                img_class = int(line.split()[1])
                img_data = imageio.imread(img_path).astype(np.float32)  # shape is (n, n)
                img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], 1)  # shape is (n, n, 1)
                # convert from height-width-channel(HWC) to channel-height-width (CHW)
                img_data = np.transpose(img_data, (2, 0, 1))
                tensor_protos = caffe2_pb2.TensorProtos()
                img_tensor = tensor_protos.protos.add()
                img_tensor.dims.extend(img_data.shape)
                img_tensor.data_type = 1
                flatten_img = img_data.reshape(np.prod(img_data.shape))
                img_tensor.float_data.extend(flatten_img)
                img_class_tensor = tensor_protos.protos.add()
                img_class_tensor.data_type = 2
                img_class_tensor.int32_data.append(img_class)
                txn.put(
                    '{}'.format(count).encode('ascii'),
                    tensor_protos.SerializeToString()
                )
                if count % 10 == 0:
                    print("Inserted {} rows".format(count))
                count += 1
        print("Inserted {} rows".format(count))
        print("\nLMDB saved at " + lmdb_path + "\n\n")
        img_to_class_fo.close()

    lmdb_paths = {
        'train': os.path.join(data_path, 'train_lmdb'),
        'val': os.path.join(data_path, 'val_lmdb'),
        'test': os.path.join(data_path, 'test_lmdb')
    }
    for i in ['train', 'val', 'test']:
        if not os.path.exists(lmdb_paths[i]):
            write_lmdb(img_to_class_paths[i], lmdb_paths[i])
        else:
            print(lmdb_paths[i], " already exists!")
    return lmdb_paths


def add_input(model, batch_size, db, db_type):
    data, label = brew.db_input(
        model=model,
        blobs_out=["data", "label"],
        batch_size=batch_size,
        db=db,
        db_type=db_type,
    )
    # Usually images have uint8 values (0 to 255).
    # Here the images are already scaled to have float values between 0.0 and 1.0.
    # So, cast unit8 values to float and scaling is not required here.
    # Scaling was done when scalogram-images were written to disk
    data = model.StopGradient(data, data)
    return data, label


def update_dims(height, width, kernel, stride, pad):
    new_height = ((height - kernel + (2 * pad)) // stride) + 1
    new_width = ((width - kernel + (2 * pad)) // stride) + 1
    return new_height, new_width


def add_cnn_model_1(model, data, num_classes, image_height, image_width, image_channels):
    conv1 = brew.conv(model, data, 'conv1', dim_in=image_channels, dim_out=32, kernel=5, stride=1, pad=2)
    h, w = update_dims(height=image_height, width=image_width, kernel=5, stride=1, pad=2)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=3, stride=2)
    h, w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)
    relu1 = brew.relu(model, pool1, 'relu1')
    conv2 = brew.conv(model, relu1, 'conv2', dim_in=32, dim_out=32, kernel=5, stride=1, pad=2)
    h, w = update_dims(height=h, width=w, kernel=5, stride=1, pad=2)
    relu2 = brew.relu(model, conv2, 'relu2')
    pool2 = brew.average_pool(model, relu2, 'pool2', kernel=3, stride=2)
    h, w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)
    conv3 = brew.conv(model, pool2, 'conv3', dim_in=32, dim_out=64, kernel=5, stride=1, pad=2)
    h, w = update_dims(height=h, width=w, kernel=5, stride=1, pad=2)
    relu3 = brew.relu(model, conv3, 'relu3')
    pool3 = brew.average_pool(model, relu3, 'pool3', kernel=3, stride=2)
    h, w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)
    fc1 = brew.fc(model, pool3, 'fc1', dim_in=64 * h * w, dim_out=64)
    fc2 = brew.fc(model, fc1, 'fc2', dim_in=64, dim_out=num_classes)
    softmax = brew.softmax(model, fc2, 'softmax')
    return softmax


def add_optmzer_lossfunc(model, softmax, label):
    cross_entropy = model.LabelCrossEntropy([softmax, label], 'cross_entropy')
    loss = model.AveragedLoss(cross_entropy, "loss")
    model.AddGradientOperators([loss])  # look at documentation
    optimizer.build_sgd(
        model,
        base_learning_rate=0.01,
    )


def add_accuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


def add_check_points(model, time_stamp, checkpoint_iters, db_type):
    ITER = brew.iter(model, "iter")
    model.Checkpoint([ITER] + model.params, [], db=os.path.join(time_stamp, "cifar10_checkpoint_%05d.lmdb"),
                     db_type=db_type, every=checkpoint_iters)


def main():
    root_path = '/home/osboxes/zementis/scalogram/fault_diagnosis'
    data_path = os.path.join(root_path, 'data')
    labels_path = os.path.join(data_path, 'labels.txt')
    labels_to_classes_map = get_labels_to_classes_map(labels_path)
    fault_types_path = {
        'baseLine': os.path.join(data_path, 'raw_signals', 'baseLine'),
        'rollingDefect': os.path.join(data_path, 'raw_signals', 'rollingDefect'),
        'innerRace': os.path.join(data_path, 'raw_signals', 'innerRace'),
        'outerRace': os.path.join(data_path, 'raw_signals', 'outerRace')
    }
    sub_signal_len = 400
    # The sample rate is 12 kHz and the approximate motor speed is 1797 RPM. Therefore, there are approximately 401
    # sample points per revolution (12000 / (1797 / 60)).
    for fault_type, fault_dir_path in fault_types_path.items():
        sub_signals = get_sub_signals(fault_dir_path, sub_signal_len)
        sub_signal_idx = 0
        for sub_signal in sub_signals:
            sub_signal_idx += 1
            scalo = get_scalogram(sub_signal)
            scaled_scalo = get_scaled_data(scalo)  # scales an array to have values between 0.0 and 1.0
            img_obj = PIL.Image.fromarray(scaled_scalo)
            img_f_name = str(sub_signal_idx) + '_' + fault_type + '.tiff'
            img_obj.save(os.path.join(data_path, img_f_name))
            if sub_signal_idx == 50:
                break  # stop after creating 50 images in each class
    # Create txt files mapping image names to classes
    img_to_class_paths = create_img_to_class_files(data_path, labels_to_classes_map)
    # Create lmdb files
    lmdb_paths = write_lmdb_files(data_path, img_to_class_paths)
    model_files_path = os.path.join(root_path, 'model_files')
    if not os.path.isdir(model_files_path):
        os.makedirs(model_files_path)
    workspace.ResetWorkspace(model_files_path)
    unique_timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    checkpoint_dir = os.path.join(model_files_path, unique_timestamp)
    os.makedirs(checkpoint_dir)
    print("Checkpoint output location: ", checkpoint_dir)

    # Dataset specific params
    image_width = sub_signal_len
    image_height = sub_signal_len
    image_channels = 1
    num_classes = 4
    init_net_out_fname = 'init_net.pb'
    predict_net_out_fname = 'predict_net.pb'

    # Training params
    n_iters = 600  # total training iterations
    batch_size = 10  # batch size for training
    n_val_images = 30  # total number of validation images
    validation_interval = 50  # validate every <validation_interval> training iterations
    n_checkpoint_iters = 200  # output checkpoint db every <checkpoint_iters> iterations

    # TRAINING MODEL
    train_model = model_helper.ModelHelper(name="train_net")
    data, label = add_input(train_model, batch_size=batch_size, db=lmdb_paths['train'], db_type='lmdb')
    softmax = add_cnn_model_1(train_model, data, num_classes, image_height, image_width, image_channels)
    add_optmzer_lossfunc(train_model, softmax, label)
    add_check_points(train_model, unique_timestamp, n_checkpoint_iters, db_type="lmdb")

    # VALIDATION MODEL
    # Initialize with ModelHelper class without re-initializing params
    val_model = model_helper.ModelHelper(name="val_net", init_params=False)
    data, label = add_input(val_model, batch_size=n_val_images, db=lmdb_paths['val'], db_type='lmdb')
    softmax = add_cnn_model_1(val_model, data, num_classes, image_height, image_width, image_channels)
    add_accuracy(val_model, softmax, label)

    # DEPLOY MODEL
    # Initialize with ModelHelper class without re-initializing params
    deploy_model = model_helper.ModelHelper(name="deploy_net", init_params=False)
    # Add model definition, expect input blob called "data"
    add_cnn_model_1(deploy_model, "data", num_classes, image_height, image_width, image_channels)
    print("Training, Validation, and Deploy models all defined!")

    # Initialize and create the training network
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)
    # Initialize and create validation network
    workspace.RunNetOnce(val_model.param_init_net)
    workspace.CreateNet(val_model.net, overwrite=True)
    # Placeholder to track loss and validation accuracy
    training_loss = np.zeros(int(math.ceil(n_iters / validation_interval)))
    val_accuracy = np.zeros(int(math.ceil(n_iters / validation_interval)))
    val_count = 0
    val_iter_list = np.zeros(int(math.ceil(n_iters / validation_interval)))

    # run the network (forward & backward pass)
    for i in range(n_iters):
        workspace.RunNet(train_model.net)
        # Validate every <validation_interval> training iterations
        if (i % validation_interval) == 0:
            print("Training iter: ", i)
            training_loss[val_count] = workspace.FetchBlob('loss')
            workspace.RunNet(val_model.net)
            val_accuracy[val_count] = workspace.FetchBlob('accuracy')
            print("Loss: ", str(training_loss[val_count]))
            print("Validation accuracy: ", str(val_accuracy[val_count]) + "\n")
            val_iter_list[val_count] = i
            val_count += 1

    fig = pyplot.figure()
    fig.add_subplot(111)
    pyplot.title("Training Loss and Validation Accuracy")
    pyplot.plot(val_iter_list, training_loss, 'b')
    pyplot.plot(val_iter_list, val_accuracy, 'r')
    pyplot.xlabel("Training iteration")
    pyplot.legend(('Training Loss', 'Validation Accuracy'), loc='upper right')
    pyplot.savefig("loss_and_accuracy.png")
    pyplot.close()


    # Save trained model
    workspace.RunNetOnce(deploy_model.param_init_net)
    workspace.CreateNet(deploy_model.net, overwrite=True)
    init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
    init_net_out_path = os.path.join(checkpoint_dir, init_net_out_fname)
    predict_net_out_path = os.path.join(checkpoint_dir, predict_net_out_fname)
    with open(init_net_out_path, 'wb') as f:
        f.write(init_net.SerializeToString())
    with open(predict_net_out_path, 'wb') as f:
        f.write(predict_net.SerializeToString())
    print("Model saved as " + init_net_out_path + " and " + predict_net_out_path)


if __name__ == '__main__':
    main()
