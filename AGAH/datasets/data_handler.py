import os
import numpy as np
import h5py
import scipy.io as scio


def preprocess(x, mean, std):
    mean, std = np.array(mean), np.array(std)
    return (x - mean.reshape(3, 1, 1)) / (std.reshape(3, 1, 1) + 1e-5)


def load_data(path, type='flickr25k'):
    if type == 'flickr25k':
        return load_flickr25k(path)
    else:
        return load_nus_wide(path)


def load_flickr25k(path):
    percentage = 40
    #image_file = h5py.File(os.path.join(path, 'mirflickr25k-iall.mat'))
    image_file = h5py.File('data/MIR-Flickr25K/Unpaired-Image/mirflickr25k-big-iall-unpaired-'+str(percentage)+'.mat')
    images = image_file['IAll'][:]
    image_file.close()


    #tag_data = scio.loadmat(os.path.join(path, 'mirflickr25k-yall.mat'))
    tag_data = scio.loadmat('data/MIR-Flickr25K/Unpaired-Text/mirflickr25k-yall-unpaired-'+str(percentage)+'-back.mat')
    tags = tag_data['YAll']

    label_data = scio.loadmat(os.path.join(path, 'mirflickr25k-lall.mat'))
    labels = label_data['LAll']

    return images, tags, labels


def load_nus_wide(path_dir):
    percentage = 80
    #image_file = h5py.File(os.path.join(path_dir, 'nus-wide-tc21-iall.mat'))
    image_file = h5py.File('data/NUS-WIDE-TC21/Unpaired-Image/nus-wide-tc21-iall-unpaired-'+str(percentage)+'.mat')
    images = image_file['IAll'][:]
    image_file.close()

    tag_data = scio.loadmat(os.path.join(path_dir, 'nus-wide-tc21-yall.mat'))
    #tag_data = scio.loadmat('data/NUS-WIDE-TC21/Unpaired-Text/nus-wide-tc21-yall-unpaired-'+str(percentage)+'-back.mat')
    tags = tag_data['YAll']

    label_data = scio.loadmat(os.path.join(path_dir, 'nus-wide-tc21-lall.mat'))
    labels = label_data['LAll']

    return images, tags, labels


def data_enhance(images, tags, labels, c=0.5):
    num = images.shape[0]
    ind1 = np.random.permutation(num // 2)
    ind2 = np.random.permutation(num // 2)
    inhanced_imgs = c * images[ind1] + (1 - c) * images[ind2]
    inhanced_tags = c * tags[ind1] + (1 - c) * tags[ind2]
    inhanced_labels = c * labels[ind1] + (1 - c) * labels[ind2]
    new_imgs = np.concatenate((images, inhanced_imgs))
    new_tags = np.concatenate((tags, inhanced_tags))
    new_labels = np.concatenate((labels, inhanced_labels))
    return new_imgs, new_tags, new_labels


def load_pretrain_model(path):
    return scio.loadmat(path)

