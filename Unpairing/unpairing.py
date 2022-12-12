import hdf5storage
import numpy as np
from scipy import io
from PIL import Image
import scipy.io as scio
import h5py


def unpair_images_mir(percentage, back=0):
    mat_tags = io.loadmat('Original Data/MIR-Flickr25K/mirflickr25k-yall.mat')
    tags = mat_tags['YAll']

    unpaired_sample = ([0] * 1386)
    count = 0

    # Percentages
    if back == 0:
        for i in range(0, 10000):
            if count < percentage:
                tags[i] = unpaired_sample
            count += 1
            if count == 99:
                count = 0
        unpaired_tags = np.asarray(tags, dtype=int)
        data_tags = {'YAll': unpaired_tags}
        io.savemat('Unpaired Data/MIR-Flickr25K UI/mirflickr25k-yall-unpaired-' + str(percentage) + '.mat', data_tags)

    elif back == 1:
        for i in range(0, 10000):
            if count > 100 - percentage:
                tags[i] = unpaired_sample
            count += 1
            if count == 99:
                count = 0
        unpaired_tags = np.asarray(tags, dtype=int)
        data_tags = {'YAll': unpaired_tags}
        io.savemat('Unpaired Data/MIR-Flickr25K UI/mirflickr25k-yall-unpaired-' + str(percentage) + '-back.mat', data_tags)

    else:
        raise SystemExit('Wrong Back')


def unpair_text_mir(percentage):
    mir = h5py.File('Original Data/MIR-Flickr25K/mirflickr25k-iall.mat', 'r', libver='latest', swmr=True)
    images = np.array(mir['IAll'])
    mir.close()

    unpaired_sample = np.zeros(shape=(3, 224, 224), dtype=np.uint8)
    count = 0

    # Percentages
    for i in range(0, len(images)):
        if count < percentage:
            images[i] = unpaired_sample
        count += 1
        if count == 99:
            count = 0

    unpaired_images = np.asarray(images, dtype=np.uint8).transpose()
    data_images = {'IAll': unpaired_images}
    hdf5storage.write(data_images, '.',
                      'Unpaired Data/MIR-Flickr25K UT/mirflickr25k-iall-unpaired-' + str(percentage) + '.mat',
                      matlab_compatible=True)


def unpair_text_mir_dadh(percentage):
    data_file = scio.loadmat("Original Data/FLICKR-25K.mat")
    images = data_file['images'][:]

    unpaired_sample = ([0] * 4096)
    count = 0

    # Percentages
    for i in range(0, 10000):
        if count < percentage:
            images[i] = unpaired_sample
        count += 1
        if count == 99:
            count = 0

    unpaired_images = np.asarray(images, dtype=int)
    data_images = {'IAll': unpaired_images}
    io.savemat('Unpaired Data/MIR-Flickr25K UT (DADH)/mirflickr25k-iall-unpaired-' + str(percentage) + '.mat',
               data_images)


def unpair_images_nus(percentage, back=0):
    mat_tags = io.loadmat('Original Data/NUS-WIDE-TC21.mat')
    tags = mat_tags['text']
    unpaired_sample = ([0] * 1000)
    count = 0

    # Percentages
    if back == 0:
        for i in range(0, 10000):
            if count < percentage:
                tags[i] = unpaired_sample
            count += 1
            if count == 99:
                count = 0
        unpaired_tags = np.asarray(tags, dtype=int)
        data_tags = {'YAll': unpaired_tags}
        io.savemat('Unpaired Data/NUS-WIDE UI/nus-wide-tc21-yall-unpaired-' + str(percentage) + '.mat', data_tags)

    elif back == 1:
        for i in range(0, 10000):
            if count > 100 - percentage:
                tags[i] = unpaired_sample
            count += 1
            if count == 99:
                count = 0
        unpaired_tags = np.asarray(tags, dtype=int)
        data_tags = {'YAll': unpaired_tags}
        io.savemat('Unpaired Data/NUS-WIDE UI/nus-wide-tc21-yall-unpaired-' + str(percentage) + '-back.mat', data_tags)


def unpair_text_nus(percentage):
    nuswide = h5py.File("Original Data/NUS-WIDE-TC21/nus-wide-tc21-iall.mat", 'r', libver='latest', swmr=True)
    images = np.array(nuswide['IAll'])
    nuswide.close()

    unpaired_sample = np.zeros(shape=(3, 224, 224), dtype=np.uint8)
    count = 0

    # Percentages
    for i in range(0, 10000):
        if count < percentage:
            images[i] = unpaired_sample
        count += 1
        if count == 99:
            count = 0

    unpaired_images = np.asarray(images, dtype=np.uint8).transpose()
    data_images = {'IAll': unpaired_images}
    hdf5storage.write(data_images, '.', 'Unpaired Data/NUS-WIDE UT/nus-wide-tc21-iall-unpaired-' + str(percentage) + '.mat',
                      matlab_compatible=True)


def unpair_text_nus_dadh(percentage):
    data_file = scio.loadmat("Original Data/NUS-WIDE-TC21.mat")
    images = data_file['image'][:]

    unpaired_sample = ([0] * 4096)
    count = 0

    # Percentages
    for i in range(0, 10000):
        if count < percentage:
            images = unpaired_sample
        count += 1
        if count == 99:
            count = 0

    unpaired_images = np.asarray(images, dtype=np.uint8).transpose()
    data_images = {'IAll': unpaired_images}
    hdf5storage.write(data_images, '.', 'Unpaired Data/NUS-WIDE UI (DADH)/nus-wide-tc21-iall-unpaired-' + str(percentage) + '.mat',
                      matlab_compatible=True)


# unpair_images_mir(20)

unpair_text_mir(20)

# unpair_text_mir_dadh(20)
