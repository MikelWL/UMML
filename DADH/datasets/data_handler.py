import scipy.io as scio
import mat73


def load_data(path, type='mir'):
    if type == 'mir':
        return load_flickr25k(path)
    elif type == 'mir-ui':
        return load_flickr25k_unpaired_images(path)
    elif type == 'mir-ut':
        return load_flickr25k_unpaired_text(path)
    elif type == 'mir-uit':
        return load_flickr25k_unpaired_combo(path)
    elif type == 'nus':
        return load_nus_wide(path)
    elif type == 'nus-ui':
        return load_nus_wide_unpaired_image(path)
    elif type == 'nus-ut':
        return load_nus_wide_unpaired_text(path)
    elif type == 'nus-uit':
        return load_nus_wide_unpaired_combo(path)
    else:
        raise SystemExit('Wrong flag')


def load_flickr25k(path):

    data_file = scio.loadmat(path)
    images = data_file['images'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels


def load_flickr25k_unpaired_images(path):
    percentage = 20
    unpaired_file_path = 'Unpaired Data/MIR-Flickr25K UI/mirflickr25k-yall-unpaired-'+str(percentage)+'.mat'

    data_file = scio.loadmat(path)
    images = data_file['images'][:]
    images = (images - images.mean()) / images.std()
    data_file_text = scio.loadmat(unpaired_file_path)
    tags = data_file_text['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels


def load_flickr25k_unpaired_text(path):
    percentage = 20
    unpaired_file_path = 'Unpaired Data/MIR-Flickr25K UT (DADH)/mirflickr25k-iall-unpaired-'+str(percentage)+'.mat'

    data_file = scio.loadmat(path)
    data_file_image = scio.loadmat(unpaired_file_path)
    images = data_file_image['images'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels


def load_flickr25k_unpaired_combo(path):
    percentage = 20
    unpaired_file_path_image = 'Unpaired Data/MIR-Flickr25K UT (DADH)/mirflickr25k-iall-unpaired-'+str(percentage)+'.mat'
    unpaired_file_path_text = 'Unpaired Data/MIR-Flickr25K UI/mirflickr25k-yall-unpaired-' + str(percentage) + '.mat'

    data_file = scio.loadmat(path)
    data_file_image = scio.loadmat(unpaired_file_path_image)
    images = data_file_image['images'][:]
    images = (images - images.mean()) / images.std()
    data_file_text = scio.loadmat(unpaired_file_path_text)
    tags = data_file_text['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels


def load_nus_wide(path_dir):
    data_file = scio.loadmat(path_dir)
    images = data_file['image'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['text'][:]
    labels = data_file['label'][:]

    return images, tags, labels


def load_nus_wide_unpaired_image(path_dir):
    percentage = 20
    unpaired_file_path = 'Unpaired Data/NUS-WIDE UI/nus-wide-tc21-yall-unpaired-'+str(percentage)+'.mat'

    data_file = scio.loadmat(path_dir)
    images = data_file['image'][:]
    images = (images - images.mean()) / images.std()
    data_file_text = scio.loadmat(unpaired_file_path)
    tags = data_file_text['YAll'][:]
    labels = data_file['label'][:]
    return images, tags, labels


def load_nus_wide_unpaired_text(path_dir):
    percentage = 20
    unpaired_file_path = 'Unpaired Data/NUS-WIDE UT (DADH)/nus-wide-tc21-iall-unpaired-'+str(percentage)+'.mat'

    data_file = scio.loadmat(path_dir)
    data_file_images = mat73.loadmat(unpaired_file_path)
    images = data_file_images['IAll'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['text'][:]
    labels = data_file['label'][:]
    return images, tags, labels


def load_nus_wide_unpaired_combo(path_dir):
    percentage = 20
    unpaired_file_path_image = 'Unpaired Data/NUS-WIDE UT (DADH)/nus-wide-tc21-iall-unpaired-'+str(percentage)+'.mat'
    unpaired_file_path_text = 'Unpaired Data/NUS-WIDE UI/nus-wide-tc21-yall-unpaired-'+str(percentage)+'-back.mat'

    data_file = scio.loadmat(path_dir)
    data_file_images = mat73.loadmat(unpaired_file_path_image)
    images = data_file_images['IAll'][:]
    images = (images - images.mean()) / images.std()
    data_file_tags = scio.loadmat(unpaired_file_path_text)
    tags = data_file_tags['YAll'][:]
    labels = data_file['label'][:]
    return images, tags, labels


def load_pretrain_model(path):
    return scio.loadmat(path)
