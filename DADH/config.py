import warnings
import torch


class Default(object):
    load_model_path = None  # load model path

    pretrain_model_path = './data/imagenet-vgg-f.mat'

    # visualization
    vis_env = 'main'  # visdom env
    vis_port = 8097  # visdom port
    flag = 'mir'
    
    batch_size = 128
    image_dim = 4096
    hidden_dim = 8192
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 1
    max_epoch = 300

    bit = 64  # hash code length
    lr = 0.00005  # initial learning rate

    device = 'cuda:0'

    # hyper-parameters
    alpha = 10
    gamma = 1
    beta = 1
    mu = 0.00001
    lamb = 1

    margin = 0.4
    dropout = False

    def data(self, flag):
        if flag[0] == 'm':

            if flag == 'mir_ui':
                self.dataset = 'mir-ui'
            elif flag == 'mir_ut':
                self.dataset = 'mir-ut'
            elif flag == 'mir_uit':
                self.dataset = 'mir-uit'
            else:
                self.dataset = 'mir'

            #self.dataset = 'mir'
            self.data_path = './data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 10000

        if flag[0] == 'n':

            if flag == 'nus_ui':
                self.dataset = 'nus-ui'
            elif flag == 'nus_ut':
                self.dataset = 'nus-ut'
            elif flag == 'nus_uit':
                self.dataset = 'nus-uit'
            else:
                self.dataset = 'nus'

            self.data_path = './data/NUS-WIDE-TC21.mat'
            self.db_size = 193734
            self.num_label = 21
            self.query_size = 2100
            self.text_dim = 1000
            self.training_size = 10000
        '''
        if flag == 'mir_deleted_text':
            self.dataset = 'mir-deleted-text'
            self.data_path = './data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 10000
        if flag == 'mir_deleted_image':
            self.dataset = 'mir-deleted-image'
            self.data_path = './data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 10000
        if flag == 'mir_deleted_combo':
            self.dataset = 'mir-deleted-combo'
            self.data_path = './data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 10000
        if flag == 'nus_deleted_text':
            self.dataset = 'nus-deleted-text'
            self.data_path = './data/NUS-WIDE-TC21.mat'
            self.db_size = 193734
            self.num_label = 21
            self.query_size = 2100
            self.text_dim = 1000
            self.training_size = 10000
        if flag == 'nus_deleted_image':
            self.dataset = 'nus-deleted-image'
            self.data_path = './data/NUS-WIDE-TC21.mat'
            self.db_size = 193734
            self.num_label = 21
            self.query_size = 2100
            self.text_dim = 1000
            self.training_size = 10000
        if flag == 'nus_deleted_combo':
            self.dataset = 'nus-deleted-combo'
            self.data_path = './data/NUS-WIDE-TC21.mat'
            self.db_size = 193734
            self.num_label = 21
            self.query_size = 2100
            self.text_dim = 1000
            self.training_size = 10000
    '''

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)
            
        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                    print('\t{0}: {1}'.format(k, getattr(self, k)))




opt = Default()
