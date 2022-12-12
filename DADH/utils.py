import torch
import numpy as np
import visdom
from scipy import io


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_map_k_classes(qB, rB, query_label, retrieval_label, k=None):
    '''

    :param qB: Query Bits
    :param rB: Retrieval Bits
    :param query_label: Query Label
    :param retrieval_label: Retrieval Label
    :param k: mAP at k (None = 100% recall)
    :return: mAP for each class
    '''
    num_query = query_label.shape[0]
    label_check = query_label.cpu()
    label_check = label_check.tolist()
    map = 0.
    map_classes = []
    map_classes_count = []
    for i in label_check[0]:
        map_classes.append(0.)
        map_classes_count.append(0)
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]

        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0

        new_label_check = [int(o) for o in label_check[i]]
        for e in range(0, len(label_check[i])):
            if new_label_check[int(e)] == 1:
                map_classes[e] += torch.mean(count / tindex)
                map_classes_count[e] += 1
        map += torch.mean(count / tindex)
    for i in range(0, len(map_classes_count)):
        print(float(map_classes[i] / map_classes_count[i]))
        map_classes[i] = map_classes[i] / map_classes_count[i]
    map = map / num_query
    return map


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self.index = {}

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def __getattr__(self, name):
        return getattr(self.vis, name)
