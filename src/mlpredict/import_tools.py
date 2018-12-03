
import os
import json
import pkg_resources

import mlpredict.api


def import_dnn(dnn_obj):
    """Import dnn. Tries local definition first
    Returns:
        net: instance of class dnn"""
    # try:
    if os.path.isfile(dnn_obj):
        net = import_dnn_file(dnn_obj)
    else:
        net = import_dnn_default(dnn_obj)
    # except:
    #     net = {}
    #     print('Deep neural network representation could not be found')
    return net


def import_dnn_default(dnn_name):
    """Import dnn from default path
    Returns:
        net: instance of class dnn"""
    dnn_path = pkg_resources.resource_filename(
        'mlpredict', 'dnn_architecture/%s.json'
        % dnn_name)
    net = import_dnn_file(dnn_path)
    return net


def import_dnn_file(dnn_path):
    """Import dnn from local path
    Returns:
        net: instance of class dnn"""
    net = mlpredict.api.dnn(0, 0)
    with open(dnn_path) as json_data:
        tmpdict = json.load(json_data)
    net['layers'] = tmpdict['layers']
    net['input'] = tmpdict['input']
    return net


def import_gpu(gpu_obj):
    """Import gpu definition. Tries local definition first
    Returns:
        gpu_stats"""
    try:
        if os.path.isfile(gpu_obj):
            gpu_stats = import_gpu_file(gpu_obj)
        else:
            gpu_stats = import_gpu_default(gpu_obj)
    except BaseException:
        gpu_stats = {}
        print('GPU definition could not be found')
    return gpu_stats


def import_gpu_default(gpu_name):
    """Import gpu definition from default path
    Returns:
        gpu_stats"""
    gpu_file = pkg_resources.resource_filename(
        'mlpredict', 'GPUs/%s.json' % gpu_name)
    gpu_stats = import_gpu_file(gpu_file)
    return gpu_stats


def import_gpu_file(gpu_path):
    """Import gpu definition from local path
    Returns:
        gpu_stats"""
    with open(gpu_path) as json_data:
        gpu_stats = json.load(json_data)
    return gpu_stats
