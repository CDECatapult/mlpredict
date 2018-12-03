
import os
import json
import pkg_resources

import mlpredict.api


class DnnImportError(Exception):

    def __init__(self, message):
        """This exception will be raised if an invalid dnn representation is
        attempted to be imported
        """
        super().__init__(message)


class GpuImportError(Exception):

    def __init__(self, message):
        """This exception will be raised if an invalid GPU definition is
        attempted to be imported
        """
        super().__init__(message)


def import_dnn(dnn_obj):
    """Import dnn. Tries local definition first
    Returns:
        net: instance of class dnn"""
    if os.path.isfile(dnn_obj):
        net = import_dnn_file(dnn_obj)
    else:
        net = import_dnn_default(dnn_obj)
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
    try:
        net['layers'] = tmpdict['layers']
        net['input'] = tmpdict['input']
        return net
    except:
        raise DnnImportError('Invalid format of %s' % dnn_path)


def import_gpu(gpu_obj):
    """Import gpu definition. Tries local definition first
    Returns:
        gpu_stats"""
    if os.path.isfile(gpu_obj):
        gpu_stats = import_gpu_file(gpu_obj)
    else:
        gpu_stats = import_gpu_default(gpu_obj)
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
    if not all(key in gpu_stats.keys() for key in ['bandwidth','cores', 'clock']):
        raise GpuImportError(
            'Invalid GPU definition. Keys "bandwidth", "cores", '
            'and "clock" are required')
    return gpu_stats
