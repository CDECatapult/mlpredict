
import os
import json
import pkg_resources

import mlpredict


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
    """Import dnn definition from local file or mlpredict.
    Tries local definition first.
    Returns:
        net: instance of class Dnn
    """
    if os.path.isfile(dnn_obj):
        net = import_dnn_file(dnn_obj)
    else:
        net = import_dnn_default(dnn_obj)
    return net


def import_dnn_default(dnn_name):
    """Import dnn from default path (mlpredict).
    Returns:
        net: instance of class Dnn"""
    try:
        dnn_file = pkg_resources.resource_filename(
            'mlpredict', 'dnn_architecture/%s.json'
            % dnn_name)
        print(dnn_file)
    except:
        raise DnnImportError('No local network definition found. Attempted to '
            'import %s.json from mlpredict. File not found.' % dnn_name)
    net = import_dnn_file(dnn_file)
    return net


def import_dnn_file(dnn_file):
    """Import dnn from local path.
    Returns:
        net: instance of class Dnn
    """
    net = mlpredict.api.dnn(0, 0)
    with open(dnn_file) as json_data:
        tmpdict = json.load(json_data)
    try:
        net['layers'] = tmpdict['layers']
        net['input'] = tmpdict['input']
    except:
        raise DnnImportError('Invalid format of %s' % dnn_file)
    return net


def import_gpu(gpu_obj):
    """Import gpu definition from local file or mlpredict.
    Tries local definition first.
    Returns:
        gpu_stats
    """
    if os.path.isfile(gpu_obj):
        gpu_stats = import_gpu_file(gpu_obj)
    else:
        gpu_stats = import_gpu_default(gpu_obj)
    return gpu_stats


def import_gpu_default(gpu_name):
    """Import gpu definition from default path (mlpredict).
    Returns:
        gpu_stats
    """
    try:
        gpu_file = pkg_resources.resource_filename(
            'mlpredict', 'GPUs/%s.json' % gpu_name)
    except:
        raise GpuImportError('No local GPU definition found. Attempted to '
            'import %s.json from mlpredict. File not found.' % gpu_name)
    gpu_stats = import_gpu_file(gpu_file)
    return gpu_stats


<<<<<<< HEAD
def import_gpu_file(gpu_file):
    """Import gpu definition from local path
    Returns:
        gpu_stats
    """
    with open(gpu_file) as json_data:
        gpu_stats = json.load(json_data)
    if not all(key in gpu_stats.keys() for key in ['bandwidth','cores', 'clock']):
        raise GpuImportError(
            'Invalid GPU definition. Keys "bandwidth", "cores", '
            'and "clock" are required')
    return gpu_stats
