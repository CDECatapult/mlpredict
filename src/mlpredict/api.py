import json
from sklearn.externals import joblib
import pkg_resources
import os

from mlpredict.prediction import predict_walltime


def new_dnn(input_dimension,input_size):
    """Creates new dnn architecture
    Args:
        input_dimension:
        input_size:
    Returns:
        net: instance of class dnn
    """
    net = dnn()
    net['layers'] = {}
    net['input'] = {}
    net['input']['dimension'] = input_dimension
    net['input']['size'] = input_size
    return net


def import_dnn(dnn_obj):
    """Import dnn. Tries local definition first
    Returns:
        net: instance of class dnn"""
    try:
        if os.path.isfile(dnn_obj):
            net = import_dnn_file(dnn_obj)
        else:
            net = import_dnn_default(dnn_obj)
    except:
        net = {}
        print('Deep neural network representation could not be found')
    return net


def import_dnn_default(dnn_name):
    """Import dnn from default path
    Returns:
        net: instance of class dnn"""
    dnn_path = pkg_resources.resource_filename(
            'mlpredict', 'dnn_architecture/%s.json'
            %dnn_name)
    net = import_dnn_file(dnn_path)
    return net


def import_dnn_file(dnn_path):
    """Import dnn from local path
    Returns:
        net: instance of class dnn"""
    net = dnn()
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
    except:
        gpu_stats = {}
        print('GPU definition could not be found')
    return gpu_stats


def import_gpu_default(gpu_name):
    """Import gpu definition from default path
    Returns:
        gpu_stats"""
    gpu_file = pkg_resources.resource_filename(
            'mlpredict', 'GPUs/%s.json' %gpu_name)
    gpu_stats = import_gpu_file(gpu_file)
    return gpu_stats


def import_gpu_file(gpu_path):
    """Import gpu definition from local path
    Returns:
        gpu_stats"""
    with open(gpu_path) as json_data:
        gpu_stats = json.load(json_data)
    return gpu_stats


class dnn(dict):
    """Class for deep neural network architecture"""


    def save(self,path):
        """Save dnn to path"""
        if not os.path.isdir(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        with open(path, 'w') as json_file:
            json.dump(self, json_file, indent=4)


    def describe(self):
        """Prints a description of of the class instance"""
        print('%d layer network\n' %(len(self['layers'])))
        print('Input size %dx%dx%d\n'
              %(self['input']['size'],self['input']['size'],
                self['input']['dimension']))
        for layer in self['layers']:
            print('%s (%s), now %dx%d with %d channels'
                  %(self['layers'][layer]['name'],
                    self['layers'][layer]['type'],
                    self['layers'][layer]['output_size'],
                    self['layers'][layer]['output_size'],
                    self['layers'][layer]['channels_out']))


    def add_layer(self,layer_type,layer_name,**kwargs):
        """Adds a layer to the class instance
        Args:
            layer_type: Type of layer ('Convolution', 'Fully_connected' or
                    'Max_pool')
            layer_name: Name of layer (string)
        Layer type specific args:
            Convolution:
                kernelsize
                channels_out
                padding
                strides
                use_bias
                activation
            Max_pool:
                pool_size
                strides
                padding
            Fully_connected:
        """

        num_layers = len(self['layers'])
        new_layer = num_layers+1
        if num_layers==0:
            input_dimension = self['input']['dimension']
            input_size = self['input']['size']
        else:
            input_dimension = self['layers'][num_layers]['channels_out']
            input_size = self['layers'][num_layers]['output_size']


        self['layers'][new_layer] = {}     # Create new layer
        self['layers'][new_layer]['name'] = layer_name
        self['layers'][new_layer]['type'] = layer_type

        if layer_type.lower()=='convolution':
            padding_reduction = (
                    (kwargs['padding'].lower()=='valid')
                    *(kwargs['kernelsize']-1))
            output_size = ((input_size - padding_reduction)/kwargs['strides'])

            self['layers'][new_layer]['matsize'] = input_size
            self['layers'][new_layer]['kernelsize'] = kwargs['kernelsize']
            self['layers'][new_layer]['channels_in'] = input_dimension
            self['layers'][new_layer]['channels_out'] = kwargs['channels_out']
            self['layers'][new_layer]['padding'] = kwargs['padding']
            self['layers'][new_layer]['strides'] = kwargs['strides']
            self['layers'][new_layer]['use_bias'] = kwargs['use_bias']
            self['layers'][new_layer]['activation'] = kwargs['activation']
            self['layers'][new_layer]['output_size'] = output_size

        if layer_type.lower()=='max_pool':
            padding_reduction = (
                    (kwargs['padding'].lower()=='valid')
                    *(kwargs['pool_size']-1))
            output_size = ((input_size - padding_reduction)/kwargs['strides'])

            self['layers'][new_layer]['pool_size'] = kwargs['pool_size']
            self['layers'][new_layer]['strides'] = kwargs['strides']
            self['layers'][new_layer]['padding'] = kwargs['padding']
            self['layers'][new_layer]['output_size'] = output_size
            self['layers'][new_layer]['channels_out'] = input_dimension

        print('%s (%s), now %dx%d with %d channels'
              %(layer_name, layer_type, output_size, output_size,
                self['layers'][new_layer]['channels_out']))


    def remove_last_layer(self):
        """Removes last layer of class instance"""
        num_layers = len(self['layers'])
        if num_layers>0:
            del self['layers'][num_layers]


    def predict(self,
                gpu,
                optimizer='SGD',
                batchsize=1,
                model_file='',
                scaler_file=''):
        """Predicts execution time of class instance
        Args:
            gpu: can be local json file with GPU definition or
            batchsize: default 1
            saved_model: tensorflow model, by default uses model from
                    all GPUs
            scaler_file: sklearn scaler used to normalise inputs to
                    tensorflow model
        Returns:
            total execution time
            layer names
            layer execution times
        """

        if model_file=='':
            model_file = pkg_resources.resource_filename(
                    'mlpredict', 'model/model_all')
        if scaler_file=='':
            scaler_file = pkg_resources.resource_filename(
                    'mlpredict', 'model/scaler_Conv_all.save')

        gpu_stats = import_gpu(gpu)

        scaler = joblib.load(scaler_file)

        layer,time = predict_walltime(
                self, model_file, scaler, batchsize, optimizer,
                gpu_stats['bandwidth'], gpu_stats['cores'], gpu_stats['clock'])

        return sum(time), layer, time
