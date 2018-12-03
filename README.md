# mlpredict

A python package to predict the execution time for one forward and backward pass a deep neural network.

To improve the underlying machine learning model see https://github.com/CDECatapult/ml-performance-prediction.


mlpredict can be installed by executing
``` bash
pip install -r requirements.txt
python setup.py install
```
from the root directory.

The mlpredict API can be used to create representations of deep neural networks and predict their execution time on different hardware.

### Create a model representations
To build the representation of a deep neural network from scratch create an instance of the dnn class
``` python
dnn_repr = mlpredict.api.dnn(input_dimension,input_size)
```
and add layers
``` python
dnn_repr.add_layer(layer_type, layer_name, **kwargs)
```
The last layer can be removed
``` python
dnn_repr.remove_last_layer()
```
and the current network architecture can be displayed
``` python
dnn_repr.describe()
```

Finally, a model can be saved to a .json file using
``` python
dnn_repr.save(filename)
```

For a full working example see the jupyter notebook  https://github.com/CDECatapult/mlpredict/blob/master/notebooks/Create_new_dnn.ipynb.


### Import existing model representations
Exiting model representations can be imported using
```python
dnn_repr = mlpredict.api.import_default(dnn_object)
```
`dnn_object` can be either the path to a previously created .json file (see above) or the name of a default model (at the moment only`'VGG16'`). An imported representation can be modified and saved as described above in the section **Create a model representations**.


### Predict execution time using mlpredict
A model representation *dnn_repr* created or imported through the mlpredict API can be used to predict the execution time of the corresponding tensorflow model on arbitrary GPUs.

```python
time_total, layer, time_layer = dnn_repr.predict(gpu,
                                                 optimizer,
                                                 batchsize)
```
returns the total execution time, the layers and the time per layer. Here, `gpu` can be a .json file with the keys 'bandwidth', 'cores', and 'clock' or the name of a default GPU ('V100', 'P100', 'M60', 'K80', 'K40', or '1080Ti'). For a complete working example see https://github.com/CDECatapult/mlpredict/blob/master/notebooks/Full_model_prediction.ipynb
