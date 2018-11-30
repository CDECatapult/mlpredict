# mlpredict

A python package to predict the execution time for one forward and backward pass a deep learning model.


To install mlpredict run
``` bash
pip install -r requirements.txt
```
and
```  bash
python setup.py install
```
from the root directory.

The mlpredict API can be used to create representations of deep neural networks and predict their execution time on different hardware.

### Create a model representations
To build the representation of a deep neural network from scratch create an instance of the dnn class
``` python
dnn_repr = mlpredict.api.new_dnn(input_dimension,input_size)
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

A completed model can be saved to a .json file using
``` python
dnn_repr.save(filename)
```

For a full working example see the jupyter notebook  https://github.com/CDECatapult/mlpredict/blob/master/notebooks/Create_new_dnn.ipynb.


### Import existing model representations
Exiting model representations can be imported using
```python
dnn_repr = mlpredict.api.import_default(filename)
```
An imported representation can be modified and saved as described above in **Create a model representations**.


### Predict execution time using mlpredict
A model representation *dnn_repr* created or imported through the mlpredict API can be used to predict the execution time of the corresponding tensorflow model on arbitrary GPUs.

```python
time_total, layer, time_layer = dnn_repr.predict(gpu,
                                                 optimizer,
                                                 batchsize)
```
returns the total execution time, the layers and the time per layer. For a complete working example see https://github.com/CDECatapult/mlpredict/blob/master/notebooks/Full_model_prediction.ipynb
