from setuptools import setup, find_packages
setup(
    name="mlpredict",
    version="0.0.1",
    packages=['mlpredict'],
    package_dir={'mlpredict': 'src/mlpredict'},
    package_data={'mlpredict': [
            'GPUs/*.json',
            'dnn_architecture/*.json',
            'model/scaler_Conv_all.save',
            'model/model_all/saved_model.pb',
            'model/model_all/variables/*']},
)
