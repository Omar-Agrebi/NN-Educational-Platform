import requests
import time

csv_content = b"""sepal_l,sepal_w,petal_l,petal_w,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
"""
upload_resp = requests.post('http://localhost:8000/api/v1/custom/upload',
                            files={'file': ('iris.csv', csv_content, 'text/csv')},
                            data={'session_id': 'test_sess'})
print('Upload:', upload_resp.status_code, upload_resp.text[:50])

payload = {
    'session_id': 'test_sess',
    'target_col': 4,
    'feature_cols': [0,1,2,3],
    'problem_type': 'multiclass_classification',
    'model_type': 'mlp',
    'hidden_layers': 2,
    'neurons_per_layer': [64, 32],
    'activations_per_layer': ['relu', 'relu'],
    'regularization': 'l2',
    'reg_lambda': 0.001,
    'learning_rate': 0.01,
    'epochs': 100,
    'batch_size': 32,
    'test_size': 0.2,
    'random_seed': 42
}
t0 = time.time()
train_resp = requests.post('http://localhost:8000/api/v1/custom/train', json=payload)
print('Train:', train_resp.status_code, 'time:', time.time() - t0)
if train_resp.status_code != 200:
    print(train_resp.text)
