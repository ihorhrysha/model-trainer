import pickle as pkl
import yaml


def pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pkl.dump(obj, f)


def unpickle(filepath):
    with open(filepath, 'rb') as f:
        obj = pkl.load(f)
    return obj


def load_yaml(filepath):
    with open(filepath) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def read_sql_query(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    return data
