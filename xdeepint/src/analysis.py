import numpy as np


def neuron_sparse_ratio(x):
    return np.sum(x == 0.0) / float(np.prod(x.shape))


def feature_sparse_ratio(x):
    assert np.ndim(x) == 2
    return np.sum(np.linalg.norm(x, ord=2, axis=1) == 0.0) / float(x.shape[0])


def deepint_stat(estimator):
    # Init
    stat = {}
    embedding_stat = {}
    pin_stat = {}
    # Computing
    for each in estimator.get_variable_names():
        if 'FTRL' not in each.upper() and 'ADAM' not in each.upper():
            var = estimator.get_variable_value(each)
            if 'Embedding' in each:
                embedding_stat[each] = {
                    'shape': var.shape,
                    'sparse_ratio': neuron_sparse_ratio(var),
                    'feature_sparse_ratio': feature_sparse_ratio(var)
                }
            elif 'PIN' in each:
                pin_stat[each] = {
                    'shape': var.shape,
                    'sparse_ratio': neuron_sparse_ratio(var)
                }
    # Embedding Analysis
    num_neuron = 0
    num_zero_neuron = 0
    num_feature = 0
    num_zero_feature = 0
    for each in embedding_stat:
        num_neuron += np.prod(embedding_stat[each]['shape'])
        num_zero_neuron += np.prod(embedding_stat[each]['shape']) * embedding_stat[each]['sparse_ratio']
        num_feature += embedding_stat[each]['shape'][0]
        num_zero_feature += embedding_stat[each]['shape'][0] * embedding_stat[each]['feature_sparse_ratio']
    stat['Embedding_Weights'] = num_neuron
    stat['Embedding_Sparse_Ratio'] = num_zero_neuron / num_neuron
    stat['Embedding_Feature_Sparse_Ratio'] = num_zero_feature / num_feature
    # PIN Analysis
    num_neuron = 0
    num_zero_neuron = 0
    for each in pin_stat:
        num_neuron += np.prod(pin_stat[each]['shape'])
        num_zero_neuron += np.prod(pin_stat[each]['shape']) * pin_stat[each]['sparse_ratio']
    stat['PIN_Weights'] = num_neuron
    stat['PIN_Sparse_Ratio'] = num_zero_neuron / num_neuron
    # Total Analysis
    stat['Total_Weights'] = stat['Embedding_Weights'] + stat['PIN_Weights']
    stat['Total_Sparse_Ratio'] = (
        stat['Embedding_Weights'] * stat['Embedding_Sparse_Ratio'] +
        stat['PIN_Weights'] * stat['PIN_Sparse_Ratio']
    ) / stat['Total_Weights']

    # Return
    return (embedding_stat, pin_stat, stat)
