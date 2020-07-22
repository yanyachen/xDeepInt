import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from src.io import (
    build_csv_vocabulary, build_csv_dataset,
    count_lines
)
from src.transformation import (
    gbdt_feature_engineering_fn, log_square_binning_fn,
    tf_signedlog
)
from src.optimizer import GFTRL
from src.nn import xDeepInt


# OpenMP Threads
os.environ['OMP_NUM_THREADS'] = '{:d}'.format(1)


# Parameters Definition
label_name = ['Label']
integer_feature_name = ['I'+str(i) for i in range(1, 13)]
categorical_feature_name = ['C'+str(i) for i in range(1, 27)]
feature_name = integer_feature_name + categorical_feature_name

train_csv_filename = './data/criteo/csv/train.csv'
test_csv_filename = './data/criteo/csv/test.csv'
vocabulary_folder = './file/criteo/vocabulary/'
vocabulary_threshold = 20
vocabulary_running_flag = False

schema_dict = dict(
    [
        (each, np.array([0], dtype=np.int64))
        for each in integer_feature_name
    ] +
    [
        (each, np.array(['']))
        for each in categorical_feature_name
    ] +
    [
        (each, np.array([0], dtype=np.int64))
        for each in label_name
    ]
)


# Evaluation Function
df_label = pd.read_csv('./data/criteo/csv/test.csv', usecols=['Label'])


def criteo_eval(pred):
    val_logloss = logloss(
        pred,
        df_label['Label'].values
    )
    val_nce = normalized_cross_entropy(
        pred,
        df_label['Label'].values
    )
    val_auc = auc(
        pred,
        df_label['Label'].values
    )
    print(
        'LogLoss: {logloss}\nNCE: {nce}\nAUC: {auc}'.format(
            logloss=val_logloss,
            nce=val_nce,
            auc=val_auc
        )
    )
    return None


# Vocabulary
if vocabulary_running_flag:
    build_csv_vocabulary(
        filenames=[train_csv_filename],
        columns=categorical_feature_name,
        vocabulary_folder=vocabulary_folder,
        threshold=vocabulary_threshold
    )


# Input Function
def train_input_fn(batch_size, num_epochs):
    dataset = build_csv_dataset(
        filenames=[train_csv_filename],
        feature_name=feature_name,
        label_name=label_name,
        schema_dict=schema_dict,
        compression_type=None,
        buffer_size=128 * 1024 * 1024,
        field_delim=',',
        use_quote_delim=True,
        na_value='',
        shuffle=False,
        shuffle_buffer_size=1024 * 8,
        num_epochs=num_epochs,
        batch_size=batch_size,
        prefetch_buffer_size=8,
        num_parallel_calls=4,
        feature_engineering_fn=log_square_binning_fn,
        binning_cols=integer_feature_name
    )
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels['Label']


def test_input_fn(batch_size):
    dataset = build_csv_dataset(
        filenames=[test_csv_filename],
        feature_name=feature_name + label_name, # TEMP
        label_name=label_name,
        schema_dict=schema_dict,
        compression_type=None,
        buffer_size=128 * 1024 * 1024,
        field_delim=',',
        use_quote_delim=True,
        na_value='',
        shuffle=False,
        shuffle_buffer_size=1024 * 8,
        num_epochs=1,
        batch_size=batch_size,
        prefetch_buffer_size=8,
        num_parallel_calls=4,
        feature_engineering_fn=log_square_binning_fn,
        binning_cols=integer_feature_name
    )
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels['Label']


# Feature Column
integer_feature_categorical_columns = [
    tf.feature_column.categorical_column_with_identity(
        key=col + '_' + 'bin',
        num_buckets=64
    )
    for col in integer_feature_name
]

categorical_feature_categorical_columns = [
    tf.feature_column.categorical_column_with_vocabulary_file(
        key=col,
        vocabulary_file=vocabulary_folder+str(col)+'.txt',
        num_oov_buckets=0,
        default_value=None,
        dtype=tf.string
    )
    for col in categorical_feature_name
]

integer_feature_embedding_columns = [
    tf.feature_column.embedding_column(
        categorical_column=each,
        dimension=8,
        combiner='mean',
        initializer=None,
        max_norm=None,
        trainable=True
    )
    for each in integer_feature_categorical_columns
]

categorical_feature_embedding_columns = [
    tf.feature_column.embedding_column(
        categorical_column=each,
        dimension=np.ceil(np.log2(
            count_lines(vocabulary_folder+each.key+'.txt')
        )) * 1,
        combiner='mean',
        initializer=None,
        max_norm=None,
        trainable=True
    )
    for each in categorical_feature_categorical_columns
]


# xDeepInt
xDeepInt_Model = tf.estimator.Estimator(
    xDeepInt,
    model_dir='./file/criteo/nn/xdeepint/',
    params={
        'feature_columns': (
            integer_feature_categorical_columns +
            categorical_feature_categorical_columns
        ),
        'num_interaction_layer': 4,
        'num_sub_spaces': 16,
        'activation_fn': tf.keras.activations.linear,
        'dropout': None,
        'residual': True,
        'embedding_size': 16,
        'initializer': tf.glorot_uniform_initializer(),
        'regularizer': tf.keras.regularizers.l2(l=1.0),
        'embedding_optimizer': GFTRL(
            alpha=0.02, beta=1.0,
            lambda1=10.0, lambda2=10.0
        ),
        'pin_optimizer': tf.train.FtrlOptimizer(
            learning_rate=0.02, learning_rate_power=-0.5,
            l1_regularization_strength=0.1, l2_regularization_strength=0.1
        )
    },
    config=tf.estimator.RunConfig(
        tf_random_seed=0,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=10,
        log_step_count_steps=1000,
        session_config=tf.ConfigProto(device_count={'GPU': 1})
    )
)

xDeepInt_Model.train(
    input_fn=lambda: train_input_fn(batch_size=1024 * 4, num_epochs=1)
)

xdeepint_pred_iterator = xDeepInt_Model.predict(
    input_fn=lambda: test_input_fn(batch_size=1024 * 16)
)
xdeepint_pred = np.array([each for each in xdeepint_pred_iterator])
criteo_eval(xdeepint_pred)
