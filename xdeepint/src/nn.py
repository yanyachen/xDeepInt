import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_util


def binary_classification_estimator_spec(
    mode,
    labels,
    logits,
    optimizers
):
    # Prediction
    logits = tf.reshape(logits, (-1,))
    probabilities = tf.nn.sigmoid(logits, name='Sigmoid')

    # Mode: PREDICT
    serving_signature = (
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    )
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            serving_signature: tf.estimator.export.PredictOutput(probabilities)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=probabilities,
            export_outputs=export_outputs
        )

    # Loss
    labels = tf.cast(labels, dtype=probabilities.dtype)
    loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    )
    loss = tf.losses.compute_weighted_loss(
        losses=loss_vec,
        weights=1.0,
        reduction=tf.losses.Reduction.SUM
    )

    # Mode: EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        auc = tf.metrics.auc(
            labels=labels,
            predictions=probabilities,
            num_thresholds=10000,
            curve='ROC',
            name='auc',
            summation_method='careful_interpolation'
        )
        average_loss = tf.metrics.mean(
            loss_vec,
            weights=array_ops.ones_like(loss_vec),
            name='average_loss'
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={
                'auc': auc,
                'average_loss': average_loss
            }
        )

    # Mode: TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:
        # train_op
        train_ops = []
        for scope, optimizer in optimizers.items():
            train_ops.append(
                optimizer.minimize(
                    loss=loss,
                    var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=scope
                    )
                )
            )
        train_op = control_flow_ops.group(*train_ops)
        # update_op
        update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
        if update_ops:
            train_op = control_flow_ops.group(train_op, *update_ops)
        # global_step
        with ops.control_dependencies([train_op]):
            train_op = state_ops.assign_add(
                training_util.get_global_step(), 1
            ).op
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )


def categorical_to_embedding(
    categorical_columns,
    embedding_size,
    combiner='mean',
    initializer=None
):
    embedding_columns = [
        tf.feature_column.embedding_column(
            categorical_column=categorical_column,
            dimension=embedding_size,
            combiner=combiner,
            initializer=initializer
        )
        for categorical_column in categorical_columns
    ]
    return embedding_columns


class VectorDense_Layer(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        kernel_regularizer=None,
        dropout=None
    ):
        super(VectorDense_Layer, self).__init__()
        self.units = units
        self.dropout = dropout

        self.permute_layer = tf.keras.layers.Permute(
            dims=(2, 1)
        )

        if self.dropout is not None and self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(
                rate=float(self.dropout)
            )

        self.dense_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

    def call(self, inputs, training):
        net = self.permute_layer(inputs)
        if self.dropout is not None and self.dropout > 0:
            net = self.dropout_layer(net, training=training)
        net = self.dense_layer(net)
        outputs = self.permute_layer(net)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = copy.copy(input_shape)
        output_shape[1] = self.units
        return tf.TensorShape(output_shape)


class Polynomial_Block(tf.keras.Model):

    def __init__(
        self,
        num_interaction_layer,
        num_sub_spaces,
        activation,
        dropout,
        residual,
        initializer,
        regularizer
    ):
        super(Polynomial_Block, self).__init__()
        self.num_interaction_layer = num_interaction_layer
        self.num_sub_spaces = num_sub_spaces
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        field_size = input_shape[1].value
        self.vector_dense_layers = [
            VectorDense_Layer(
                units=int(field_size * self.num_sub_spaces),
                activation=self.activation,
                use_bias=False,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                dropout=self.dropout[i] if self.dropout is not None else None
            )
            for i in range(self.num_interaction_layer)
        ]

    def call(
        self,
        inputs,
        field_size,
        embedding_size,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )

        # Split
        inputs = tf.concat(
            tf.split(inputs, self.num_sub_spaces, axis=2),
            axis=1
        )

        # Interaction
        interaction = inputs
        if not self.residual:
            interaction_list = []
            interaction_list.append(interaction)

        for layer_id in range(0, self.num_interaction_layer, +1):

            weighted_inputs = self.vector_dense_layers[layer_id](
                inputs,
                training=training
            )

            if self.residual:
                interaction = tf.keras.layers.multiply(
                    [interaction, (1.0 + weighted_inputs)]
                )
            else:
                interaction = tf.keras.layers.multiply(
                    [interaction, weighted_inputs]
                )
                interaction_list.append(interaction)

        # Output
        if self.residual:
            interaction_outputs = interaction
        else:
            interaction_outputs = tf.keras.backend.concatenate(
                interaction_list, axis=1
            )

        # Combine
        interaction_outputs = tf.concat(
            tf.split(interaction_outputs, self.num_sub_spaces, axis=1),
            axis=2
        )

        return interaction_outputs


class MultiHead_Polynomial_Block(tf.keras.Model):

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_interaction_layer,
        num_sub_spaces,
        activation,
        dropout,
        residual,
        initializer,
        regularizer
    ):
        super(MultiHead_Polynomial_Block, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_interaction_layer = num_interaction_layer
        self.num_sub_spaces = num_sub_spaces
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        self.projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=tf.keras.activations.linear,
            use_bias=False,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer
        )
        self.polynomial_block_list = [
            Polynomial_Block(
                num_interaction_layer=self.num_interaction_layer[i],
                num_sub_spaces=self.num_sub_spaces[i],
                activation=self.activation[i],
                dropout=self.dropout,
                residual=self.residual,
                initializer=self.initializer,
                regularizer=self.regularizer
            )
            for i in range(self.num_heads)
        ]

    def call(
        self,
        inputs,
        field_size,
        embedding_size,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )

        # Linear Projection
        inputs = self.projection_layer(inputs)

        # Split
        inputs_heads = tf.split(inputs, self.num_heads, axis=2)

        # Polynomial Interaction
        outputs_heads = [
            self.polynomial_block_list[i](
                inputs_heads[i],
                field_size=field_size,
                embedding_size=int(self.hidden_size / self.num_heads),
                training=training
            )
            for i in range(self.num_heads)
        ]

        # Combine
        outputs = tf.concat(outputs_heads, axis=1)

        # Return
        return outputs


def xDeepInt(features, labels, mode, params, config):
    '''
    feature_columns
    num_interaction_layer, num_sub_spaces
    activation_fn
    dropout, residual
    embedding_size
    initializer, regularizer
    embedding_optimizer, pin_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Input
    with variable_scope.variable_scope('Embedding'):
        num_feature = len(params['feature_columns'])
        embedding_columns = categorical_to_embedding(
            categorical_columns=params['feature_columns'],
            embedding_size=params['embedding_size']
        )
        input_units = tf.feature_column.input_layer(
            features=features,
            feature_columns=embedding_columns
        )
        feature_input_units = tf.keras.backend.reshape(
            input_units,
            shape=(-1, num_feature, params['embedding_size'])
        )

    with variable_scope.variable_scope('PIN'):
        # Interaction
        feature_interaction_block = Polynomial_Block(
            num_interaction_layer=params['num_interaction_layer'],
            num_sub_spaces=params['num_sub_spaces'],
            activation=params['activation_fn'],
            dropout=params['dropout'],
            residual=params['residual'],
            initializer=params['initializer'],
            regularizer=params['regularizer']
        )
        feature_interaction_units = feature_interaction_block(
            inputs=feature_input_units,
            field_size=num_feature,
            embedding_size=params['embedding_size'],
            training=is_training
        )

        # Output
        vector_linear_block = VectorDense_Layer(
            units=1,
            activation=tf.keras.activations.linear,
            use_bias=True,
            kernel_initializer=params['initializer'],
            kernel_regularizer=params['regularizer'],
            dropout=None
        )
        vector_logits = vector_linear_block(
            inputs=feature_interaction_units,
            training=is_training,
        )

        logits = tf.reduce_sum(
            tf.keras.backend.squeeze(vector_logits, axis=1),
            axis=1,
            keepdims=True
        )

    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Embedding': params['embedding_optimizer'],
            'PIN': params['pin_optimizer']
        }
    )
