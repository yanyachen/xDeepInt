import tensorflow as tf


class GFTRL(tf.train.Optimizer):
    def __init__(
        self,
        alpha,
        beta,
        lambda1,
        lambda2,
        use_locking=False,
        name='GFTRL'
    ):
        super(GFTRL, self).__init__(use_locking, name)
        self._alpha = alpha
        self._beta = beta
        self._lambda1 = lambda1
        self._lambda2 = lambda2

    def _create_slots(self, var_list):
        for var in var_list:
            with tf.colocate_with(var):
                self._zeros_slot(var, 'n', self._name)
                self._zeros_slot(var, 'z', self._name)

    def _prepare(self):
        self._alpha_tensor = tf.convert_to_tensor(self._alpha, name='alpha')
        self._beta_tensor = tf.convert_to_tensor(self._beta, name='beta')
        self._lambda1_tensor = tf.convert_to_tensor(self._lambda1, name='lambda1')
        self._lambda2_tensor = tf.convert_to_tensor(self._lambda2, name='lambda2')

    def _apply_dense(self, grad, var):
        # Init
        var_dtype = var.dtype.base_dtype
        n = self.get_slot(var, 'n')
        z = self.get_slot(var, 'z')
        input_dim = tf.convert_to_tensor(var.shape[0].value, dtype=var_dtype)
        output_dim = tf.convert_to_tensor(var.shape[1].value, dtype=var_dtype)
        # Compute
        grad2 = tf.square(grad)
        sigma = (tf.sqrt(n + grad2) - tf.sqrt(n)) / self._alpha_tensor
        new_n = tf.assign(n, n + grad2)
        new_z = tf.assign(z, z + grad - sigma * var)
        z_norm = tf.reduce_sum(tf.square(new_z), axis=1, keepdims=True)
        z_norm = tf.sqrt(z_norm)
        z_norm = tf.tile(
            input=z_norm,
            multiples=(1, output_dim)
        )
        # Update
        new_var = tf.assign(
            var,
            tf.where(
                condition=tf.math.less_equal(z_norm, self._lambda1_tensor * tf.sqrt(output_dim)),
                x=tf.zeros(shape=(input_dim, output_dim)),
                y=new_z * tf.divide(
                    ((self._lambda1_tensor * tf.sqrt(output_dim)) / z_norm - 1),
                    ((self._beta_tensor + tf.sqrt(new_n)) / self._alpha_tensor + self._lambda2_tensor)
                )
            )
        )
        # Return
        updates = [new_var, new_n, new_z]
        return tf.group(*updates)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_sparse(self, grad, var):
        return self._apply_sparse(grad, var)
