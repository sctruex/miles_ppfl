import tensorflow as tf



class PolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a polynomial decay schedule.

    It is commonly observed that a monotonically decreasing learning rate, whose
    degree of change is carefully chosen, results in a better performing model.
    This schedule applies a polynomial decay function to an optimizer step,
    given a provided `initial_learning_rate`, to reach an `end_learning_rate`
    in the given `decay_steps`.

    It requires a `step` value to compute the decayed learning rate. You
    can just pass a TensorFlow variable that you increment at each training
    step.

    The schedule is a 1-arg callable that produces a decayed learning rate
    when passed the current optimizer step. This can be useful for changing the
    learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      return ((initial_learning_rate - end_learning_rate) *
              (1 - step / decay_steps) ^ (power)
             ) + end_learning_rate
    ```

    If `cycle` is True then a multiple of `decay_steps` is used, the first one
    that is bigger than `step`.

    ```python
    def decayed_learning_rate(step):
      decay_steps = decay_steps * ceil(step / decay_steps)
      return ((initial_learning_rate - end_learning_rate) *
              (1 - step / decay_steps) ^ (power)
             ) + end_learning_rate
    ```

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate.
    Example: Fit a model while decaying from 0.1 to 0.01 in 10000 steps using
    sqrt (i.e. power=0.5):

    ```python
    ...
    starter_learning_rate = 0.1
    end_learning_rate = 0.01
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)

    model.compile(optimizer=tf.keras.optimizers.SGD(
                      learning_rate=learning_rate_fn),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data, labels, epochs=5)
    ```

    The learning rate schedule is also serializable and deserializable using
    `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        end_learning_rate=0.0001,
        power=1.0,
        cycle=False,
        initial_step=0, 
        name=None,
    ):
        """Applies a polynomial decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
          end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The minimal end learning rate.
          power: A scalar `float32` or `float64` `Tensor` or a
            Python number. The power of the polynomial. Defaults to `1.0`.
          cycle: A boolean, whether it should cycle beyond decay_steps.
          name: String.  Optional name of the operation. Defaults to
            'PolynomialDecay'.
        """
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.initial_step = initial_step
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "PolynomialDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            end_learning_rate = tf.cast(self.end_learning_rate, dtype)
            power = tf.cast(self.power, dtype)

            initial_step = tf.cast(self.initial_step, dtype)
            global_step_recomp = tf.cast(tf.add(step,initial_step), dtype)
            decay_steps_recomp = tf.cast(self.decay_steps, dtype)
            if self.cycle:
                # Find the first multiple of decay_steps that is bigger than
                # global_step. If global_step is zero set the multiplier to 1
                multiplier = tf.where(
                    tf.equal(global_step_recomp, 0),
                    1.0,
                    tf.math.ceil(global_step_recomp / self.decay_steps),
                )
                decay_steps_recomp = tf.multiply(decay_steps_recomp, multiplier)
            else:
                # Make sure that the global_step used is not bigger than
                # decay_steps.
                global_step_recomp = tf.minimum(
                    global_step_recomp, decay_steps_recomp
                )

            p = tf.divide(global_step_recomp, decay_steps_recomp)
            return tf.add(
                tf.multiply(
                    initial_learning_rate - end_learning_rate,
                    tf.pow(1 - p, power),
                ),
                end_learning_rate,
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "cycle": self.cycle,
            "name": self.name,
        }



class MinCapableStepLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_steps, end_learning_rate, gamma, initial_step=0, name="MinCapableStep", **_):
        """A `LearningRateSchedule` that uses a gamma decay schedule.

        When training a model, it is often useful to lower the learning rate as
        the training progresses. This schedule applies a gamma decay function
        to an optimizer step, given a provided initial learning rate, to never fall
        blow the `end_learning_rate`.

        The schedule is a 1-arg callable that produces a decayed learning
        rate when passed the current optimizer step. This can be useful for changing
        the learning rate value across different invocations of optimizer functions.
        It is computed as:

        ```python
        def decayed_learning_rate(step):
            return max(end_learning_rate, 
                       initial_learning_rate * gamma ^ ((initial_step + step) // decay_steps))
        ```

        You can pass this schedule directly into a `keras.optimizers.Optimizer`
        as the learning rate.
        Example: When fitting a Keras model, decay every 1000 steps with a gamma (base)
        of 0.96 with a minimum possible learning rate of 1e-5:

        ```python
        initial_learning_rate = 0.1
        lr_schedule = keras.optimizers.schedules.MinCapableStepLR(
            initial_learning_rate,
            decay_steps=1000,
            end_learning_rate=1e-5,
            gamma=0.96)

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(data, labels, epochs=5)
        ```

        The learning rate schedule is also serializable and deserializable using
        `keras.optimizers.schedules.serialize` and
        `keras.optimizers.schedules.deserialize`.

        Args:
            initial_learning_rate: A Python float. The initial learning rate.
            decay_steps: A Python integer. Must be positive. See the decay
                computation above.
            end_learning_rate: A Python float. The minimum possible learning rate.
            gamma: A Python float. The decay rate.
            name: String.  Optional name of the operation.  Defaults to
                `"MinCapableStep`".

        Returns:
            A 1-arg callable learning rate schedule that takes the current optimizer
            step and outputs the decayed learning rate, a scalar tensor of the
            same type as `initial_learning_rate`.
        """
        self._initial_learning_rate = initial_learning_rate
        self._step_size = decay_steps
        self._min_lr = end_learning_rate
        self._gamma = gamma
        self._initial_step = initial_step
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or "MinCapableStep") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self._initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            initial_steps = tf.cast(self._initial_step, dtype)
            decay_steps = tf.cast(self._step_size, dtype)
            decay_rate = tf.cast(self._gamma, dtype)
            global_step = tf.cast(step, dtype)
            min_lr = tf.cast(self._min_lr, dtype)

            p = tf.floor(
                tf.divide(tf.add(global_step, initial_steps), decay_steps))
            return tf.maximum(min_lr, tf.multiply(initial_learning_rate, tf.pow(decay_rate, p)))
