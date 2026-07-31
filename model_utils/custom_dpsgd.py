from typing import Optional

from model_utils.custom_gaussian import CustomGaussianSumQuery
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import make_keras_generic_optimizer_class
from tensorflow_privacy import GenericDPSGDOptimizer

def make_custom_gaussian_query_optimizer_class(cls):
  """Returns a differentially private optimizer using the `GaussianSumQuery`.

  Args:
    cls: `DPOptimizerClass`, the output of `make_keras_optimizer_class`.

  Returns:
    A DP-SGD subclass of `cls` using the `GaussianQuery`, the canonical DP-SGD
    implementation.
  """

  def return_gaussian_query_optimizer(
      l2_norm_clip: float,
      noise_multiplier: float,
      num_microbatches: Optional[int] = None,
      gradient_accumulation_steps: int = 1,
      *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
      **kwargs):
    """
    Args:
      l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
      noise_multiplier: Ratio of the standard deviation to the clipping norm.
      num_microbatches: Number of microbatches into which each minibatch is
        split. Default is `None` which means that number of microbatches is
        equal to batch size (i.e. each microbatch contains exactly one example).
        If `gradient_accumulation_steps` is greater than 1 and
        `num_microbatches` is not `None` then the effective number of
        microbatches is equal to `num_microbatches *
        gradient_accumulation_steps`.
      gradient_accumulation_steps: If greater than 1 then optimizer will be
        accumulating gradients for this number of optimizer steps before
        applying them to update model weights. If this argument is set to 1 then
        updates will be applied on each optimizer step.
      *args: These will be passed on to the base class `__init__` method.
      **kwargs: These will be passed on to the base class `__init__` method.
    """
    dp_sum_query = CustomGaussianSumQuery(
        l2_norm_clip, l2_norm_clip * noise_multiplier)
    return cls(
        dp_sum_query=dp_sum_query,
        num_microbatches=num_microbatches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        *args,
        **kwargs)

  return return_gaussian_query_optimizer



CustomDPSGDOptimizer = make_custom_gaussian_query_optimizer_class(GenericDPSGDOptimizer)
