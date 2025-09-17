import keras
from keras import ops

import tensorflow as tf

### to think about orthogonal regualrizer in PointNet

class OrthogonalRegularizer(keras.regularizers.Regularizer):
  '''
  original from keras examples (https://keras.io/examples/vision/pointnet/)
  '''
  def __init__(self, num_features, l2reg=0.001):
      self.num_features = num_features
      self.l2reg = l2reg # regularization factor
      self.eye = ops.eye(num_features) # identity matrix

  def __call__(self, t_m):
      t_m = ops.reshape(t_m, (-1, self.num_features, self.num_features))
      # t_m is the predicted transformation matrix
      # reshape from (batch, K*K) ---> (batch, K, K)
      t_m_t = tf.matmul(t_m, t_m, transpose_b=True)
      # matrix multiplication of t_m and t_m transpose (dot product).
      # finally return the desired term
      return ops.sum(self.l2reg * ops.square(t_m_t - self.eye))