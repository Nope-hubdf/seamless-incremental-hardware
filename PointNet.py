import numpy as np

import tensorflow as tf

from keras import ops
import keras
from keras import layers


######################
# orthogonal regularizer
######################

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
    
############
# T-net
###########

def conv1d_bn(x, filters):
  x = layers.Conv1D(filters, kernel_size=1)(x)
  x = layers.BatchNormalization()(x)
  return layers.Activation('relu')(x)

def dense_bn(x, filters):
  x = layers.Dense(units=filters, )(x)
  x = layers.BatchNormalization()(x)
  return layers.Activation('relu')(x)

def tnet(inputs, num_features):
    # Initialise bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv1d_bn(inputs, 32)
    x = conv1d_bn(x, 64)
    x = conv1d_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(num_features * num_features,
                     kernel_initializer="zeros",
                     bias_initializer=bias,
                     activity_regularizer=reg,)(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])
  
  
#################
# Putting all together: PointNet
#################

def pointnet(inputs, num_classes):
  x = tnet(inputs, 3) # top level we have only 3 features
  x = conv1d_bn(x, 32)
  x = conv1d_bn(x, 32)
  x = tnet(x, 32) # because prev conv layer has 32 feats
  x = conv1d_bn(x, 32)
  x = conv1d_bn(x, 64)
  x = conv1d_bn(x, 512)
  x = layers.GlobalAveragePooling1D()(x)
  x = dense_bn(x, 256)
  x = layers.Dropout(0.3)(x) # extra from the
  x = dense_bn(x, 128)
  x = layers.Dropout(0.3)(x)
  output = layers.Dense(num_classes, activation='softmax')(x) 
  # finally we predict the softmax probs for all class labels in the data
  return output


input_shape = (None, 3) # None refers to num points, 3 are features (x, y, z)
input = keras.Input(shape=input_shape, name='pc_input')

pointnet_output = pointnet(input, 10)

pointnet_model = keras.Model(input, pointnet_output)


pointnet_model.summary()