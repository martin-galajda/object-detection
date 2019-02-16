from keras import backend as K

# Inspired from https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/
def make_masked_binary_cross_entropy(mask_value):
  def masked_binary_cross_entropy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, K.cast(mask_value, K.floatx())), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)

  return masked_binary_cross_entropy


