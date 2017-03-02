from keras.applications.vgg16 import VGG16, decode_predictions
from keras.layers import Input
from keras import backend as K
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from processing import load_and_process, deprocess_and_save, preprocess_batch
from evaluator import Eval

# Specify size of images that we are considering
img_h = 600
img_w = 600
img_d = 3

# Fetch the pretrained VGG-16 (without tail)
model = VGG16(weights='imagenet', include_top=False)

# Extract the input tensor
inp = model.inputs[0]

# Extract the layers of the model
lyr_dict = dict([(lyr.name, lyr.output) for lyr in model.layers])

# The "activation loss":
# Make the activations as high as possible, so minimise negative squares
def activation_loss(gen):
    return -K.sum(K.square(gen))

# The "L2-loss":
# Don't want an overly bright image. This is basically the inverse of the activation loss.
def l2_loss(gen):
    return K.sum(K.square(gen))

# The "continuity loss":
# Make sure the generated image has continuity (squared difference of neighbouring pixels)
def continuity_loss(gen):
    row_diff = K.square(gen[:, :img_h - 1, :img_w - 1, :] - gen[:, 1:, :img_w - 1, :])
    col_diff = K.square(gen[:, :img_h - 1, :img_w - 1, :] - gen[:, :img_h - 1, 1:, :])
    return K.sum(row_diff + col_diff)

# Define the overall loss as the combination of the above
activation_wt = 1.0
cont_wt = 0.1
l2_wt = 0.1

loss = K.variable(0.)
# Make the activations at many scales "light up"
for lyr in lyr_dict:
    loss += activation_wt * activation_loss(lyr_dict[lyr])
# Make the L2 of the image small
loss += l2_wt * lw_loss(inp)
# Finally, enforce continuity
loss += continuity_wt * continuity_loss(inp)

# Build up a function that returns the loss and its gradients
outputs = [loss]
grads = K.gradients(loss, inp_comb)
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

# A function that will give us the gradients wrt the input
f = K.function([inp_comb, K.learning_phase()], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, img_h, img_w, img_d))
    outs = f([x, 0])
    loss_val = outs[0]
    grads_val = np.array(outs[1:]).flatten().astype('float64')
    return loss_val, grads_val

evaluator = Eval(eval_loss_and_grads)

# Start from the base image, and apply gradient-based optimisation for some no. of iterations
iters = 7
x = load_and_process('trin.jpg', target_size=(img_h, img_w))

for i in range(iters):
    x, new_loss, _ = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Iteration', i, '- loss:', new_loss)
    x = x.reshape((1, img_h, img_w, img_d))
    deprocess_and_save(np.copy(x), 'nst_trin_{}.jpg'.format(i))
    
