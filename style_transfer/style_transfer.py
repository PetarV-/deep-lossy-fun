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

# Fetch the base and reference images, store them in tensors
inp_base = K.variable(load_and_process('trin.jpg', target_size=(img_h, img_w)))
inp_ref = K.variable(load_and_process('wave.jpg', target_size=(img_h, img_w)))
# Construct a variable for the final result
inp_comb = K.placeholder((1, img_h, img_w, img_d))

# All of these are concatenated to form a batch of inputs to the ResNet
inp = K.concatenate([inp_base, inp_ref, inp_comb], axis=0)

# Fetch the pretrained VGG-16 (this time without tail)
model = VGG16(input_tensor=inp, weights='imagenet', include_top=False)

# Extract the layers of the model
lyr_dict = dict([(lyr.name, lyr.output) for lyr in model.layers])

# Helper function to extract the Gram matrix of a tensor
def gram(x):
    # Flatten each channel
    flat = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # Compute outer products of channel features with themselves
    gram = K.dot(flat, K.transpose(flat))
    return gram

# The "style loss": 
# how much do the Gram matrices of the reference and generated activations differ?
# (using the mean square difference)
def style_loss(gen, ref):
    g1 = gram(gen)
    g2 = gram(ref)
    (h, w, d) = K.int_shape(gen)
    size = h * w
    return K.sum(K.square(g1 - g2)) / (4.0 * (d ** 2) * (size ** 2))

# The "content loss":
# How much do the base and generated activations differ in content? (use the squared difference)
def content_loss(gen, base):
    return K.sum(K.square(gen - base))

# The "continuity loss":
# Make sure the generated image has continuity (squared difference of neighbouring pixels)
def continuity_loss(gen):
    row_diff = K.square(gen[:, :img_h - 1, :img_w - 1, :] - gen[:, 1:, :img_w - 1, :])
    col_diff = K.square(gen[:, :img_h - 1, :img_w - 1, :] - gen[:, :img_h - 1, 1:, :])
    return K.sum(row_diff + col_diff)

# Define the overall loss as the weighted combination of the three
content_wt = 1e6
style_wt = 1e9
continuity_wt = 1e-7

loss = K.variable(0.)
# Make the contents of a very deep layer (corresponding to complex features) match
content_fts = lyr_dict['block4_conv2']
base_fts = content_fts[0, :, :, :]
gen_fts = content_fts[2, :, :, :]
loss += content_wt * content_loss(gen_fts, base_fts)
# Make the styles at many scales match
for lyr in lyr_dict:
    style_fts = lyr_dict[lyr]
    ref_fts = style_fts[1, :, :, :]
    gen_fts = style_fts[2, :, :, :]
    loss += style_wt * style_loss(gen_fts, ref_fts)
# Finally, enforce continuity
loss += continuity_wt * continuity_loss(inp_comb)

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
iters = 10
x = load_and_process('trin.jpg', target_size=(img_h, img_w))

for i in range(iters):
    x, new_loss, _ = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Iteration', i, '- loss:', new_loss)
    x = x.reshape((1, img_h, img_w, img_d))
    deprocess_and_save(np.copy(x), 'nst_trin_{}.jpg'.format(i))
    
