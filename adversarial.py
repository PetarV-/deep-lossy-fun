from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.layers import Input
from keras import backend as K
from processing import preprocess_batch, deprocess_batch, load_and_process, deprocess_and_save
import numpy as np

# Fetch the pretrained ResNet-50
model = ResNet50(weights='imagenet', include_top=True)

# We're using the full predictor, so must give 224 x 224 x 3 images
img_h = 224
img_w = 224
img_d = 3

# Extract the input and output tensor
inp = model.inputs[0]
out = model.outputs[0]

cid = 89 # Class 89: Sulphur-crested Cockatoo :)

# Define the loss as the negative confidence in the chosen class
loss = K.variable(0.)
loss -= out[:, cid]

# A function that will give us the gradients wrt the input
f = K.function([inp, K.learning_phase()], K.gradients(loss, inp))

# FGSM: Take some steps in the direction of eps * sgn(grad)
def fgsm(x, eps=32, alp=1.0):
    num_iter = min(eps + 4, 1.25 * eps)
    conf = 0.0
    x = np.copy(x)
    while conf < 0.99 and num_iter > 0:
        grads = np.array(f([x, 0])).reshape((1, img_h, img_w, img_d)).astype('float64')
        adv_x = x - alp * np.sign(grads)
        sub_x = np.minimum(x + eps, np.maximum(x - eps, adv_x))
        next_x = preprocess_batch(deprocess_batch(np.copy(sub_x)))
        x = next_x
        conf = model.predict(x)[0, cid]
        print('Confidence:', conf)
    return x

img = load_and_process('elephant.jpg', target_size=(img_h, img_w))
preds = model.predict(img)
print('Original image predictions:', decode_predictions(preds, top=5)[0])

img = fgsm(np.copy(img))

preds = model.predict(img)
print('Adversarial image predictions:', decode_predictions(preds, top=5)[0])

deprocess_and_save(img, 'adv_elephant.jpg')

