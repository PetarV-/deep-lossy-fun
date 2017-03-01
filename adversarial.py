from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.layers import Input
from keras import backend as K
from processing import load_and_process, deprocess_and_save

# Fetch the pretrained ResNet-50
model = ResNet50(weights='imagenet', include_top=True)

# We're using the full predictor, so must give 224 x 224 x 3 images
img_h = 224
img_w = 224
img_d = 3

# Extract the input and output tensor
inp = model.inputs[0]
out = model.outputs[0]

# Define the loss as the negative confidence in the chosen class
loss = K.variable(0.)
loss -= out[:,409] # class 409: Sulphur-crested Cockatoo :)

# A function that will give us the gradients wrt the input
f = K.function([inp], [K.gradients(loss, inp)])

# FGSM: Take one step in the direction of eps * sgn(grad)
def fgsm(x, eps=0.03):
    grads = f([x]).reshape((1, img_h, img_w, img_d)).astype('float64')
    return x - eps * grads

img = load_and_process('elephant.jpg', target_size=(img_h, img_w))
preds = model.predict(img)
print('Original image predictions:', decode_predictions(preds, top=5)[0])

img = fgsm(img)

preds = model.predict(img)
print('Adversarial image predictions:', decode_predictions(preds, top=5)[0])

deprocess_and_save(img, 'adv_elephant.jpg')

