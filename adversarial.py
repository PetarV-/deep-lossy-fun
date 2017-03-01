from keras.applications.resnet50 import ResNet50
from keras.layers import Input

model = ResNet50(weights='imagenet', include_top=True)

img_h = 224
img_w = 224

inp = model.inputs[0]
out = model.outputs[0]



