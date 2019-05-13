'''
contception fully convolutional network.
default receptive field / patch_size = 32*32.
default inputs_size = 256*256

Note: if you edit patch_size, you probably need to manually edit the
model itself.  Simplest is to modify net3 and net8 since they have a
non-trivial kernel size already, ensuring that net3s kernel size +
net8s kernel size == patch_size should make it run correctly (at
least for default values on other params). 
'''

from keras.layers import Conv2D, BatchNormalization, Input, concatenate, ZeroPadding2D, add
from keras.layers.advanced_activations import LeakyReLU
import keras

def ConvFactory(filters, kernel_size, padding, inp, name, 
        padding_type='valid', stride=1, train=True):
    if padding != 0:
        padded = ZeroPadding2D(padding)(inp)
    else:
        padded = inp
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding_type, name=name+"_conv", strides=stride,trainable=train)(padded)
    activated = LeakyReLU(0.01)(conv)
    bn = BatchNormalization(name=name+"_bn")(activated)
    return bn

def SimpleFactory(ch_1x1, ch_3x3, inp, name,train=True):
    conv1x1 = ConvFactory(ch_1x1, 1, 0, inp, name + "_1x1",train=train)
    conv3x3 = ConvFactory(ch_3x3, 3, 1, inp, name + "_3x3",train=train)
    return concatenate([conv1x1, conv3x3])


def build_model(patch_size=32, stride=1, train_conv=True):

    inputs = Input(shape=(256, 256, 1))
    # inforcing zero padding of size 32 on the imput image
    c1 = ConvFactory(64, 3, patch_size, inputs, "c1", train=train_conv)
    # the following structure make sure the receptive field is 32*32
    # if change patch_size, need to change the network structure accordingly
    net1 = SimpleFactory(16, 16, c1, "net1", train=train_conv)
    #(None, 318, 318, 32)
    net2 = SimpleFactory(16, 32, net1, "net2", train=train_conv)
    #(None, 318, 318, 48)
    net3 = ConvFactory(16, 14, 0, net2, "net3", train=train_conv)
    #(None, 305, 305, 16)
    net4 = SimpleFactory(112, 48, net3, "net4", train=train_conv)
    #(None, 305, 305, 160)
    net5 = SimpleFactory(64, 32, net4, "net5", train=train_conv)
    #(None, 305, 305, 96)
    net6 = SimpleFactory(40, 40, net5, "net6", train=train_conv)
    #(None, 305, 305, 80)
    net7 = SimpleFactory(32, 96, net6, "net7", train=train_conv)
    #(None, 305, 305, 128)
    net8 = ConvFactory(32, 18, 0, net7, "net8", train=train_conv)
    #(None, 288, 288, 32)
    net9 = ConvFactory(64, 1, 0, net8, "net9", train=train_conv)
    #(None, 288, 288, 64)
    net10 = ConvFactory(64, 1, 0, net9, "net10", train=train_conv)
    #(None, 288, 288, 64)
    final = ConvFactory(1, 1, 0, net10, "net11", stride=stride, train=train_conv)
    #(None, 288, 288, 1)
    model = keras.models.Model(inputs=inputs, outputs=final)
    # print model summary
    print(model.summary())

    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer = adam, loss = 'mae', metrics = ['accuracy'])

    return model
