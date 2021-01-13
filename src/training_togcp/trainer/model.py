from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras import Model
import tensorflow as tf

import os
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display
import numpy as np


def load_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)

    width = tf.shape(image)[1]
    width = width // 2

    colored_image = image[:, :width, :]
    line_image = image[:, width:, :]

    colored_image = tf.cast(colored_image, tf.float32)
    line_image = tf.cast(line_image, tf.float32)

    return line_image, colored_image

def resize(line, color, height, width):
    color = tf.image.resize(
        color,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    line = tf.image.resize(
        line, 
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return line, color

def normalize(line, color):
    color = (color / 127.5) - 1
    line = (line / 127.5) - 1

    return line, color

def load_image_train(filename):
    color, line = load_image(filename)
    color, line = normalize(line, color)
    return line, color

def load_image_test(filename):
    line, color = load_image(filename)
    line, color = normalize(line, color)
    return line, color

'''
##############################################################################
    Aux Functions to Assist
##############################################################################
'''

def downsample(filters, size, strides=2, padding="same", apply_batchnorm=True, alpha=0.2):
    initializer = tf.random_normal_initializer(0., 0.02) ## <-- what are you doing?

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(
        filters,        # Integer to specify dimensionality of output space
        size,           # height and width of convolution window. Single -> all dim
        strides=strides,      # <-- tuple specifying strides of convolution along 
                        #     height/width. Single int means applied to all dimensions
        padding=padding, # <-- same  >> even padding to maintain that output and 
                        #              input have same dimensions
                        #     valid >> no padding
        kernel_initializer=initializer, # initializer for kernel weights matrix
        use_bias=False  # whether the layer uses a bias vector
    ))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization()) # normalizes its inputs.

    result.add(tf.keras.layers.LeakyReLU(alpha=alpha)) # leaky relu gradient 

    return result


def upsample(filters, size, strides=2, padding="same", apply_batchnorm=True, dropout=False, alpha=0.2):
    initializer = tf.random_normal_initializer(0., 0.02) # <-- What are you?!!

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(
        filters,        # Integer to specify dimensionality of output space
        size,           # height and width of convolution window. Single -> all dim
        strides=strides,        # <-- tuple specifying strides of convolution along 
                        #     height/width. Single int means applied to all dimensions
        padding=padding, # <-- same  >> even padding to maintain that output and 
                        #              input have same dimensions
                        #     valid >> no padding
        kernel_initializer=initializer, # initializer for kernel weights matrix
        use_bias=False  # whether the layer uses a bias vector
    ))

    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())

    if dropout:
        result.add(tf.keras.layers.Dropout(dropout))

    result.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    ## pix2pix uses relu ^^

    return result

'''
##############################################################################
    Generator
##############################################################################
'''

def Generator(drop_rate, alpha, inp_shape=(512,512,3)):
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    down_stack = [ ## <-- VERY QuESTIONABLE GOTTA DO MORE RESEARCH ON DOWNSAMPLE
        # so it seems like conv2d will perform matrix multiplication with a kernel 
        # (prob halves dims when doing so)
        downsample(16, 4, apply_batchnorm=False, alpha=alpha), # (bs, 256, 256, 64)
        downsample(32, 4, alpha=alpha), # (bs, 128, 128, 128)
        downsample(64, 4, alpha=alpha), # (bs, 64, 64, 256)
        downsample(128, 4, alpha=alpha), # (bs, 32, 32, 512)
        downsample(128, 4, alpha=alpha), # (bs, 16, 16, 512)
        downsample(128, 4, alpha=alpha), # (bs, 8, 8, 512)
        downsample(128, 4, alpha=alpha), # (bs, 4, 4, 512)
        downsample(128, 4, alpha=alpha), # (bs, 2, 2, 512)
    ]

    up_stack = [
        upsample(128, 4, dropout=drop_rate, alpha=alpha),
        upsample(128, 4, dropout=drop_rate, alpha=alpha),
        upsample(128, 4, dropout=drop_rate, alpha=alpha),
        upsample(128, 4, alpha=alpha),
        upsample(128, 4, alpha=alpha),
        upsample(64, 4, alpha=alpha),
        upsample(32, 4, alpha=alpha)
    ]

    # the last upsample, perform tanh on last one and no batch 
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        inp_shape[2], 
        4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh'
    )

    # perform downsampling and create residual connections
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1]) # excludes last one? idk why but it seems like 
                                 # deepcolor did the same thing

    # perform upsample on x 
    # (deepcolor did it slightly differently, seemed to have skipped first one 
    #   and cat on last)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

'''
##############################################################################
    Discriminator
##############################################################################
'''

def Discriminator(alpha, learning_rate, shape=[512, 512, 3]):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=shape, name='input_image')
    targ = tf.keras.layers.Input(shape=shape, name='target_image')

    x = tf.keras.layers.concatenate([inp, targ])

    down1 = downsample(16, 4, apply_batchnorm=False, alpha=alpha)(x)
    down2 = downsample(32, 4, alpha=alpha)(down1)
    down3 = downsample(64, 4, alpha=alpha)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        512,
        4, 
        strides=1, 
        kernel_initializer=initializer, 
        use_bias=False
    )(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    last = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer=initializer
    )(leaky_relu)

    model = tf.keras.Model(inputs=[inp, targ], outputs=last)

    #optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

    #model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return model

'''
##############################################################################
    Loss Functions
##############################################################################
'''

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, vgg_net1, vgg_net2, shape=(512, 512, 3)):
    pixelLevelLoss_weight=100
    totalVariationLoss_weight=.0001
    featureLevelLoss_weight=.01

    net1_out = vgg_net1([tf.image.resize(target, (224,224))]) 
    net2_out = vgg_net2([tf.image.resize(gen_output, (224,224))])

    ftloss = K.mean(K.sqrt(K.sum(K.square(net1_out - net2_out))))

    ganloss = loss_object(
        tf.ones_like(disc_generated_output), 
        disc_generated_output
    )

    tvloss = K.abs(K.sqrt(
        K.sum(K.square(gen_output[:, 1:, :, :] - gen_output[:, :-1, :, :])) + 
        K.sum(K.square(gen_output[:, :, 1:, :] - gen_output[:, :, :-1, :]))
    ))
    l1loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = ganloss + pixelLevelLoss_weight*l1loss + totalVariationLoss_weight*tvloss + featureLevelLoss_weight*ftloss

    return total_gen_loss, ganloss, l1loss, tvloss



def showImages(model, testInput, target):
    prediction = model(testInput, training=True)
    plt.figure(figsize=(15,15))

    display_list = [testInput[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.show()


'''
##############################################################################
    Train steps
##############################################################################
'''

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

@tf.function
def train_step(input, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dsc_tape:
        gen_output = generator(input, training=True)

        dsc_real = discriminator([input, target], training=True)
        dsc_gen = discriminator([input, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            dsc_gen, 
            gen_output, 
            target, 
            vggn1,
            vggn2
        )
        dsc_loss = discriminator_loss(dsc_real, dsc_gen)

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = dsc_tape.gradient(
        dsc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('dsc_loss', dsc_loss, step=epoch)