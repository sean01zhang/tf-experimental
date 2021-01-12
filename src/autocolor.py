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


'''
###############################################################################
    Training Data is 512x512 for each square (side by side, colored | lines)
    Try some GAN BOIS
###############################################################################
'''
PATH = '../archive/data'

BUFFER_SIZE = 400
BATCH_SIZE = 10
IMG_HEIGHT = 512
IMG_WIDTH = 512

# config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# #device_count = {'GPU': 1}
# )
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)


def load_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)

    width = tf.shape(image)[1]
    width = width // 2

    colored_image = image[:, :width, :]
    line_image = image[:, width:, :]

    colored_image = tf.cast(colored_image, tf.float32)
    line_image = tf.cast(line_image, tf.float32)

    return colored_image, line_image

def resize(color, line, height, width):
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

    return color, line

def normalize(color, line):
    color = (color / 127.5) - 1
    line = (line / 127.5) - 1

    return color, line

def load_image_train(filename):
    color, line = load_image(filename)
    color, line = normalize(color, line)
    return color, line

def load_image_test(filename):
    color, line = load_image(filename)
    color, line = normalize(color, line)
    return color, line

'''
##############################################################################
    Here is the input pipeline.
##############################################################################
'''
train_dataset = tf.data.Dataset.list_files(PATH + "/train/*.png")
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + "/val/*.png")
test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)


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

    return total_gen_loss, ganloss, l1loss


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

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    return real_loss + generated_loss
'''
##############################################################################
    Helpful functions
##############################################################################
'''
# def writeLog(callback, name, loss, batchNum, flush=False):
#     summary = tf.Summary()
#     summary_value = summary.value.add()
#     summary_value.tag = name
#     summary_value.simple_value = loss
#     callback.writer.add_summary(summary, batch_no)

#     if flush:
#       callback.writer.flush()

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
    Verify Generators Work
##############################################################################
'''
generator = Generator(drop_rate=0.5, alpha=0.2)
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

discriminator = Discriminator(alpha=0.2, learning_rate=0.0002)
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16

vgg = VGG16(weights='imagenet')
vggn1 = Model(inputs=vgg.input, outputs=tf.keras.layers.ReLU()(vgg.get_layer('block2_conv2').output))
vggn2 = Model(inputs=vgg.input, outputs=tf.keras.layers.ReLU()(vgg.get_layer('block2_conv2').output))


for example_input, example_target in test_dataset.take(1):
  showImages(generator, example_input, example_target)


'''
##############################################################################
    Train steps
##############################################################################
'''

import datetime
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

'''
##############################################################################
    Time to train!
##############################################################################
'''

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for input, targ in test_ds.take(1):
            showImages(generator, input, targ)

        print(f"Epoch: {epoch}")

        for n, (input, targ) in train_ds.enumerate():
            print('.', end="")
            if (n+1) % 100 == 0:
                print()
            train_step(input, targ, epoch)
        print()

        if (epoch+1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Time taken for epoch {epoch+1} is {time.time()-start} s\n")
    checkpoint.save(file_prefix=checkpoint_prefix)



'''
##############################################################################
    Do the magic!
##############################################################################
'''

fit(train_dataset, 50, test_dataset)