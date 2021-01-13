import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display

'''
###############################################################################
    Training Data is 512x512 for each square (side by side, colored | lines)
    Try some GAN BOIS
###############################################################################
'''

PATH = '../archive/data'

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_HEIGHT = 512
IMG_WIDTH = 512


def load_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)

    width = tf.shape(image)[1]
    width = width // 2

    colored_image = image[:, :width, :]
    line_image = image[:, width:, :]

    return line_image, colored_image

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

    return line, color

def normalize(color, line):
    color = (color / 127.5) - 1
    line = (line / 127.5) - 1

    return line, color

def load_image_train(filename):
    line, color = load_image(filename)
    line, color = normalize(color, line)
    return line, color

def load_image_test(filename):
    line, color = load_image(filename)
    line, color = normalize(color, line)
    return line, color


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
    Time to build a generator!
##############################################################################
'''

OUTPUT_CHANNELS = 3


## Downsampler: conv2d with leakyReLu and apply batchnorm >> one layer
## Reference: pix2pix (though I think its pretty bad... refer to paper.)

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02) ## <-- what are you doing?

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(
        filters,        # Integer to specify dimensionality of output space
        size,           # height and width of convolution window. Single -> all dim
        strides=2,      # <-- tuple specifying strides of convolution along 
                        #     height/width. Single int means applied to all dimensions
        padding="same", # <-- same  >> even padding to maintain that output and 
                        #              input have same dimensions
                        #     valid >> no padding
        kernel_initializer=initializer, # initializer for kernel weights matrix
        use_bias=False  # whether the layer uses a bias vector
    ))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization()) # normalizes its inputs.

    result.add(tf.keras.layers.LeakyReLU()) # leaky relu gradient 

    return result


## Upsampler: deconv2d, relu? (deepcolor ref)
##   convTrans, dropout, batch norm at all except last? (sketch2color ref)
## argh why they use numpy im confusion


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02) # <-- What are you?!!

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(
        filters,        # Integer to specify dimensionality of output space
        size,           # height and width of convolution window. Single -> all dim
        strides=2,      # <-- tuple specifying strides of convolution along 
                        #     height/width. Single int means applied to all dimensions
        padding="same", # <-- same  >> even padding to maintain that output and 
                        #              input have same dimensions
                        #     valid >> no padding
        kernel_initializer=initializer, # initializer for kernel weights matrix
        use_bias=False  # whether the layer uses a bias vector
    ))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    down_stack = [ ## <-- VERY QuESTIONABLE GOTTA DO MORE RESEARCH ON DOWNSAMPLE
        # so it seems like conv2d will perform matrix multiplication with a kernel 
        # (prob halves dims when doing so)
        downsample(64, 4, apply_batchnorm=False), # (bs, 256, 256, 64)
        downsample(128, 4), # (bs, 128, 128, 128)
        downsample(256, 4), # (bs, 64, 64, 256)
        downsample(512, 4), # (bs, 32, 32, 512)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4)
    ]

    # the last upsample, perform tanh on last one and no batch 
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS, 
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
    Time to build a discriminator!
##############################################################################
'''

def Discriminator(learning_rate, shape=[512, 512, 3]):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=shape, name='input_image')
    targ = tf.keras.layers.Input(shape=shape, name='target_image')

    x = tf.keras.layers.concatenate([inp, targ])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

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

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

    model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return model


# tf.keras.losses.BinaryCrossentropy(from_logits=True) <-- this was what was used
#   in pix2pix



'''

    SELF NOTICE PLEASE CHECK PAPER ON THEIR GENERATOR LOSS AND STUFF. 
    I dont think the generator is much different - both employ a modified U-net

    sketch2color - generator loss has a gan loss = PixLoss + FeatureLoss + TV
    deepcolor - ????

    The paper definitely has better loss function compared to pix2pix - adapt!

'''

## Generator Loss - based on sketch2color
def pix_tv_loss(real, generated):
    l1_loss = tf.reduce_mean(tf.abs(real - generated))

    import tensorflow.keras.backend as kb

    # l2_loss = tf.reduce_mean(kb.sqrt(kb.sum(kb.square(real - generated))))
    tv_loss = kb.abs(kb.sqrt(
        kb.sum(kb.square(generated[:, 1:, :, :] - generated[:, :-1, :, :])) +
        kb.sum(kb.square(generated[:, :, 1:, :] - generated[:, :, :-1, :]))
    ))

    return lambda l1_w, tv_w : l1_w*l1_loss + tv_w*tv_loss


def featurelvl_loss(y, g):
    import tensorflow.keras.backend as kb
    return lambda l2_w : l2_w * kb.mean(kb.sqrt(kb.sum(kb.square(y - g))))


def create_gan(gen, dsc, vgg_net1, vgg_net2, learning_rate, shape=(512, 512, 3)):
    pixelLevelLoss_weight=100
    totalVariationLoss_weight=.0001
    featureLevelLoss_weight=.01

    dsc.trainable = False

    sketch_inp = tf.keras.Input(shape)
    gen_color_output = gen([sketch_inp])
    

    discr_out = dsc([sketch_inp, gen_color_output])
    color_inp = tf.keras.Input(shape)

    net1_out = vgg_net1([tf.image.resize(color_inp, (224,224))]) 
    # by default, resize method is bilinear (3rd arg) ^^vv
    net2_out = vgg_net2([tf.image.resize(gen_color_output, (224,224))])

    l2_loss = featurelvl_loss(net1_out, net2_out)
    l1_tv_loss = pix_tv_loss(color_inp, gen_color_output)

    # creating the GAN model
    model = tf.keras.Model(inputs=[sketch_inp, color_inp], outputs=discr_out)

    # 1st arg is obv learning rate passed in
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)


    # configures the model for training
    model.compile(
        optimizer, 
        loss= lambda y_t, y_p : tf.keras.losses.binary_crossentropy(y_t, y_p) +\
                l2_loss(featureLevelLoss_weight) +\
                l1_tv_loss(pixelLevelLoss_weight, totalVariationLoss_weight)
    )

    return model



'''
##############################################################################
    Time for the train method
##############################################################################
'''

def train(gen, dsc, gan, sketch, image, 
    latent_dim, 
    seed_skets, 
    seed_imgs, 
    output_frequency, 
    n_epochs=100, 
    n_batch=128, ## they're using 128?
    init_epoch=0,
    train_ds,
    test_ds
):
    #batch_per_epoch = TOTAL_IMAGES // n_batch <-- ???????
    #half_batch = n_batch // 2

    for epoch in range(init_epoch, n_epochs):
        start = time.time()
        gen_losses = []
        dsc_losses = []

        display.clear_output(wait=True)

        for n, (input_img, targ) in train_ds.enumerate():
            # train step here
            

            # train dsc on half a batch of samples
            


            # train dsc on half a batch of gen images



            # train gen on a batch