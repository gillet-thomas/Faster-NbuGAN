from __future__ import print_function
import sys
import os, cv2
from PIL import Image
import numpy as np


import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions


class FasterNBUGan():

    def __init__(self, ancestor, directory, path, target, targx):
        self.ancestor = ancestor
        self.directory = directory
        self.path = path
        self.target = target
        self.targx = targx

        inputs = Input(shape=(224, 224, 3)) # Input image shape
        optimizer_g = Adam(0.0002)
        optimizer_d = SGD(0.01)

        # Build generator
        generator = self.build_generator(inputs)
        self.G = Model(inputs, generator)
        self.G._name = 'Generator'

        # Build discriminator
        discriminator = self.build_discriminator(self.G(inputs))
        self.D = Model(inputs, discriminator)
        self.D.compile(loss=tensorflow.keras.losses.binary_crossentropy, optimizer=optimizer_d, metrics=[self.custom_acc])
        self.D._name = 'Discriminator'

        # VGG16 trained on ImageNet is used
        self.target = VGG16(weights='imagenet')
        self.target.trainable = False

        # Build GAN: stack generator, discriminator and target
        img = (self.G(inputs) / 2 + 0.5) * 255  # image's pixels will be between [0, 255]
        self.stacked = Model(inputs=inputs, outputs=[self.G(inputs), self.D(inputs), self.target(preprocess_input(img))])
        self.stacked.compile(loss=[self.generator_loss, tensorflow.keras.losses.binary_crossentropy,
                                   tensorflow.keras.losses.categorical_crossentropy], optimizer=optimizer_g)
        
        self.stacked.summary()

    def generator_loss(self, y_true, y_pred):
        # Hinge loss
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.3, 0), axis=-1)  # Hinge loss

    def custom_acc(self, y_true, y_pred):
        return binary_accuracy(K.round(y_pred), K.round(y_true))

    # Basic classification model
    def build_discriminator(self, inputs):
        D = Conv2D(64, 3, strides=(2, 2), padding='same')(inputs)  # Downsample the image by 2
        D = LeakyReLU()(D)                                         # Activation function
        
        D = Conv2D(128, 3, strides=(2, 2), padding='same')(D) 
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)

        D = Conv2D(256, 3, strides=(2, 2), padding='same')(D) 
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)

        D = Conv2D(512, 3, strides=(2, 2), padding='same')(D) 
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)

        D = Conv2D(512, 3, strides=(2, 2), padding='same')(D) 
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)

        D = Flatten()(D)
        D = Dense(512)(D)
        D = LeakyReLU()(D)

        D = Dense(1, activation='sigmoid')(D)

        return D

    def build_generator(self, generator_inputs):
        G = Conv2D(64, 3, padding='same')(generator_inputs)         # Input shape (none, 224, 224, 3)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)                                   # G shape (none, 224, 224, 64)

        G = Conv2D(128, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)                                   # G shape (none, 112, 112, 128)

        G = Conv2D(256, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)                                   # G shape (none, 56, 56, 256)
        residual = G

        # Residual Blocks
        for _ in range(4):
            G = Conv2D(256, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = Activation('relu')(G)
            G = Conv2D(256, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = layers.add([G, residual])
            residual = G                                            # G shape (none, 56, 56, 256)

        G = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)                                   # G shape (none, 112, 112, 128)

        G = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)                                   # G shape (none, 224, 224, 64)

        G = Conv2D(3, 3, padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('tanh')(G)

        G = layers.add([G * 8 / 255, generator_inputs])             # Add the [generated noise * epsilon] to the original image

        return G

    def train_discriminator(self, x_batch, Gx_batch):
        # Gx_bacth = G(z) = batch of fake images
        # train real images on disciminator: D(x) = update D params per classification for real images
        # train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
        self.D.trainable = True
        d_loss_real = self.D.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(len(x_batch), 1)))  # real=1, positive label smoothing
        d_loss_fake = self.D.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)))                        # fake=0
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)         # Average loss for real and fake images
        return d_loss                                           # (discriminator loss, discriminator accuracy)

    def train_generator(self, x_batch):
        arr = np.zeros(1000)
        arr[targx] = 1                                          # targx is index of the target class (rhinoceros beetle)
        full_target = np.tile(arr, (len(x_batch), 1))

        # Only updata params of Generator
        self.D.trainable = False
        self.target.trainable = False

        # input x_batch is the real images that will be used to generate the adversarial images
        # target 1 = target for generator = real image for all generated images
        # target 2 = target for discriminator = 1 for all generated images
        # target 3 = target for VGG16 = target class for all generated images
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), full_target])
        return stacked_loss  # (generator loss, hinge loss, gan loss, adv loss)

    def train_GAN(self):

        # Create a directory to save the adversarial images
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Load and preprocess image
        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        x_train = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        x_train = np.expand_dims(x_train, axis=0)                              # (1, 224, 224, 3)
        x_train = np.array(x_train, dtype=np.float32)
        x_train = (x_train * 2. / 255 - 1).reshape(len(x_train), 224, 224, 3)  # Normalized image between [-1, 1]

        epochs = EPOCH
        for epoch in range(epochs):
            print("===========================================")
            print("EPOCH: ", epoch)
            Gx = self.G.predict(x_train)    # Gx (1, 224, 224, 3) from -1.0086238 to 1.0082918 (not clipped)
            Gx = np.clip(Gx, -1, 1)         # Gx (1, 224, 224, 3) from -1.0 to 1.0 (safety check)

            (d_loss, d_acc) = self.train_discriminator(x_train, Gx)
            (g_loss, hinge_loss, gan_loss, adv_loss) = self.train_generator(x_train)

            print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nTarget Loss: %f\nGAN Loss: %f" % (d_loss, d_acc * 100., g_loss, adv_loss, gan_loss))
            np.save(directory + '/adversImg.npy', Gx)

            # For keras: scale image to 255, reshape, preprocess, and predict
            img_normalized = np.load(directory + "/adversImg.npy").copy()
            img = (img_normalized / 2.0 + 0.5) * 255    # (1, 224, 224, 3) 0.0 to 255.0
            image = img.reshape((1, 224, 224, 3))       # (1, 224, 224, 3) 0.0 to 255.0 (safety check)
            Image.fromarray((image[0]).astype(np.uint8)).save(directory + "/adv_loaded.png", 'png')

            # Load original image
            og = cv2.imread(path)
            og = cv2.cvtColor(og, cv2.COLOR_BGR2RGB)
            og_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            # Noise Blowing Up Strategy -> extract noise, scale it, and add it back to the original image
            # 'final' is the resulting image from the NBU strategy

            # Resize final image to 224x224, and clip it
            final_R = cv2.resize(final, (224,224), interpolation=cv2.INTER_LANCZOS4)
            final_R = np.clip(final_R, 0, 255)

            # Preprocess and predict final image
            image = final_R.reshape((1, 224, 224, 3))                                   
            yhat = self.target.predict(preprocess_input(image))
            pred_labels = decode_predictions(yhat, 1000)                                

            # Print max probability class and target class probability
            pred_max = pred_labels[0][0] # to check
            for label in pred_labels[0]:
                if label[1] == target:
                    print(label[1], label[2]) # to check
            print(pred_max[1], pred_max[2]) # to check

            # Save image if target class is predicted with probability >= 0.5 or at the last epoch
            if (np.argmax(yhat, axis=1) == targx and pred_max[2] >= 0.5) or epoch == epochs - 1:
                Img = Image.fromarray(final.astype(np.uint8))
                filename = f"hr.png" # to check
                Img.save(directory + "/" + filename, 'png')

                full_path = os.path.join(directory, "epoch_value.txt")
                with open(full_path, 'w') as file:
                    file.write(str(epoch))

                break

def get_targets(ancestor):
    if (ancestor == "abacus"):
        target = 'bannister'
        targx = 421
    elif (ancestor == "acorn"):
        target = 'rhinoceros_beetle'
        targx = 306
    elif (ancestor == "baseball"):
        target = 'ladle'
        targx = 618
    elif (ancestor == "broom"):
        target = 'dingo'
        targx = 273
    elif (ancestor == "brown_bear"):
        target = 'pirate'
        targx = 724
    elif (ancestor == "canoe"):
        target = 'Saluki'
        targx = 176
    elif (ancestor == "hippopotamus"):
        target = 'trifle'
        targx = 927
    elif (ancestor == "llama"):
        target = 'agama'
        targx = 42
    elif (ancestor == "maraca"):
        target = 'conch'
        targx = 112
    elif (ancestor == "mountain_bike"):
        target = 'strainer'
        targx = 828

    return target, targx


if __name__ == "__main__":
    EPOCH = 10000
    SEED = 42
    np.random.seed(5)
    tensorflow.random.set_seed(1)

    ancestor = sys.argv[1]
    ancestor_id = sys.argv[2]

    directory = f"results/{ancestor}"
    path = f"ancestors/{ancestor}/{ancestor}{ancestor_id}.png"

    target, targx = get_targets(ancestor)

    dcgan = FasterNBUGan(ancestor, directory, path, target, targx)
    dcgan.train_GAN()
