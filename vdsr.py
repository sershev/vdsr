from keras.layers import Conv2D, Input, add
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import cv2
import glob
import random
import numpy as np
import os

class VDSR():

    def __init__(self, model_path=None, save_path="", num_layers=10, validation_steps=10):

        self.SCR_DIR = "./data/"
        self.TRAIN_IMAGE_FILES = glob.glob(self.SCR_DIR + "train/*/*")
        self.TRAIN_TOTAL_IMAGES = len(self.TRAIN_IMAGE_FILES)

        self.TEST_IMAGE_FILES = glob.glob(self.SCR_DIR + "test/*/*")
        self.TEST_TOTAL_IMAGES = len(self.TEST_IMAGE_FILES)

        #self.mc = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks = [] #[self.mc]

        if model_path:
            self.model = load_model(model_path)
        else:
            self.model = VDSR.create_model(num_layers)

        self.validation_steps = validation_steps


    @staticmethod
    def create_model(num_layers):

        input = Input(shape=(None, None, 3))

        l = Conv2D(64, (3, 3), padding="same", activation="relu")(input)

        for i in range(num_layers-1) :
            l = Conv2D(64, (3,3), padding="same", activation="relu")(l)

        l = Conv2D(3, (3, 3), padding="same")(l)
        l = add([input, l])

        m = Model(inputs=input, outputs=l)

        return m


    def batch_generator(self, scales=[2,3,4], train=True):

        while True:
                scale = random.sample(scales, 1)[0]

                X = []
                Y = []

                if train:
                    sample_number = random.randint(0, self.TRAIN_TOTAL_IMAGES-1)
                    img = cv2.imread(self.TRAIN_IMAGE_FILES[sample_number])
                else:
                    sample_number = random.randint(0, self.TEST_TOTAL_IMAGES-1)
                    img = cv2.imread(self.TEST_IMAGE_FILES[sample_number])

                img = img/255.

                Y.append(img.copy())
                shape = img.shape[0:2][::-1]
                img = cv2.resize(img, (0,0), fx=1./scale, fy=1./scale)
                img = cv2.resize(img, shape)
                X.append(img)

                yield np.array(X), np.array(Y)


    def train(self, train_scales, test_scales, filepath="./models/model.h5", steps_per_epoch=100, epochs=10, ):



        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit_generator(
            self.batch_generator(train=True, scales=train_scales),
            steps_per_epoch=steps_per_epoch,
            validation_data=self.batch_generator(train=False, scales=test_scales),
            validation_steps=10,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=1)


    def predict_one(self, image_path, scale, debug = False):

        if os.path.isfile(image_path):
            img = cv2.imread(image_path) / 255.

            scaled_img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale))
            new_shape = (1,) + scaled_img.shape
            scaled_img_reshaped = np.reshape(scaled_img, new_shape)

            hr_img = np.clip(self.model.predict(scaled_img_reshaped)[0], 0, 1)

            if debug:
                cv2.imshow("origin", (img * 255).astype(np.uint8))
                cv2.imshow("interpolated", (scaled_img * 255).astype(np.uint8))
                cv2.imshow("hr-image", (hr_img * 255).astype(np.uint8))
                while True:
                    key = cv2.waitKey(3000)
                    if key == 27:  # ESC
                        break

            return hr_img


if __name__ == "__main__":

    vdsr = VDSR()
    vdsr.train(train_scales=[2], test_scales=[2], steps_per_epoch = 100, epochs = 10)