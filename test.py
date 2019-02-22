from vdsr import VDSR
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def run(image_path = "", model_path = "./models/model.h5", scale=4):

    vdsr = VDSR(model_path)
    vdsr.predict_one(image_path, debug=True, scale=scale)

if __name__ == "__main__":


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))


    img_path = r"D:\ml\resources\datasets\faces\putin\Putin_0007.jpg"
    #img_path = r"D:\ml\vdsr\data\test\Urban100\img090.jpg"
    run(img_path, scale=2)