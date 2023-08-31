"""
CNN training verification image
"""
import json
import io
import os
import os.path

import keras_preprocessing.image
import numpy as np
import PIL.Image
import requests
import tensorflow as tf
# from tensorflow import keras


def label_to_array(text, labels):
    """
    :return: numpy array
    """
    hots = np.zeros(shape=(len(labels) * len(text)))
    for i, char in enumerate(text):
        index = i * len(labels) + labels.index(char)
        hots[index] = 1
    return hots


def array_to_label(array, labels):
    """
    :param array: numpy array
    :param labels: label
    :return: label string
    """
    text = []
    for index in array:
        text.append(labels[index])
    return ''.join(text)


def load_image_data(image_dir_path, image_height, image_width, labels, target_label_length):
    """
    load image data
    RGB change to grayscale
    """
    image_name_list = os.listdir(image_dir_path)
    image_data = np.zeros(shape=(len(image_name_list), image_height, image_width, 1))
    label_data = np.zeros(shape=(len(image_name_list), len(labels) * target_label_length))

    for index, image_name in enumerate(image_name_list):
        img = keras_preprocessing.image.utils.load_img(os.path.join(image_dir_path, image_name), color_mode='grayscale')
        x = keras_preprocessing.image.utils.img_to_array(img)
        # print(image_name.split('_')[0])
        
        y = label_to_array(image_name.split('_')[0], labels)
        if hasattr(img, 'close'):
            img.close()
        image_data[index] = x
        label_data[index] = y
    return image_data, label_data


class FixCaptchaLengthModel(object):
    """
    verification image length model
    Attributes:
        image_height: 
        image_width: 
        learning_rate: 
        dropout: dropout
        label_number: 
        fixed_length: 
    """

    def __init__(self, image_height, image_width, label_number, fixed_length,
                 learning_rate=0.0001, dropout=0.25):
        self.image_height = image_height
        self.image_width = image_width
        # 这里固定转化为灰度图像
        self.image_channel = 1
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.label_number = label_number
        self.fixed_length = fixed_length
        self.kernel_size = (2, 2)#(3, 3)
        self.pool_size = (2, 2)
        self.padding = 'valid'
        self.activation = 'relu'

    def model(self):
        """
        :return: keras.Sequential instance
        """
        #model = keras.Sequential()
        model = tf.keras.Sequential()
        #optimizer = tf.keras.optimizers.Adam()
        #model.compile(optimizer=optimizer )
        
        # input layer
        input = tf.keras.Input(shape=(self.image_height, self.image_width, self.image_channel), batch_size=None)
        model.add(input)
        # one layer
        model.add(tf.keras.layers.Convolution2D(filters=32, kernel_size=self.kernel_size, strides=1, padding=self.padding,
                                       activation=self.activation))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.pool_size))
        model.add(tf.keras.layers.Dropout(rate=self.dropout))
        # two layer
        model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=self.kernel_size, strides=1, padding=self.padding,
                                       activation=self.activation))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.pool_size))
        model.add(tf.keras.layers.Dropout(rate=self.dropout))
        # three layer
        model.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=self.kernel_size, strides=1, padding=self.padding,
                                       activation=self.activation))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.pool_size))
        model.add(tf.keras.layers.Dropout(rate=self.dropout))
        model.add(tf.keras.layers.Flatten())
        # four layer
        model.add(tf.keras.layers.Dense(units=1024, activation=self.activation))
        model.add(tf.keras.layers.Dropout(rate=self.dropout))
        # five layer
        model.add(tf.keras.layers.Dense(units=self.fixed_length * self.label_number, activation="sigmoid"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="binary_crossentropy",
                      metrics=["binary_accuracy"])
        return model

    def load_from_disk(self, model_file_path):
        """
        :param model_file_path: 
        :return: keras.Sequential
        """
        if not os.path.exists(model_file_path):
            raise Exception('%s do not exists' % model_file_path)
        model = self.model()
        model.load_weights(model_file_path)
        return model


class CheckAccuracyCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_x, train_y, validation_x, validation_y, label_number, fixed_label_length, batch_size=128):
        super(CheckAccuracyCallback, self).__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.label_number = label_number
        self.fixed_label_length = fixed_label_length
        self.batch_size = batch_size

    def _compare_accuracy(self, data_x, data_y):
        predict_y = self.model.predict_on_batch(data_x)
        predict_y = tf.keras.backend.reshape(predict_y, [len(data_x), self.fixed_label_length, self.label_number])
        data_y = tf.keras.backend.reshape(data_y, [len(data_y), self.fixed_label_length, self.label_number])
        equal_result = tf.keras.backend.equal(tf.keras.backend.argmax(predict_y, axis=2),
                                           tf.keras.backend.argmax(data_y, axis=2))
        return tf.keras.backend.mean(tf.keras.backend.min(tf.keras.backend.cast(equal_result, tf.float32), axis=1))

    def on_epoch_end(self, epoch, logs=None):
        print('\nEpoch %s with logs: %s' % (epoch, logs))
        # 
        batches = (len(self.train_x) + self.batch_size - 1) / self.batch_size
        target_batch = (epoch + 1) % batches
        batch_start = int((target_batch - 1) * self.batch_size)
        batch_x = self.train_x[batch_start: batch_start + self.batch_size]
        batch_y = self.train_y[batch_start: batch_start + self.batch_size]
        on_train_batch_acc = self._compare_accuracy(batch_x, batch_y)
        print('Epoch %s with image accuracy on train batch: %s' % (epoch, tf.keras.backend.eval(on_train_batch_acc)))
        on_test_batch_acc = self._compare_accuracy(self.validation_x, self.validation_y)
        print('Epoch %s with image accuracy on validation: %s\n' % (epoch, tf.keras.backend.eval(on_test_batch_acc)))


class Config(object):

    def __init__(self, **kwargs):
        self.image_height = kwargs['image_height']
        self.image_width = kwargs['image_width']
        self.fixed_length = kwargs['fixed_length']
        self.train_batch_size = kwargs['batch_size']
        self.model_save_path = kwargs['save_path']
        self.labels = kwargs['labels']
        self.train_image_dir = kwargs['train_image_dir']
        self.validation_image_dir = kwargs['validation_image_dir']
        self.learning_rate = kwargs['learning_rate']
        self.dropout_rate = kwargs['dropout_rate']
        self.epochs = kwargs['epochs']

    @staticmethod
    def load_configs_from_json_file(file_path='fixed_length_captcha.json'):
        """
        :param file_path: file path
        :return: dict instance
        """
        with open(file_path, 'r') as fd:
            config_content = fd.read()
        return Config(**json.loads(config_content))


class Predictor(object):
    """
    
    """

    def __init__(self, config_file_path='fixed_length_captcha.json'):
        self.config = Config.load_configs_from_json_file(config_file_path)
        self.model = FixCaptchaLengthModel(self.config.image_height, self.config.image_width, len(self.config.labels),
                                           self.config.fixed_length, learning_rate=self.config.learning_rate,
                                           dropout=self.config.dropout_rate).load_from_disk(self.config.model_save_path)
        self.label_number = len(self.config.labels)

    def predict(self, image_file_path):
        """
        :param image_file_path: 
        :return: predict text
        """
        with open(image_file_path, 'rb') as f:
            return self.predict_single_image_content(f.read())

    def predict_remote_image(self, remote_image_url, headers=None, timeout=30, save_image_to_file=None):
        """
        预测远程图片
        :param remote_image_url: URL
        :param headers: 
        :param timeout: 
        :param save_image_to_file: 
        :return: predict text
        """
        response = requests.get(remote_image_url, headers=headers, timeout=timeout, stream=True)
        content = response.content
        if save_image_to_file is not None:
            with open(save_image_to_file, 'wb') as fd:
                fd.write(content)
        return self.predict_single_image_content(content)

    def predict_single_image_content(self, image_content):
        """
        :param image_content: byte content
        :return: predict text
        """
        p_image = PIL.Image.open(io.BytesIO(image_content))
        if p_image.mode not in ('L', 'I;16', 'I'):
            p_image = p_image.convert('L')
        image_data = np.zeros(shape=(1, self.config.image_height, self.config.image_width, 1))
        image_data[0] = keras_preprocessing.image.img_to_array(p_image)
        if hasattr(p_image, 'close'):
            p_image.close()
        result = self.model.predict_on_batch(image_data)
        result = tf.keras.backend.reshape(result, [1, self.config.fixed_length, self.label_number])
        result = tf.keras.backend.argmax(result, axis=2)
        return array_to_label(tf.keras.backend.eval(result)[0], self.config.labels)


def train():
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    # tf.config.list_physical_devices('CPU')
    tf.device('/GPU:0')
    # is_gpu = len(tf.config.list_physical_devices('GPU')) > 0 
    # tf.debugging.set_log_device_placement(True)

    # tf.test.gpu_device_name()

    # tf.device(f'/{device}:0')
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    config = Config.load_configs_from_json_file()
    train_x, train_y = load_image_data(config.train_image_dir, config.image_height, config.image_width,
                                       config.labels, config.fixed_length)
    validation_x, validation_y = load_image_data(config.validation_image_dir, config.image_height, config.image_width,
                                                 config.labels, config.fixed_length)
    print('total train image number: %s' % len(train_x))
    print('total validation image number: %s' % len(train_y))
    model = FixCaptchaLengthModel(config.image_height, config.image_width, len(config.labels), config.fixed_length,
                                  learning_rate=config.learning_rate, dropout=config.dropout_rate)
    if os.path.exists(config.model_save_path):
        # print(11111111111111111111111111111111)
        model = model.load_from_disk(config.model_save_path)
    else:
        # print(22222222222222222222222222222222)
        model = model.model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=config.model_save_path, save_weights_only=True, save_best_only=True),
        CheckAccuracyCallback(train_x, train_y, validation_x, validation_y, len(config.labels), config.fixed_length,
                              batch_size=config.train_batch_size)
    ]
    model.fit(train_x, train_y, batch_size=config.train_batch_size, epochs=config.epochs,
              validation_data=(validation_x, validation_y), callbacks=callbacks)

    model.save('./model/model.dat') 
    
if __name__ == '__main__':
    # train()
    # predictor = Predictor()
    # print(predictor.predict_remote_image('url', save_image_to_file='./train_images/75421_1.png'))
    """
    """

    predictor = Predictor()
    result = predictor.predict('./download_image/1218.png')
    print(result)
    # predictor.predict_single_image_content(b'PNGxxxxx')
    # predictor.predict_remote_image('http://xxxxxx/xx.jpg', save_image_to_file='remote.jpg')