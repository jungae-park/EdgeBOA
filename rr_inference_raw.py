import numpy as np
import pathlib
import os
import time
import random
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.3*1024)])
  

mobilenetv1_model = tf.keras.applications.MobileNet(weights='imagenet')
mobilenetv2_model = tf.keras.applications.MobileNetV2(weights='imagenet')
inceptionv3_model = tf.keras.applications.InceptionV3(weights='imagenet')

image_path = pathlib.Path('./dataset/imagenet/imagenet_1000_raw')
raw_image_path = sorted(list(image_path.glob('*.JPEG')))

def mobilenet_load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[224, 224])
    
def inception_load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[299, 299])

def image_to_array(image):
    return tf.keras.preprocessing.image.img_to_array(image, dtype=np.int32)

def mobilenetv1_image_preprocess(image_array):
    return tf.keras.applications.mobilenet.preprocess_input(
        image_array[tf.newaxis, ...])

def mobilenetv2_image_preprocess(image_array):
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array[tf.newaxis, ...])

def inceptionv3_image_preprocess(image_array):
    return tf.keras.applications.inception_v3.preprocess_input(
        image_array[tf.newaxis, ...])



models = ['mobilenetv1', 'mobilenetv2', 'inceptionv3']

selected_models = random.choices(models, k=10)

mobilenetv1=[]
mobilenetv2=[]
inceptionv3=[]
mobilenetv1_image_preprocessed=[]
mobilenetv2_image_preprocessed=[]
inceptionv3_image_preprocessed=[]

for model in selected_models:
    print(model)
    if models=='mobilenetv1':
        if len(mobilenetv1_image_preprocessed)!=1000:
            for image_path in raw_image_path:
                mobilenet_image = mobilenet_load_image(image_path)
                mobilenet_image_array = image_to_array(mobilenet_image)
                mobilenetv1_images_preprocessed = mobilenetv1_image_preprocess(mobilenet_image_array)
                mobilenetv1_image_preprocessed.append(mobilenetv1_images_preprocessed)
        else:
            start_time=time.time()
            for inference_image in mobilenetv1_image_preprocessed:
                preds = mobilenetv1_model.predict(inference_image)
            mobilenetv1.append(time.time()-start_time)
    elif model=='mobilenetv2':
        if len(mobilenetv2_image_preprocessed)!=1000:
            for image_path in raw_image_path:
                mobilenet_image = mobilenet_load_image(image_path)
                mobilenet_image_array = image_to_array(mobilenet_image)
                mobilenetv2_images_preprocessed = mobilenetv2_image_preprocess(mobilenet_image_array)
                mobilenetv2_image_preprocessed.append(mobilenetv2_images_preprocessed)

        else:
            start_time=time.time()
            for inference_image in mobilenetv2_image_preprocessed:
                preds = mobilenetv2_model.predict(inference_image)
            mobilenetv2.append(time.time()-start_time)
    else:
        if len(inceptionv3_image_preprocessed)!=1000:
            for image_path in raw_image_path:
                inception_image = inception_load_image(image_path)
                inception_image_array = image_to_array(inception_image)
                inceptionv3_images_preprocessed = inceptionv3_image_preprocess(inception_image_array)
                inceptionv3_image_preprocessed.append(inceptionv3_images_preprocessed)
        else:
            start_time=time.time()
            for inference_image in inceptionv3_image_preprocessed:
                preds = inceptionv3_model.predict(inference_image)
            inceptionv3.append(time.time()-start_time)


print('total time:', sum(mobilenetv1+mobilenetv2+inceptionv3))
