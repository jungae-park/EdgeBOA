import pandas as pd
import numpy as np
from BOA import BOA  
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


df = pd.DataFrame([[0.815430419921875,3.1736192703247,6.14680004119873,126.965363264083,0.126184133291244,7.33752252877316,7.92492664423908,1.15,4.25,270.5882353],
[0.695271362304687,6.59675288200378,6.22818779945373,127.261553764343,0.12682222533226,7.13844688566247,7.8850532497763,0.615,3.53,174.2209632],
[0.836008178710937,23.3072459697723,5.74585628509521,140.173117160797,0.139707246303558,5.9092497808731,7.15782485489108, 11.5, 23.8, 483.1932773]],
columns=['Accuracy','Model_load_time','Data_load_time','Inference_task_time', 'Inference_time','IPS_model_load_dataset_load_inference','IPS_inference_only','Flop_giga','parameter_m','Flop_parameter_m'],
index=['mobilenetv1','mobilenetv2','inceptionv3'])

dependent_vars = ['Accuracy', 'Inference_time', 'IPS_inference_only']
independent_vars = ['Model_load_time', 'Data_load_time', 'Flop_giga']

def optimize_with_boa(df, dependent_vars, independent_vars):
    boa = BOA()

def select_and_infer_with_boa(df, n=10):
    selected_models = np.random.choice(df.index, size=n, replace=True)
    results = []
    for model in selected_models:
        model_data = df.loc[model]
        optimize_with_boa(model_data, dependent_vars, independent_vars)
        results.append(model)
    return results
 
results = select_and_infer_with_boa(df)
print("model list:", results)


mobilenetv1=[]
mobilenetv2=[]
inceptionv3=[]
mobilenetv1_image_preprocessed=[]
mobilenetv2_image_preprocessed=[]
inceptionv3_image_preprocessed=[]

for model in results:
    print(model)
    if model=='mobilenetv1':
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
