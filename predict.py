

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
import argparse
import json

batch_size=32
image_size=224

def process_image(image_arr):
    image_arr=tf.cast(image_arr,tf.float32)
    image_arr=tf.image.resize(image_arr,(224,224))
    image_arr /=255
    return image_arr.numpy()


def predict(image_path,model_name,top_k,class_names):
    #first we need to load the image then process it
    image = Image.open(image_path)
    processed_image = process_image(np.asarray(image))
    expanded_image = np.expand_dims(process_image(np.asarray(image)),axis=0)
#     expand_img = np.expand_dims(processed_image,axis=0)
    #predict the results
    probs= model_name.predict(expanded_image)
    result = tf.math.top_k(probs,k=top_k)
    #then map to class names
    top_classes=[]
    for i in result.indices.numpy()[0]:
        top_classes.append(class_names[str(i+1)])
    return result.values.numpy() , top_classes


if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    #add the arguments to the parser
    parser.add_argument("image_path")
    parser.add_argument("model_name")
    parser.add_argument("--top_k")
    parser.add_argument("--category_names")
    
    results = parser.parse_args()
    #load the model
    model = tf.keras.models.load_model('./{}'.format(results.model_name),custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    image_path = results.image_path
    top_k = int(results.top_k)
    if top_k is None:
        top_k=5
    category_names = results.category_names
    #open the category names mapping
    if category_names is None:
        category_names="label_map.json"
    with open(category_names,"r") as file:
        class_names=json.load(file)
        
     #then run the prediction
    probs,classes = predict(image_path,model,top_k,class_names)
    
    print(probs[0])
    print(classes)