import numpy as np
from keras.preprocessing import image
from keras.layers import *
from keras.models import load_model
classes=['Cat','Dog','Wild']
loaded_model = load_model('animalfaces_final.h5')
image = tf.keras.utils.load_img('img2.jpeg',target_size=(64,64))
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.   
predictions = loaded_model.predict(input_arr)
classes_x=np.argmax(predictions,axis=1)
print(f"Its a {classes[classes_x[0]]}!")