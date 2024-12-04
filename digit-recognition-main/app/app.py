#pip install gradio
import gradio as gr
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
mist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mist.load_data() # Load data
x_train = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
model = tf.keras.models.Sequential()
#aading the input to maodel
model.add(tf.keras.layers.Flatten())
#Bulid the input and the hidden layers
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#bulid the output layer
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model_log = model.fit(x=x_train,y =y_train, batch_size=60,verbose=1,epochs=5,validation_split=.3)
x_test = x_test[0].reshape(-1,28,28)
preditions=model.predict(x_test)
def predict_image(img):
    img_3d = img.reshape(-1,28,28)
    im_resize=img_3d/255.0
    prediction = model.predict(im_resize)
    predictions = np.argmax(prediction)
    return predictions
iface = gr.Interface(predict_image,inputs="sketchpad", outputs="label")
iface.launch(debug='True')