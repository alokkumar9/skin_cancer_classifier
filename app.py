# import system libs
import os
import time
import shutil
import itertools

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import gradio as gr

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split


# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
print ('modules loaded')
#---Training-----------------------------
#  ! pip install -q kaggle
# from google.colab import files

# files.upload()
# ! mkdir ~/.kaggle

# ! cp kaggle.json ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json
# ! kaggle datasets list
# !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
# ! mkdir kaggle
# ! unzip skin-cancer-mnist-ham10000.zip -d kaggle
# data_dir = '/content/kaggle/hmnist_28_28_RGB.csv'
# data = pd.read_csv(data_dir)
# print(data.shape)
# data.head()

# Label = data["label"]
# Data = data.drop(columns=["label"])
# print(data.shape)
# Data.head()

# from imblearn.over_sampling import RandomOverSampler

# oversample = RandomOverSampler()
# Data, Label  = oversample.fit_resample(Data, Label)
# print(Data.shape)
# Data = np.array(Data).reshape(-1,28, 28,3)
# print('Shape of Data :', Data.shape)

# Label = np.array(Label)
# Label
# classes = {4: ('nv', ' melanocytic nevi'),
#            6: ('mel', 'melanoma'),
#            2 :('bkl', 'benign keratosis-like lesions'),
#            1:('bcc' , ' basal cell carcinoma'),
#            5: ('vasc', ' pyogenic granulomas and hemorrhage'),
#            0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
#            3: ('df', 'dermatofibroma')}



# X_train , X_test , y_train , y_test = train_test_split(Data , Label , test_size = 0.25 , random_state = 49)

# print(f'X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}')
# print(f'y_train shape: {y_train.shape}\ny_test shape: {y_test.shape}')

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# datagen = ImageDataGenerator(rescale=(1./255)
#                              ,rotation_range=10
#                              ,zoom_range = 0.1
#                              ,width_shift_range=0.1
#                              ,height_shift_range=0.1)

# testgen = ImageDataGenerator(rescale=(1./255))

# from keras.callbacks import ReduceLROnPlateau

# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
#                                             , patience = 2
#                                             , verbose=1
#                                             ,factor=0.5
#                                             , min_lr=0.00001)

# model = keras.models.Sequential()

# # Create Model Structure
# model.add(keras.layers.Input(shape=[28, 28, 3]))
# model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.MaxPooling2D())
# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.MaxPooling2D())
# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.MaxPooling2D())
# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.MaxPooling2D())

# model.add(keras.layers.Flatten())

# model.add(keras.layers.Dropout(rate=0.2))
# model.add(keras.layers.Dense(units=256, activation='relu', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Dense(units=32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.L1L2()))
# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Dense(units=7, activation='softmax', kernel_initializer='glorot_uniform', name='classifier'))

# model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

# model.summary()

# history = model.fit(X_train ,
#                     y_train ,
#                     epochs=25 ,
#                     batch_size=128,
#                     validation_data=(X_test , y_test) ,
#                     callbacks=[learning_rate_reduction])

# def plot_training(hist):
#     tr_acc = hist.history['accuracy']
#     tr_loss = hist.history['loss']
#     val_acc = hist.history['val_accuracy']
#     val_loss = hist.history['val_loss']
#     index_loss = np.argmin(val_loss)
#     val_lowest = val_loss[index_loss]
#     index_acc = np.argmax(val_acc)
#     acc_highest = val_acc[index_acc]

#     plt.figure(figsize= (20, 8))
#     plt.style.use('fivethirtyeight')
#     Epochs = [i+1 for i in range(len(tr_acc))]
#     loss_label = f'best epoch= {str(index_loss + 1)}'
#     acc_label = f'best epoch= {str(index_acc + 1)}'

#     plt.subplot(1, 2, 1)
#     plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
#     plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
#     plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
#     plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
#     plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.tight_layout
#     plt.show()

#     plot_training(history)

#     train_score = model.evaluate(X_train, y_train, verbose= 1)
# test_score = model.evaluate(X_test, y_test, verbose= 1)

# print("Train Loss: ", train_score[0])
# print("Train Accuracy: ", train_score[1])
# print('-' * 20)
# print("Test Loss: ", test_score[0])
# print("Test Accuracy: ", test_score[1])

# y_true = np.array(y_test)
# y_pred = model.predict(X_test)

# y_pred = np.argmax(y_pred , axis=1)
# y_true = np.argmax(y_true , axis=1)

# classes_labels = []
# for key in classes.keys():
#     classes_labels.append(key)

# print(classes_labels)

# # Confusion matrix
# cm = cm = confusion_matrix(y_true, y_pred, labels=classes_labels)

# plt.figure(figsize= (10, 10))
# plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()

# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation= 45)
# plt.yticks(tick_marks, classes)


# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

# plt.tight_layout()
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')

# plt.show()

# #Save the model
# model.save('skin_cancer_model.h5')

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# print("model converted")

# # Save the model.
# with open('skin_cancer_model.tflite', 'wb') as f:
#     f.write(tflite_model)

#Training End------------------------------------------

skin_classes = {4: ('nv', ' melanocytic nevi'),
           6: ('mel', 'melanoma'),
           2 :('bkl', 'benign keratosis-like lesions'), 
           1:('bcc' , ' basal cell carcinoma'),
           5: ('vasc', ' pyogenic granulomas and hemorrhage'),
           0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
           3: ('df', 'dermatofibroma')}

#Use saved model
loaded_model = tf.keras.models.load_model('skin_cancer_model.h5', compile=False)
loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

def predict_digit(image):
    if image is not None:
        
        #Use saved model
        loaded_model = tf.keras.models.load_model('skin_cancer_model.h5', compile=False)
        loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
        img = image.resize((28, 28))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        print(img_array)


        predictions = loaded_model.predict(img_array)
        print(predictions)
        #class_labels = [] # data classes
        score = tf.nn.softmax(predictions[0])*100


        print(score)
        print(skin_classes[np.argmax(score)])
        simple = pd.DataFrame(
        {
        "skin condition": ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"],
        "probability": score, 
        "full skin condition": [ 'Actinic keratoses', 
              ' basal cell carcinoma',
              'benign keratosis-like lesions',
              'dermatofibroma',
              ' melanocytic nevi',
              ' pyogenic granulomas and hemorrhage',
              'melanoma'],
        }
        )




        predicted_skin_condition=skin_classes[np.argmax(score)][1]+"   ("+ skin_classes[np.argmax(score)][0]+")"
        return  predicted_skin_condition, gr.BarPlot(
            simple,
            x="skin condition",
            y="probability",
            x_title="Skin Condition",
            y_title="Classification Probabilities",
            title="Skin Cancer Classification Probability",
            tooltip=["full skin condition", "probability"],
            vertical=False,
            y_lim=[0, 100],
            color="full skin condition"
        )
        
    else:
        simple_empty = pd.DataFrame(
        {
        "skin condition": ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"],
        "probability": [0,0,0,0,0,0,0],
        "full skin condition": [ 'Actinic keratoses', 
              ' basal cell carcinoma',
              'benign keratosis-like lesions',
              'dermatofibroma',
              ' melanocytic nevi',
              ' pyogenic granulomas and hemorrhage',
              'melanoma'],
        }
        )

        return " ", gr.BarPlot(
            simple_empty,
            x="skin condition",
            y="probability",
            x_title="Digits",
            y_title="Identification Probabilities",
            title="Identification Probability",
            tooltip=["full skin condition", "probability"],
            vertical=False,
            y_lim=[0, 100],
            
        )
    
skin_images = [
    ("skin_image/mel.jpg",'mel'),
    ("skin_image/nv3.jpg",'nv'),
    ("skin_image/bkl.jpg",'bkl'),
    ("skin_image/df.jpg",'df'),
    ("skin_image/akiec.jpg",'akiec'),
    ("skin_image/bcc.jpg",'bcc'),
    ("skin_image/vasc.jpg",'vasc'),
    ("skin_image/nv2.jpg",'nv'),
    ("skin_image/akiec2.jpg",'akiec'),
    ("skin_image/bkl2.jpg",'bkl'),
    ("skin_image/nv.jpg",'nv'),
    
    ] 

def image_from_gallary(evt: gr.SelectData):
    print(evt.index)
    return skin_images[evt.index][0]



css='''
#title_head{
text-align: center;
text-weight: bold;
text-size:30px;
}
#name_head{
text-align: center;
}
'''

with gr.Blocks(css=css) as demo:
   
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h1>Skin Cancer Classifier</h1>", elem_id='title_head')
            gr.Markdown("<h2>By Alok</h2>", elem_id="name_head")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr.Markdown("<h3>Browse or Select from given Image</h3>", elem_id='info')
                img_upload=gr.Image(type="pil", height=200, width=300)
            with gr.Row():
                clear=gr.ClearButton(img_upload)
                btn=gr.Button("Identify")
                    
        with gr.Column():
            gry=gr.Gallery(value=skin_images, columns=5, show_label=False, allow_preview=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("Most probable skin condition")
            label=gr.Label("")
    with gr.Row():
        with gr.Column():
            gr.Markdown("Other possible values")
            bar = gr.BarPlot()
    
    
            
    btn.click(predict_digit,inputs=[img_upload],outputs=[label,bar])
    gry.select(image_from_gallary, outputs=img_upload)

           
        
    
demo.launch(debug=True)

    

