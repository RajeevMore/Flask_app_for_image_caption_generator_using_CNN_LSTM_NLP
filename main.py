import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from caption_generator_module import *


# libraries for model 


import pandas 
import tensorflow
from tensorflow.keras.models import load_model
import cv2
import string
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout



from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
# Model saved with Keras model.save()
MODEL_PATH ='model_caption.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#model predict 
def model_predict(file_path, model):
    #path = 'Flickr8k_Dataset/Flicker8k_Dataset/111537222_07e56d5a30.jpg'
    max_length = 32
    tokenizer = load(open("tokenizer.pkl","rb"))
    model = load_model('model_caption.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    photo =extract_features(file_path, xception_model)
    img = Image.open(file_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    return (description)
    #plt.imshow(img)
    
  
@app.route('/')
def upload_form():
    return render_template('input_image.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    # Save file in upload_folder
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    
        # Make prediction
        result = model_predict(file_path, model)
        print(result)
        flash(result)
        return render_template('input_image.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename)) #status code

if __name__ == "__main__":
    app.run(debug=True)
