import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained VGG16 model
model = load_model("Sports_vgg16.h5",compile=False)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


index = ["air hockey", "ampute football", "archery", "arm wrestling", "axe throwing", "balance beam",
         "barrel racing", "baseball", "basketball", "baton twirling", "bike polo", "billiards", "bmx", "bobsled",
         "bowling", "boxing", "bull riding", "bungee jumping", "canoe slalom", "cheerleading", "chuckwagon racing",
         "cricket", "croquet", "curling", "disc golf", "fencing", "field hockey", "figure skating men", "figure skating pairs",
         "figure skating women", "fly fishing", "football", "formula 1 racing", "frisbee", "gaga", "giant slalom", "golf",
         "hammer throw", "hang gliding", "harness racing", "high jump", "hockey", "horse jumping", "horse racing",
         "horseshoe pitching", "hurdles", "hydroplane racing", "ice climbing", "ice yachting", "jai alai", "javelin", "jousting",
         "judo", "lacrosse", "log rolling", "luge", "motorcycle racing", "mushing", "nascar racing", "olympic wrestling",
         "parallel bar", "pole climbing", "pole dancing", "pole vault", "polo", "pommel horse", "rings", "rock climbing",
         "roller derby", "rollerblade racing", "rowing", "rugby", "sailboat racing", "shot put", "shuffleboard", "sidecar racing",
         "ski jumping", "sky surfing", "skydiving", "snowboarding", "snowmobile racing", "speed skating", "steer wrestling",
         "sumo wrestling", "surfing", "swimming", "table tennis", "tennis", "track bicycle", "trapeze", "tug of war",
         "ultimate", "uneven bars", "volleyball", "water cycling", "water polo", "weightlifting", "wheelchair basketball",
         "wheelchair racing", "wingsuit flying"]
# Define a function to preprocess the image for the VGG16 model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img=img/255.0
    return img

# Create a route to upload an image
@app.route('/')
def upload_image():
    return render_template('index.html')

# Create a route to handle image upload and classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(request.url)

    image_file = request.files['image']

    if image_file.filename == '':
        return redirect(request.url)

    if image_file:
        image_path = os.path.join('static/upload', image_file.filename)
        image_file.save(image_path)
        img = preprocess_image(image_path)
        pred = model.predict(img)
        pred_class=np.argmax(pred,axis=1)
        pred_sport=index[pred_class[0]]
        

        return render_template('index.html', image_path=image_path, predictions=pred_sport)

if __name__ == '__main__':
    app.run(debug=True)
