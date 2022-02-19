import os
from unittest import result
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 512 * 512

# Replace with valid values
ENDPOINT = os.environ['ENDPOINT']
prediction_key = os.environ['PREDICTION_KEY']
project_id = os.environ['PROJECT_ID']

# Now there is a trained endpoint that can be used to make a prediction
# credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

# base_image_location = os.path.join (os.path.dirname(__file__), "Images")
publish_iteration_name = "Iteration1"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    animal_name=''
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_url=app.config['UPLOAD_FOLDER']+filename
        with open(file_url, 'rb') as image_contents:
            results=predictor.classify_image(project_id, publish_iteration_name, image_contents.read())
            for prediction in results.predictions:
                if prediction.probability * 100 > 90:
                    animal_name=prediction.tag_name
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename, animal_name=animal_name)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()