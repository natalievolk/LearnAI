from flask import Flask, jsonify, flash, request, redirect, url_for, render_template
from torch_utils import transform_image, get_prediction
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg','jpeg'}
UPLOAD_FOLDER = os.getcwd()+'/covers'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CATEGORIES = {0:'Recreational Activities', 1: "Self-help", 2: "STEM", 3:"Social Science", 4:"Fiction"}

def allowed_file(filename):
    return'.' in filename and filename.rsplit('.',1) [1].lower() in ALLOWED_EXTENSIONS

def make_prediction():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify ({'error': 'no file'})
        elif not allowed_file(file.filename):
            return jsonify ({'error': 'format not supported'})

        # try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        prediction = get_prediction(tensor)
        data = {'prediction': prediction.item(), 'class_name': CATEGORIES.get(prediction.item())}
        return jsonify(data)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',
                                  #  filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    #render_template("index.html")

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
