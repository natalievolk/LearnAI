from flask import Flask, request, jsonify
from app.torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg','jpeg'}
CATEGORIES = {0:'Recreational Activities', 1: "Self-help", 2: "STEM", 3:"Social Science", 4:"Fiction"}

def allowed_file(filename):
    return'.' in filename and filename.rsplit('.',1) [1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify ({'error': 'no file'})
        elif not allowed_file(file.filename):
            return jsonify ({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name': CATEGORIES.get(prediction.item())}
            return jsonify(data)

        except:
            return jsonify ({'error': 'error during prediction'})

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True)
