from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(filename):
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def edge_mask(img, line_size, blur_value):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

    return edges

def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image
    segmented_image = segmented_image.reshape(img.shape)
    return segmented_image

def cartoonify(filename, line_size=7, blur_value=7, k=10):
    img = read_file(filename)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

    edges = edge_mask(img, line_size, blur_value)
    img = color_quantization(img, k)
    blurred = cv2.bilateralFilter(img, d=3, sigmaColor=200, sigmaSpace=200)

    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    return cartoon


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Increase k value for more colors
            cartoon_image = cartoonify(filename, k=5, line_size=5, blur_value=11)

            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'cartoon.png'), cartoon_image)

            return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'cartoon.png'), mimetype='image/png')
    # return render_template('index.html');
    return '''
    <!doctype html>
    <title>Cartoonify Image</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@100&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/index.css">
        <div class="border">
            <div class="heading">
                <h1 class="h1">IMAGE<br>CARTOONIFIER</h1>
            </div>
            <div class="input">    
                <form style="color: white" style="background-color:black" method=post enctype=multipart/form-data>
                    <input type=file name=file>
                    <input type=submit value=Upload>
                </form>
            </div>
    </div>
    '''

if __name__ == '__main__':
    app.run(debug=True)