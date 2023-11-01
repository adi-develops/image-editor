import os
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'webp', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def processImage(filename, operation):
    print(f"File name is {filename} and operation is {operation['operation']}")
    img = cv2.imread(f"uploads/{filename}")
    print(operation)
    match operation["operation"]:
        case "transform":
            match operation["transformType"]:
                case "flip":
                    match operation["flipType"]:
                        case "vertical-flip":
                            flippedImage = cv2.flip(img, 0)
                            cv2.imwrite(f"static/{filename}", flippedImage)
                            return newFileName
                        case "horizontal-flip":
                            flippedImage = cv2.flip(img, 1)
                            cv2.imwrite(f"static/{filename}", flippedImage)
                            return newFileName
                        case "both-flip":
                            flippedImage = cv2.flip(img, -1)
                            cv2.imwrite(f"static/{filename}", flippedImage)
                            return newFileName
                        
                case "rotate":
                    angle = int(operation["rotateAngle"])
                    (height, width) = img.shape[:2]
                    rotPoint = (width//2, height//2)
                    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
                    dimensions = (width, height)
                    rotatedImage = cv2.warpAffine(img, rotMat, dimensions)
                    cv2.imwrite(f"static/{filename}", rotatedImage)
                    return filename
                
                case "resize":
                    new_width = int(img.shape[1] * float(operation["scale_width"]))
                    new_height = int(img.shape[0] * float(operation["scale_height"]))
                    upscaled_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f"static/{filename}", upscaled_image)
                    return filename

        case "filter":
            match operation["filterType"]:
                case "cgray":
                    imageProcessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(f"static/{filename}", imageProcessed)
                    return newFileName
                case "blur":
                    kernel_size = 7
                    b, g, r = cv2.split(img)
                    b = cv2.medianBlur(b, kernel_size)
                    g = cv2.medianBlur(g, kernel_size)
                    r = cv2.medianBlur(r, kernel_size)
                    watercolor = cv2.merge((b, g, r))
                    cv2.imwrite(f"static/{filename}", watercolor)
                    return filename
                case "red":
                    b, g, r = cv2.split(img)
                    blank = np.zeros_like(b)
                    red_channel = cv2.merge([blank, blank, r])
                    cv2.imwrite(f"static/{filename}", red_channel)
                    return filename
                case "blue":
                    b, g, r = cv2.split(img)
                    blank = np.zeros_like(b)
                    blue_channel = cv2.merge([b, blank, blank])
                    cv2.imwrite(f"static/{filename}", blue_channel)
                    return filename
                case "green":
                    b, g, r = cv2.split(img)
                    blank = np.zeros_like(b)
                    green_channel = cv2.merge([blank, g, blank])
                    cv2.imwrite(f"static/{filename}", green_channel)
                    return filename
        
        case "convert":
            match operation["convertType"]:
                case "cwebp":
                    newFileName = f"{filename.split('.')[0]}.webp"
                    cv2.imwrite(f"static/{newFileName}", img)
                    return newFileName
                case "cjpg":
                    newFileName = f"{filename.split('.')[0]}.jpg"
                    cv2.imwrite(f"static/{newFileName}", img)
                    return newFileName
                case "cpng":
                    newFileName = f"{filename.split('.')[0]}.png"
                    cv2.imwrite(f"static/{newFileName}", img)
                    return newFileName


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/edit", methods=["GET", "POST"])
def edit():
    if request.method == "POST":
        operation = request.form
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new = processImage(filename, operation)
            flash(new)
            return redirect(url_for('home'))

@app.route('/download/<filename>')
def download_file(filename):
    directory = 'static'
    return send_from_directory(directory, filename, as_attachment=True)

app.run(debug=True, port=8000)