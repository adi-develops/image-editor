import os
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2

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
                            newFileName = filename
                            cv2.imwrite(f"static/{newFileName}", flippedImage)
                            return newFileName
                        case "horizontal-flip":
                            print("Horizontal Flip invoked")
                            flippedImage = cv2.flip(img, 1)
                            newFileName = filename
                            cv2.imwrite(f"static/{newFileName}", flippedImage)
                            return newFileName
                        case "both-flip":
                            flippedImage = cv2.flip(img, -1)
                            newFileName = filename
                            cv2.imwrite(f"static/{newFileName}", flippedImage)
                            return newFileName
        case "filter":
            match operation["filterType"]:
                case "cgray":
                    imageProcessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    newFileName = filename
                    cv2.imwrite(f"static/{newFileName}", imageProcessed)
                    return newFileName
        
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