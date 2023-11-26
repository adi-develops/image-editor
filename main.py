import os
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import face_recognition


aryaman_image = face_recognition.load_image_file("face_assets/aryaman.jpg")
aryaman_face_encoding = face_recognition.face_encodings(aryaman_image)[0]

tom_image = face_recognition.load_image_file("face_assets/tommy2.jpg")
tom_face_encoding = face_recognition.face_encodings(tom_image)[0]

known_face_encodings = [
    aryaman_face_encoding,
    tom_face_encoding
]

known_face_names = [
    "Aryaman",
    "Tom"
]

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
                            return filename
                        case "horizontal-flip":
                            flippedImage = cv2.flip(img, 1)
                            cv2.imwrite(f"static/{filename}", flippedImage)
                            return filename
                        case "both-flip":
                            flippedImage = cv2.flip(img, -1)
                            cv2.imwrite(f"static/{filename}", flippedImage)
                            return filename
                        
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
                
                case "compress":
                    compression_quality = int(operation["imageQuality"])
                    cv2.imwrite(f"static/{filename}", img, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
                    return filename
                


        case "filter":
            match operation["filterType"]:
                case "cgray":
                    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                    # imageProcessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(f"static/{filename}", gray)
                    return filename
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
                
        case "recognize":
            face_locations = []
            face_encodings = []
            face_names = []

            # Resize the input image
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            # Face recognition on the input image
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            print(known_face_names)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

                # Draw rectangles and display the image
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    
            cv2.imwrite(f"static/{filename}", img)
            return filename
        


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/instructions")
def instructions():
    return render_template("instructions.html")

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