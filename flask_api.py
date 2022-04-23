from flask import Flask, request, flash, redirect, render_template, url_for
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from predict import predict


app = Flask(__name__)

app.secret_key = 'belatrix'

SAVED_FOLDER = 'static/uploads/input'
app.config['SAVED_FOLDER'] = SAVED_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'webp', 'jfif', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/image', methods=['POST'])
def get_object_detection():
    path1 = './static/uploads/input/'
    path2 = './static/uploads/output/'
    for v in os.listdir(path1):
        os.remove(os.path.join(path1, v))
    for s in os.listdir(path2):
        os.remove(os.path.join(path2, s))
    if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['SAVED_FOLDER'], filename))
        # print(path_image)
    path_image = './static/uploads'
    predict(path_image)
    input_folder = './static/uploads/input'
    output_folder = './static/uploads/output'
    currentDateAndTime = datetime.now()
    final_output_folder_name = './output/' + currentDateAndTime.strftime('%Y-%m-%d_%H-%M-%S') +'/'
    os.mkdir(final_output_folder_name)

    for file in os.listdir(output_folder):
        print(os.path.join(output_folder, file))
        print(os.path.join(final_output_folder_name, file))
        src_path = os.path.join(output_folder, file)
        dst_path = os.path.join(final_output_folder_name, file)
        cmd = "cp " + src_path + " " + dst_path
        print(cmd)
        os.system(cmd)

    for file in os.listdir(input_folder):
        print(os.path.join(input_folder, file))
        print(os.path.join(final_output_folder_name, file))
        src_path = os.path.join(input_folder, file)
        dst_path = os.path.join(final_output_folder_name, file)
        cmd = "cp " + src_path + " " + dst_path
        print(cmd)
        os.system(cmd)

    st = './static/uploads/output/final_image.jpg'
        # ts = './static/output_contour.jpg'
        # print(st)
    return render_template('index.html', filename=st)
        
@app.route('/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='./' + filename), code=301)


if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)
