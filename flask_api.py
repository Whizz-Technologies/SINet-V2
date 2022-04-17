from flask import Flask, request, flash, redirect, render_template, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict


app = Flask(__name__)

app.secret_key = 'belatrix'

SAVED_FOLDER = 'static'
app.config['SAVED_FOLDER'] = SAVED_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'webp', 'jfif', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/image', methods=['POST'])
def get_object_detection():
    path = './static/'
    for v in os.listdir(path):
        os.remove(os.path.join(path, v))
    if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['SAVED_FOLDER'], filename))
        path_image = SAVED_FOLDER + '/' + filename
        print(path_image)
        predict(path_image)
        st = './static/output.jpg'
        ts = './static/output_contour.jpg'
        # print(st)
        return render_template('index.html', filename=st, filename2=ts)
        
@app.route('/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='./' + filename), code=301)

@app.route('/<filename2>')
def display_image2(filename2):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='./' + filename2), code=301)



if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)
