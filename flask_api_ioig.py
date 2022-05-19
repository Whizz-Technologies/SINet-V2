from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
import os
from image_to_image_generation import predict


app = Flask(__name__)

app.secret_key = 'belatrix'

CAMO_FOLDER = 'static/uploads/camo_shape'
app.config['CAMO_FOLDER'] = CAMO_FOLDER
PATT_FOLDER = 'static/uploads/pattern'
app.config['PATT_FOLDER'] = PATT_FOLDER
BACK_FOLDER = 'static/uploads/background'
app.config['BACK_FOLDER'] = BACK_FOLDER
OVER_FOLDER = 'static/uploads/overlayed'
app.config['OVER_FOLDER'] = OVER_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'webp', 'jfif', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/home_image')
def home3():
    return render_template('index3.html')


@app.route('/image_over', methods=['POST'])
def get_image_over_image():
    path1='./static/uploads/camo_shape/'
    path2='./static/uploads/pattern/'
    path3='./static/uploads/background/'
    for v in os.listdir(path1):
        os.remove(os.path.join(path1, v))
    for s in os.listdir(path2):
        os.remove(os.path.join(path2, s))
    for t in os.listdir(path3):
        os.remove(os.path.join(path3, t))
    if 'file1' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file1 = request.files['file1']
    if file1 and allowed_file(file1.filename):
        filename1 = secure_filename(file1.filename)
        file1.save(os.path.join(app.config['CAMO_FOLDER'], filename1))
    if 'file2' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file2 = request.files['file2']
    if file2 and allowed_file(file2.filename):
        filename2 = secure_filename(file2.filename)
        file2.save(os.path.join(app.config['PATT_FOLDER'], filename2))
    if 'file3' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file3 = request.files['file3']
    if file3 and allowed_file(file3.filename):
        filename3 = secure_filename(file3.filename)
        file3.save(os.path.join(app.config['BACK_FOLDER'], filename3))
    scale_factor = request.form.get("num")
    scale = int(scale_factor)
    center, background_size , background, img, pattern_path = predict(path1, path2, path3, scale)
    return render_template('center_value.html', cen=center, back=background_size)



if __name__=='__main__':
    app.run(debug=True)
