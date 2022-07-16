from flask import Flask, request, flash, redirect, render_template, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict
from scraper import search
from image_to_image_generation import predict_ioig, overlay
from batch_image_infer_run import run
import cv2
from PIL import Image

app = Flask(__name__)

app.secret_key = 'belatrix'

SAVED_FOLDER = 'static/uploads/input'
app.config['SAVED_FOLDER'] = SAVED_FOLDER
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def home2():
    return render_template('index2.html')

@app.route('/home_image')
def home3():
    return render_template('index3.html')

@app.route('/h_kml')
def home4():
    return render_template('index4.html')


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

    st = './static/uploads/output/final_image.jpg'
        # ts = './static/output_contour.jpg'
        # print(st)
    return render_template('index.html', filename=st)
        
@app.route('/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='./' + filename), code=301)

@app.route('/down_img', methods=['POST'])
def get_flickr_img():
    text = request.form.get("quantity")
    percent = request.form.get("num")
    term = str(text)
    max = int(percent)
    search(qs=term, qg=None, max_pages=max)
    flash('Images downloaded!')
    return render_template('index2.html')

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
    global scale
    scale = float(scale_factor)
    center, background_size , background, img, pattern_path = predict_ioig(path1, path2, path3, scale)
    cv2.imwrite('./static/uploads/filled_contour.jpg', img)
    return render_template('center_value.html', cen=center, back=background_size)

@app.route('/centre_value', methods=['POST'])
def get_centre_value():
    background_path='./static/uploads/background/'
    pattern_path='./static/uploads/pattern/'
    overlayed_path = './static/uploads/overlayed/'
    image = cv2.imread('./static/uploads/filled_contour.jpg')
    if os.listdir(background_path):
        print("Background Image Found")
        background_path = background_path + os.listdir(background_path)[0]
        background = Image.open(background_path).convert("RGBA")
    if os.listdir(pattern_path):
        print("Pattern Image Found")
        pattern_path = pattern_path + os.listdir(pattern_path)[0]  
    x_val = request.form.get("x_value")
    y_val = request.form.get("y_value")
    if x_val == '' or y_val == '':
        x_val = 30
        y_val = 30
        # ts = './static/output_contour.jpg'
        # print(st)
    overlayed_image = overlay(background, scale, pattern_path, image, int(x_val), int(y_val))
    overlayed_image = overlayed_image.convert('RGB')
    overlayed_image.save(overlayed_path + 'overlayed.jpg')
    st = './static/uploads/overlayed/overlayed.jpg'
    return render_template('over_image.html', pic=st)


@app.route('/<pic>')
def display_image2(pic):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', pic='./uploads/overlayed/' + pic), code=301)


@app.route('/kml', methods=['POST'])
def get_kml():
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
    
    number_img = request.form.get("num_img")
    fold_name = request.form.get("folder_name")
    folder_name = str(fold_name)
    num_img = int(number_img)
    print(num_img)
    if num_img < 10:
        num_img = 10
    else:
        num_img == num_img
    print (num_img)
    value, picture = run(num_img, folder_name, path1, path2, path3)
    cv2.imwrite('./static/uploads/' + 'combined' + '.jpg', picture)
    cv2.waitKey(1)
    stt = './static/uploads/combined.jpg'
    return render_template('index4.html', value=value, click=stt)

@app.route('/<click>')
def display_image3(click):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', click='./uploads/' + click), code=301)


if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)
