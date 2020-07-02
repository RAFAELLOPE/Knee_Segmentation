from flask import Flask, redirect, render_template, request, url_for
import os
import model
import numpy as np 


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = 'C:\\temp'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def upload():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        
        #Form parameters
        compactness = int(request.form['compactness'])
        segments = int(request.form['segments'])
        img = request.files['input_file']

        #Save image
        filename = img.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(file_path)

        img_arr = model.read_img(file_path)
        segmentation = model.segment_image(img_arr, compactness, segments)
        seg_path = file_path = os.path.join(APP_ROOT, 'static', 'segmentation.png')
        segmentation.save(seg_path)
        
        return render_template('result.html', image_path = 'segmentation.png')


if __name__ == "__main__":
    app.run(debug=True)