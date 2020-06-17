from flask import Flask, render_template, Response,url_for, jsonify, request, redirect, json,flash, session
import os
import face_retrieval
from werkzeug.utils import secure_filename
from vptree import VPTree
import pickle
from scipy.spatial.distance import cosine

def cosine_similarity(p1, p2):
    return cosine(p1, p2)
pickle_in_dic = open('embeded_face_train_resnet50_vptree_new.pickle', 'rb')
dic = pickle.load(pickle_in_dic)
pickle_in_dic.close()

app = Flask(__name__)
curr_dir = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(*[curr_dir, 'static', 'images', 'upload'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/name', methods=['GET', 'POST'])
def upload_image():
    file_list = []
    if request.method == 'POST':
        file = request.files['image']
        count = int(request.form['count'])

        if file.filename == '':
            return redirect('/')

        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_embedding = face_retrieval.get_embeddings(filenames=[os.path.join(app.config['UPLOAD_FOLDER'], filename)], need_to_extract=True)
        
        result = dic['tree'].get_n_nearest_neighbors(img_embedding[0], count)

        file_index_list = []
        for each_result in result:
            file_index_list.append(each_result[2])
        for file_index in file_index_list:
            file_list.append(dic['filenames'][file_index])

    return render_template('nameidol.html', file_list=file_list)

if __name__ == '__main__':
    app.run(debug=1, threaded=False)
    