import os

import pandas as pd
import matplotlib
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from NBC.config import Config
from NBC.models.alumni import Alumni
from NBC.models.testing import Testing
from NBC.models.training import Training
from NBC.models.user import User

matplotlib.use('agg')

import matplotlib.pyplot as plt


def grouping_school_city(data):
    if 'yogya' in data.lower():
        return 'Dalam Kota'
    else:
        return 'Luar Kota'


def grouping_school_type(data):
    if 'smk' in data.lower():
        return 'SMK'
    elif 'sma' in data.lower():
        return 'SMA'
    else:
        return 'Lain'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSION


def get_data_length():
    len_dict = {
        'len_alumni': len(Alumni.query.all()),
        'len_users': len(User.query.all()),
        'len_training': len(Training.query.all()),
        'len_testing': len(Testing.query.all())
    }
    return len_dict


def create_bar_chart(output_path, f1, recall, precision, scores):
    # plot the k fold
    fig, ax = plt.subplots()
    bar_width = 0.2
    N = 10
    ind = np.arange(N)
    rect_f1 = ax.bar(ind - 2 * bar_width, f1, bar_width, color='r', label='f1-score')
    rect_recall = ax.bar(ind - bar_width, recall, bar_width, color='g', label='recall')
    rect_precision = ax.bar(ind, precision, bar_width, color='b', label='precision')
    rect_score = ax.bar(ind + bar_width, scores, bar_width, color='y', label='accuracy')
    ax.set_xlabel('Folds')
    ax.set_ylabel('Scores')
    ax.set_title('10 Fold Cross Validation')
    ax.set_xticks(ind + bar_width / 2)
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    ax.legend()
    ax.set_ylim([0, 1])
    fig.set_size_inches(15, 10)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.clf()


def create_pie_chart(path, labels, sizes):
    filename = 'piechart.png'
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(os.path.join(path, filename))
    plt.clf()


def check_upload_file(request):
    if 'file' not in request.files:
        return False
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return False
    if file and allowed_file(file.filename):
        return file
    else:
        return False


def predict_data(data_train, data_target, index=False):
    if index:
        target = pd.DataFrame(data_target, index=[0])
    else:
        target = pd.DataFrame(data_target)
    # convert class into number
    data_train['ket_lulus'] = data_train['ket_lulus'].replace(['Tidak Tepat Waktu', 'Tepat Waktu'], [0, 1])
    # concat train and target dataframe for encoding
    df_concat = pd.concat([data_train, target.drop('id', axis=1)], keys=['train', 'test'],
                          sort=True).fillna(0)
    # convert into int
    df_concat['ket_lulus'] = df_concat['ket_lulus'].astype(int)
    # One Hot Encoding
    enc = pd.get_dummies(df_concat)
    # Split X, Y, and Target after encoding
    x = enc.loc['train'].drop('ket_lulus', axis=1)
    y = enc.loc['train']['ket_lulus']
    target = enc.loc['test'].drop('ket_lulus', axis=1)
    model = MultinomialNB()
    model.fit(x, y)
    return model.predict(target)
