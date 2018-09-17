from flask import render_template
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import csv

from . import dashboard
from NBC.views.models.alumni import Alumni
from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai
from NBC.views.models.user import User
from NBC.views.service.alumni_service import get_pandas_alumni


@dashboard.route('/')
def index():
    return render_template('dashboard.html')


@dashboard.route('/users', methods=["GET", "POST"])
def users():
    users = User.query.all()
    return render_template('users.html', users=users)


@dashboard.route('/alumni', methods=["GET"])
def alumni():
    alumni = Alumni.query.join(Mahasiswa, Alumni.id_mahasiswa == Mahasiswa.id).join(Nilai,
                                                                                    Mahasiswa.id == Nilai.id_mahasiswa) \
        .add_columns(Alumni.id_mahasiswa, Mahasiswa.name, Mahasiswa.school_type, Mahasiswa.gender,
                     Mahasiswa.school_city,
                     Mahasiswa.parent_salary, Nilai.semester_1, Nilai.semester_2, Nilai.semester_3,
                     Nilai.semester_4, Nilai.ipk, Alumni.keterangan_lulus) \
        .all()
    return render_template('alumni.html', alumni=alumni)


@dashboard.route('/mahasiswa', methods=["GET", "POST"])
def mahasiswa():
    mahasiswa = Mahasiswa.query.join(Nilai, Mahasiswa.id == Nilai.id_mahasiswa).add_columns(Mahasiswa.id,
                                                                                            Mahasiswa.name,
                                                                                            Mahasiswa.school_type,
                                                                                            Mahasiswa.gender,
                                                                                            Mahasiswa.school_city,
                                                                                            Mahasiswa.parent_salary,
                                                                                            Mahasiswa.lulus,
                                                                                            Nilai.semester_1,
                                                                                            Nilai.semester_2,
                                                                                            Nilai.semester_3,
                                                                                            Nilai.semester_4,
                                                                                            Nilai.ipk) \
        .filter(Mahasiswa.lulus == False).all()
    return render_template('mahasiswa.html', mahasiswa=mahasiswa)


@dashboard.route('/cross_validation', methods=["GET"])
def cross_validation():
    # query all the data
    alumni = get_pandas_alumni()
    # one hot encoder
    enc = pd.get_dummies(alumni)
    x = np.array(enc.drop('keterangan_lulus', axis=1))
    y = np.array(enc['keterangan_lulus'])
    # create the model
    model = MultinomialNB()
    # cross validation
    sf = StratifiedKFold(n_splits=5)
    cf = np.array([[0, 0], [0, 0]])
    scores = []
    for train_index, test_index in sf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        cf += confusion_matrix(y_test, y_pred)
        score = model.score(x_test, y_test)
        scores.append(score)
    # loop above is same as cv = cross_val_score(model, x, y, cv=5)
    return render_template('cross_validation.html', scores=scores, cf=cf)


@dashboard.route('/predict', methods=["GET"])
def predict():
    # query all the data
    alumni = get_pandas_alumni()
    # catch the data
    new_data = pd.DataFrame({
        'school_type': ['SMA'],
        'gender': ['Laki-Laki'],
        'school-city': ['Dalam Kota'],
        'parent_salary': ['Rendah'],
        'semester_1': ['B'],
        'semester_2': ['B'],
        'semester_3': ['B'],
        'semester_4': ['B'],
        'ipk': ['B']
    })

    # concat the data
    con = pd.concat([alumni, new_data], keys=['train', 'test'], sort=True).fillna(0)
    # one hot encoder, axis 1 mean column
    enc = pd.get_dummies(con).astype(int)
    x = np.array(enc.loc['train'].drop('keterangan_lulus', axis=1))
    y = np.array(enc.loc['train']['keterangan_lulus'])
    pred_data = np.array(enc.loc['test'].drop('keterangan_lulus', axis=1))
    # create the model
    model = MultinomialNB()
    # predict the 10th data from dataset for testing purpose
    model.fit(x, y)
    pred = model.predict(pred_data)
    pred_prob = model.predict_proba(pred_data)
    return render_template('prediction.html', predict=pred, predict_proba=pred_prob)


@dashboard.route('/predict_golf', methods=["GET"])
def pgolf():
    with open('golf_dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = []
        for line in reader:
            for l in line:
                data.append(l.split(','))
        data = np.array(data)
        df = pd.DataFrame(data, columns=['outlook', 'temperature', 'humidity', 'windy', 'play'])
        df['play'] = df['play'].astype(int)
        new_data = pd.DataFrame({
            'outlook': ['sunny'],
            'temperature': ['cool'],
            'humidity': ['high'],
            'windy': ['TRUE']
        })
        df2 = pd.concat([df, new_data], keys=['train', 'test'], sort=True).fillna(0)
        enc = pd.get_dummies(df2).astype(int)
        model = MultinomialNB()
        x = enc.loc['train'].drop('play', axis=1)
        y = enc.loc['train']['play']
        pred_data = enc.loc['test'].drop('play', axis=1)
        model.fit(x, y)
        predict = model.predict(pred_data)
        predict_proba = model.predict_proba(pred_data)
    return render_template('prediction.html', predict=predict, predict_proba=predict_proba)
