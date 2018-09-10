from flask import render_template
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import sklearn
import numpy as np
import pandas as pd
from sklearn_pandas import gen_features, DataFrameMapper

from . import dashboard
from NBC.views.models.alumni import Alumni
from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai
from NBC.views.models.user import User
from NBC.views.service.alumni_service import get_all_alumni, get_pandas_alumni


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
                                                                                            Nilai.ipk)\
                                                                                            .filter(Mahasiswa.lulus==False).all()
    return render_template('mahasiswa.html', mahasiswa=mahasiswa)


@dashboard.route('/cross_validation', methods=["GET"])
def cross_validation():
    # query all the data
    alumni = get_pandas_alumni()
    # labeling each one of the features from categorical string, to categorical integer
    feature_def = gen_features(
        columns=['school_type', 'gender', 'school-city', 'parent_salary', 'semester_1',
                                         'semester_2', 'semester_3', 'semester_4', 'ipk', 'keterangan_lulus'],
        classes=[sklearn.preprocessing.LabelEncoder]
    )
    mapper = DataFrameMapper(feature_def)
    result = mapper.fit_transform(alumni)
    # make the model
    model = MultinomialNB()
    # determine the x and y
    x = []
    y = []
    for data in result:
        x.append(data[:-1])
        y.append(data[-1])
    # calculate the cross validation
    scores = cross_val_score(model, x, y, cv=10)
    print(scores)
    return render_template('cross_validation.html', data=scores)
