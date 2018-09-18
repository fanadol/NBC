import os
import numpy as np
import pandas as pd
import csv
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
from flask import render_template, request, redirect, url_for, flash
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from scipy import interp

from . import dashboard
from NBC.views.models.alumni import Alumni
from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai
from NBC.views.models.user import User
from NBC.views.models.testing import Testing
from NBC.views.models.training import Training
from NBC.views.service.alumni_service import get_pandas_alumni
from NBC.views.service.mahasiswa_service import get_pandas_mahasiswa, get_all_mahasiswa
from NBC.views.service.prediction_service import save_to_db, get_all_testing, get_all_testing_result, get_an_id
from NBC.views.service.training_service import get_all_training, delete_all_training


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
    mahasiswa = get_all_mahasiswa()
    if request.method == "POST":
        # query all the data
        alumni = get_pandas_alumni()
        mhs = get_pandas_mahasiswa()
        # concat the data
        con = pd.concat([alumni, mhs.drop('id', axis=1)], keys=['train', 'test'], sort=True).fillna(0)
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
        # change the predict from number to string
        pred = pred.astype(str)
        for i in range(len(pred)):
            if pred[i] == '0':
                pred[i] = 'Tidak Tepat Waktu'
            else:
                pred[i] = 'Tepat Waktu'
        # insert the result to database
        i = 0
        for id in mhs['id'].values:
            existing_mahasiswa = get_an_id(int(id))
            if not existing_mahasiswa:
                data = Testing(
                    id_mahasiswa=int(id),
                    hasil=pred[i]
                )
                save_to_db(data)
            i += 1
        return redirect(url_for('dashboard.predict'))
    return render_template('mahasiswa.html', mahasiswa=mahasiswa)


@dashboard.route('/build', methods=["GET"])
def build():
    return render_template('model_building.html')


@dashboard.route('/upload', methods=["GET", "POST"])
def upload():
    return render_template('upload.html')


@dashboard.route('/cross_validation', methods=["GET", "POST"])
def cross_validation():
    # query all the data
    alumni = get_pandas_alumni()
    # if create model button is clicked
    if request.method == "POST":
        delete_all_training()
        for id in alumni['id'].values:
            data = Training(
                id_alumni=int(id)
            )
            save_to_db(data)
        flash('Successfully create a model', 'success')
        return redirect(url_for('dashboard.index'))
    # one hot encoder
    enc = pd.get_dummies(alumni.drop('id', axis=1))
    x = np.array(enc.drop('keterangan_lulus', axis=1))
    y = np.array(enc['keterangan_lulus'])
    # create the model
    model = MultinomialNB()
    # cross validation
    sf = StratifiedKFold(n_splits=10)
    cf = np.array([[0, 0], [0, 0]])
    f1 = []
    recall = []
    precision = []
    scores = []
    tprs = []
    aucs = []

    # make number with linspace between 0 and 1 with percentage.
    # e.g -> 0, 1%, 2% ... 99%, 100%
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train_index, test_index in sf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        # store the predict probabilities each test

        # compute confusion matrix, ROC curve and AUC
        cf += confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # compute the auc using fpr and tpr
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plot it
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        # calculate the accuracy, f1-score, recall, precision
        f1.append(f1_score(y_test, y_pred, average='macro'))
        recall.append(recall_score(y_test, y_pred, average='macro'))
        precision.append(precision_score(y_test, y_pred, average='macro'))
        score = model.score(x_test, y_test)
        scores.append(score)
        i += 1
    # loop above is same as cv = cross_val_score(model, x, y, cv=10)

    # put all scores inside a dict
    items = []
    for i in range(len(scores)):
        item = dict(numb=i + 1, f1=f1[i], precision=precision[i], recall=recall[i], score=scores[i])
        items.append(item)

    # put the average inside a dict
    avgitem = [dict(avg_f1=np.average(f1), avg_prec=np.average(precision),
                    avg_recall=np.average(recall), avg_score=np.average(scores))]

    # plot the ROC Curve, then save it into image file
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1
    # mean of the auc (from mean false positive rate and mean true positive rate)
    mean_auc = auc(mean_fpr, mean_tpr)
    # calculate the standard deviation for aucs
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    # calculate the standard deviation for tprs
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    output_path = os.path.join(path, 'static/ROC.png')
    plt.savefig(output_path)
    plt.clf()
    return render_template('cross_validation.html', scores=items, cf=cf, avg=avgitem)


@dashboard.route('/predict', methods=["GET", "POST"])
def predict():
    result = get_all_testing_result()
    return render_template('prediction.html', mahasiswa=result)


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
