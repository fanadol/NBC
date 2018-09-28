import datetime
import os
import uuid

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
from scipy.interpolate import spline
from flask import render_template, request, redirect, url_for, flash
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from scipy import interp
from flask_login import login_required

from . import dashboard
from NBC.views.models.testing import Testing
from NBC.views.models.training import Training
from NBC.views.service.alumni_service import get_all_alumni
from NBC.views.service.mahasiswa_service import get_all_mahasiswa
from NBC.views.service.prediction_service import train_test_target_split, get_all_prediction_result, \
    pd_concat_row, get_a_prediction
from NBC.views.service.training_service import delete_all_training
from NBC.views.service.user_service import get_all_users
from NBC.views.service.database_service import save_to_db
from NBC.views.models.user import User


@dashboard.route('/')
@login_required
def index():
    return redirect(url_for('dashboard.mahasiswa'))


@dashboard.route('/users', methods=["GET", "POST"])
@login_required
def users():
    users = get_all_users()
    users_to_html = users.to_html(
        classes='table table-striped table-bordered table-hover', table_id='dataTables-example', index=False, border=0)
    styled_table = users_to_html.replace('<table ', '<table style="width:100%" ')
    return render_template('admin_data.html', data=styled_table, object='User')


@dashboard.route('/alumni', methods=["GET"])
@login_required
def alumni():
    alumni = get_all_alumni()
    alumni.columns = ['Nama', 'TS', 'KS', 'JK', 'GO', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPK', 'Keterangan']
    alumni_to_html = alumni.to_html(
        classes='table table-striped table-bordered table-hover', table_id='dataTables-example', index=False, border=0
    )
    styled_table = alumni_to_html.replace('<table ', '<table style="width:100%" ')
    return render_template('admin_data.html', data=styled_table, object='Alumni')


@dashboard.route('/mahasiswa', methods=["GET", "POST"])
@login_required
def mahasiswa():
    mahasiswa = get_all_mahasiswa()
    if request.method == "POST":
        mahasiswa = get_all_mahasiswa(id=True)
        alumni = get_all_alumni()
        # concat the data without column id in each mahasiswa and alumni
        con = pd_concat_row(alumni, mahasiswa.drop('id', axis=1)).fillna(0)
        # one hot encoder, axis 1 mean column
        enc = pd.get_dummies(con).astype(int)
        # split the x, y, and target
        x, y, target = train_test_target_split(enc)
        # create the model
        model = MultinomialNB()
        # predict the 10th data from dataset for testing purpose
        model.fit(x, y)
        pred = model.predict(target)
        # change the predict from number to string
        df = pd.DataFrame(pred, columns='hasil')
        df['hasil'] = df['hasil'].replace([0, 1], ['Tidak Tepat Waktu', 'Tepat Waktu'])
        # insert the result to database
        i = 0
        for id in mahasiswa['id'].values:
            existing_mahasiswa = get_a_prediction(int(id))
            if not existing_mahasiswa:
                data = Testing(
                    id_mahasiswa=int(id),
                    hasil=df['hasil'].iloc[i]
                )
                save_to_db(data)
            i += 1
        return redirect(url_for('dashboard.predict'))
    mahasiswa.columns = ['Nama', 'TS', 'KS', 'JK', 'GO', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPK']
    mahasiswa_to_html = mahasiswa.to_html(
        classes='table table-striped table-bordered table-hover', table_id='dataTables-example', index=False, border=0)
    styled_table = mahasiswa_to_html.replace('<table ', '<table style="width:100%" ')
    return render_template('admin_data.html', data=styled_table, object='Mahasiswa')


@dashboard.route('/prediction', methods=["GET", "POST"])
@login_required
def predict():
    result = get_all_prediction_result()
    result.columns = ['Nama', 'TS', 'KS', 'JK', 'GO', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPK', 'Hasil']
    result_to_html = result.to_html(
        classes='table table-striped table-bordered table-hover', table_id='dataTables-example', index=False, border=0)
    styled_table = result_to_html.replace('<table ', '<table style="width:100%" ')
    return render_template('admin_data.html', data=styled_table, object='Prediksi')


@dashboard.route('/build', methods=["GET"])
@login_required
def build():
    return render_template('model_building.html')


@dashboard.route('/create_user', methods=["GET", "POST"])
def create_user():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        first_name = request.form.get('firstname')
        last_name = request.form.get('lastname')
        phone = request.form.get('phone')
        role = request.form.get('role')
        if not email or not password or not first_name or not last_name:
            flash('Please fill all form!', 'danger')
            return redirect(url_for('dashboard.create_user'))
        if phone:
            phone = '+62' + phone
        # check if admin, i dont know why its always error if direct pass from HTML
        if role == 'Admin':
            role = True
        else:
            role = False
        data = User(
            public_id=str(uuid.uuid4()),
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            registered_on=datetime.datetime.utcnow(),
            phone_number=phone,
            admin=role
        )
        save_to_db(data)
        flash('User successfully created', 'success')
        return redirect(url_for('dashboard.create_user'))
    return render_template('admin_create_user.html')


@dashboard.route('/cross_validation', methods=["GET", "POST"])
@login_required
def cross_validation():
    # query all the data
    alumni = get_all_alumni(id=True)
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
    alumni['keterangan_lulus'] = alumni['keterangan_lulus'].replace(['Tidak Tepat Waktu', 'Tepat Waktu'], [0, 1])
    enc = pd.get_dummies(alumni.drop(['id', 'name'], axis=1))
    x = np.array(enc.drop('keterangan_lulus', axis=1))
    y = np.array(enc['keterangan_lulus'])
    # create the model
    model = MultinomialNB()
    # cross validation
    sf = KFold(n_splits=10)
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
    plt.figure(figsize=(15, 10))
    i = 0
    for train_index, test_index in sf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fit the model, then predict the test fold
        print("TRAIN: ", train_index, "TEST: ", test_index)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        print("Y Test: {}".format(y_test))
        print("Y Prob: {}".format(y_prob))
        # compute confusion matrix, ROC curve and AUC
        cf += confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        print("FPR: {} TPR: {}".format(fpr, tpr))
        # y_smooth = spline(fpr, tpr, mean_fpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # print("fpr: {}".format(fpr))
        # print("tpr: {}".format(tpr))
        # compute the auc using fpr and tpr
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plot it
        plt.plot(fpr, tpr, lw=1, alpha=0.5, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        # calculate the accuracy, f1-score, recall, precision
        f1.append(f1_score(y_test, y_pred, average='macro'))
        recall.append(recall_score(y_test, y_pred, average='macro'))
        precision.append(precision_score(y_test, y_pred, average='macro'))
        score = model.score(x_test, y_test)
        scores.append(score)
        i += 1
    # loop above is same as cv = cross_val_score(model, x, y, cv=10)
    # print("tprs: {}".format(tprs))
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
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    output_path = os.path.join(path, 'static/ROC.png')
    output_path_kfold = os.path.join(path, 'static/kfold.png')
    plt.savefig(output_path)
    plt.clf()
    # plot the k fold
    fig, ax = plt.subplots()
    bar_width = 0.2
    N = 10
    ind = np.arange(N)
    rect_f1 = ax.bar(ind - 2*bar_width, f1, bar_width, color='r', label='f1-score')
    rect_recall = ax.bar(ind - bar_width, recall, bar_width, color='g', label='recall')
    rect_precision = ax.bar(ind, precision, bar_width, color='b', label='precision')
    rect_score = ax.bar(ind + bar_width, scores, bar_width, color='y', label='accuracy')
    ax.set_xlabel('Folds')
    ax.set_ylabel('Scores')
    ax.set_title('Stratified 10 Fold Cross Validation')
    ax.set_xticks(ind + bar_width / 2)
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    ax.legend()
    fig.set_size_inches(15, 10)
    fig.tight_layout()
    plt.savefig(output_path_kfold)
    plt.clf()
    return render_template('cross_validation.html', scores=items, cf=cf, avg=avgitem)
