import datetime
import os
import time
import numpy as np
import pandas as pd
import matplotlib
from werkzeug.utils import secure_filename

from flask import render_template, request, redirect, url_for, flash
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from scipy import interp
from flask_login import login_required

from NBC.models.alumni import Alumni
from NBC.models.hasil import Hasil
from NBC.models.nilai import Nilai
from NBC.service.nilai_service import get_a_nilai, update_a_nilai
from . import dashboard
from NBC.models.testing import Testing
from NBC.models.training import Training
from NBC.service.alumni_service import get_all_alumni, get_an_alumni, update_an_alumni
from NBC.service.prediction_service import get_all_prediction_result, delete_all_prediction
from NBC.service.training_service import delete_all_training, get_all_training, get_a_training, update_a_training
from NBC.service.user_service import get_all_users, get_an_user, update_an_user
from NBC.service.database_service import save_to_db
from NBC.models.user import User
from NBC.config import Config
from NBC.service.utils_service import allowed_file

matplotlib.use('agg')

import matplotlib.pyplot as plt


@dashboard.route('/')
@login_required
def index():
    return redirect(url_for('dashboard.alumni'))


@dashboard.route('/users', methods=["GET", "POST"])
@login_required
def users():
    users = get_all_users(id=True)
    return render_template('admin_data_users.html', data=users)


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
        data = User(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            registered_on=datetime.datetime.utcnow(),
            phone_number=phone,
            role=role
        )
        save_to_db(data)
        flash('User successfully created', 'success')
        return redirect(url_for('dashboard.create_user'))
    return render_template('admin_create_user.html')


@dashboard.route('/user/<id>', methods=["GET", "POST"])
@login_required
def edit_user(id):
    user = get_an_user(id)
    if request.method == "POST":
        try:
            email = request.form.get('email')
            first_name = request.form.get('firstname')
            last_name = request.form.get('lastname')
            phone = request.form.get('phone')
            role = request.form.get('role')
            if not email or not first_name or not last_name or not phone:
                flash('Fill all empty form', 'danger')
                return redirect(request.url)
            updated_user = {
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'registered_on': datetime.datetime.utcnow(),
                'phone_number': phone,
                'admin': role
            }
            update_an_user(user, updated_user)
            flash('User berhasil di edit.', 'success')
            return redirect(url_for('dashboard.users'))
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_edit_user.html', user=user)


@dashboard.route('/alumni', methods=["GET", "POST"])
@login_required
def alumni():
    alumni = get_all_alumni(id=True)
    # POST METHOD
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No File Choosen', 'danger')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No File Choosen', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                timestr = time.strftime("%Y%m%d")
                filename = timestr + "_" + secure_filename(file.filename)
                path = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", filename)
                file.save(path)
                csv = pd.read_csv(path)
                for index, row in csv.iterrows():
                    data_alumni = Alumni(
                        id=row['NIM'],
                        school_type=row['Tipe Sekolah'],
                        gender=row['Gender'],
                        school_city=row['Kota Sekolah'],
                        parent_salary=row['Gaji Orang Tua'],
                        ket_lulus=row['Keterangan Lulus']
                    )
                    data_nilai = Nilai(
                        id_alumni=row['NIM'],
                        semester_1=row['IPS_1'],
                        semester_2=row['IPS_2'],
                        semester_3=row['IPS_3'],
                        semester_4=row['IPS_4'],
                        ipk=row['IPK']
                    )
                    # check if the data already exist
                    exst = Alumni.query.filter_by(id=row['NIM']).first()
                    if not exst:
                        save_to_db(data_alumni)
                        save_to_db(data_nilai)
                    else:
                        pass
                flash('Successfully menambahkan data alumni', 'success')
                return redirect(request.url)
            except Exception as e:
                flash('Error: {}'.format(e), 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file extension!', 'danger')
            return redirect(request.url)
    return render_template('admin_data_alumni.html', data=alumni)


@dashboard.route('/alumni/<id>', methods=["GET", "POST"])
@login_required
def edit_alumni(id):
    al = get_an_alumni(id)
    ni = get_a_nilai(id)
    if request.method == "POST":
        try:
            # if not request.form.get('id'):
            #     flash('Fill all empty form!', 'danger')
            #     return redirect(request.url)
            updated_alumni = {
                # 'id': request.form.get('id'),
                'school_type': request.form.get('school_type'),
                'gender': request.form.get('gender'),
                'school_city': request.form.get('school_city'),
                'parent_salary': request.form.get('parent_salary'),
                'ket_lulus': request.form.get('ket_lulus')
            }
            updated_nilai = {
                'semester_1': request.form.get('semester_1'),
                'semester_2': request.form.get('semester_2'),
                'semester_3': request.form.get('semester_3'),
                'semester_4': request.form.get('semester_4'),
                'ipk': request.form.get('ipk')
            }
            update_a_nilai(ni, updated_nilai)
            update_an_alumni(al, updated_alumni)
            return redirect(url_for('dashboard.alumni'))
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_edit_alumni.html', alumni=al, nilai=ni)


@dashboard.route('/training', methods=["GET", "POST"])
@login_required
def training():
    training_data = get_all_training()
    return render_template('admin_data_training.html', data=training_data)


@dashboard.route('/training/<id>', methods=["GET", "POST"])
@login_required
def edit_training(id):
    tr = get_a_training(id)
    if request.method == "POST":
        try:
            updated_training = {
                # 'id': request.form.get('id'),
                'school_type': request.form.get('school_type'),
                'gender': request.form.get('gender'),
                'school_city': request.form.get('school_city'),
                'parent_salary': request.form.get('parent_salary'),
                'ket_lulus': request.form.get('ket_lulus'),
                'semester_1': request.form.get('semester_1'),
                'semester_2': request.form.get('semester_2'),
                'semester_3': request.form.get('semester_3'),
                'semester_4': request.form.get('semester_4'),
                'ipk': request.form.get('ipk')
            }
            update_a_training(tr, updated_training)
            flash('Data Training berhasil di edit.', 'success')
            return redirect(url_for('dashboard.training'))
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_edit_training.html', latih=tr)


@dashboard.route('/prediction', methods=["GET", "POST"])
@login_required
def predict():
    pred_data = get_all_prediction_result()
    if request.method == "POST":
        delete_all_prediction()
        flash('Data telah berhasil di delete', 'success')
        return redirect(request.url)
    return render_template('admin_data_predict.html', data=pred_data)


@dashboard.route('/prediction_csv', methods=["GET", "POST"])
@login_required
def predict_csv():
    train = get_all_alumni()
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No File Choosen', 'danger')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No File Choosen', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                timestr = time.strftime("%Y%m%d")
                filename = timestr + "_" + secure_filename(file.filename)
                path = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "prediksi", filename)
                file.save(path)
                csv = pd.read_csv(path)
                # make list of dict
                pred_list = []
                for index, row in csv.iterrows():
                    pred_list.append({
                        'id': row['NIM'],
                        'school_type': row['Tipe Sekolah'],
                        'gender': row['Gender'],
                        'school_city': row['Kota Sekolah'],
                        'parent_salary': row['Gaji Orang Tua'],
                        'semester_1': row['IPS_1'],
                        'semester_2': row['IPS_2'],
                        'semester_3': row['IPS_3'],
                        'semester_4': row['IPS_4'],
                        'ipk': row['IPK']
                    })
                selected_features = ['school_type', 'gender', 'school_city', 'parent_salary', 'semester_1',
                                     'semester_2', 'semester_3', 'semester_4', 'ipk']
                data_test = pd.DataFrame(pred_list)
                data_target = data_test[selected_features]
                train['ket_lulus'] = train['ket_lulus'].replace(['Tidak Tepat Waktu', 'Tepat Waktu'], [0, 1])
                df_concat = pd.concat([train, data_target], keys=['train', 'test'], sort=True).fillna(0)
                df_concat['ket_lulus'] = df_concat['ket_lulus'].astype(int)
                enc = pd.get_dummies(df_concat)
                x = enc.loc['train'].drop('ket_lulus', axis=1)
                y = enc.loc['train']['ket_lulus']
                target = enc.loc['test'].drop('ket_lulus', axis=1)
                model = MultinomialNB()
                model.fit(x, y)
                y_pred = model.predict(target)
                # make data frame from y pred
                y_pred = pd.DataFrame(y_pred, columns=['result'])
                y_pred['result'] = y_pred['result'].replace([0, 1], ['Tidak Tepat Waktu', 'Tepat Waktu'])
                # insert into database
                # combine the result and the data
                dfr = pd.concat([data_test, y_pred], axis=1, sort=True)
                for i, row in dfr.iterrows():
                    exst = Testing.query.filter_by(id=row['id']).first()
                    if exst:
                        flash('ID terduplikasi, silahkan kosongkan tabel prediksi terlebih dahulu', 'danger')
                        return redirect(request.url)
                    obj_testing = Testing(
                        id=row['id'],
                        school_type=row['school_type'],
                        gender=row['gender'],
                        school_city=row['school_city'],
                        parent_salary=row['parent_salary'],
                        semester_1=row['semester_1'],
                        semester_2=row['semester_2'],
                        semester_3=row['semester_3'],
                        semester_4=row['semester_4'],
                        ipk=row['ipk']
                    )
                    obj_hasil = Hasil(
                        id_testing=row['id'],
                        result=row['result']
                    )
                    save_to_db(obj_testing)
                    save_to_db(obj_hasil)
                flash('Prediksi Berhasil', 'success')
                return redirect(url_for('dashboard.predict'))
            except Exception as e:
                flash('Error: {}'.format(e), 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file extension!', 'danger')
            return redirect(request.url)
    return render_template('admin_prediksi_csv.html')


@dashboard.route('/build', methods=["GET"])
@login_required
def build():
    return render_template('model_building.html')


@dashboard.route('/cross_validation', methods=["GET", "POST"])
@login_required
def cross_validation():
    # query all the data
    alumni = get_all_alumni(id=True)
    # if create model button is clicked
    if request.method == "POST":
        delete_all_training()
        for i, row in alumni.iterrows():
            data = Training(
                nim=row['id'],
                school_type=row['school_type'],
                gender=row['gender'],
                school_city=row['school_city'],
                parent_salary=row['parent_salary'],
                semester_1=row['semester_1'],
                semester_2=row['semester_2'],
                semester_3=row['semester_3'],
                semester_4=row['semester_4'],
                ipk=row['ipk'],
                ket_lulus=row['ket_lulus']
            )
            save_to_db(data)
        flash('Successfully create a model', 'success')
        return redirect(url_for('dashboard.index'))
    # one hot encoder
    alumni['ket_lulus'] = alumni['ket_lulus'].replace(['Tidak Tepat Waktu', 'Tepat Waktu'], [0, 1])
    enc = pd.get_dummies(alumni.drop(['id'], axis=1))
    x = np.array(enc.drop('ket_lulus', axis=1))
    y = np.array(enc['ket_lulus'])
    # create the model
    model = MultinomialNB()
    # cross validation
    kf = KFold(n_splits=10)
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
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fit the model, then predict the test fold
        # print("TRAIN: ", train_index, "TEST: ", test_index)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        # print("Y Test: {}".format(y_test))
        # print("Y Prob: {}".format(y_prob))
        # compute confusion matrix, ROC curve and AUC
        cf += confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # print("Unique y test: {}".format(pd.Series(y_test).unique()))
        # print("Unique y prob: {}".format(pd.Series(y_prob).unique()))
        # print("FPR: {}".format(fpr))
        # print("Threshold: {}".format(thresholds))
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
    # put all scores inside a dict
    items = []
    for i in range(len(scores)):
        item = dict(numb=i + 1, f1=f1[i], precision=precision[i], recall=recall[i], score=scores[i])
        items.append(item)

    # put the average inside a dict
    avgitem = [dict(avg_f1=np.average(f1), avg_prec=np.average(precision),
                    avg_recall=np.average(recall), avg_score=np.average(scores))]

    # plot the ROC Curve, then save it into image file
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
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
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std.dev.')
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
    rect_f1 = ax.bar(ind - 2 * bar_width, f1, bar_width, color='r', label='f1-score')
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
