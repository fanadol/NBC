import datetime
import os
import time
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from flask import render_template, request, redirect, url_for, flash, send_from_directory
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from flask_login import login_required

from NBC.models.alumni import Alumni
from NBC.models.hasil import Hasil
from NBC.models.nilai import Nilai
from NBC.service.nilai_service import get_a_nilai, update_a_nilai
from . import dashboard
from NBC.models.testing import Testing
from NBC.models.training import Training
from NBC.service.alumni_service import get_all_alumni, get_an_alumni, update_an_alumni, delete_an_alumni, convert_nilai
from NBC.service.prediction_service import get_all_prediction_result, delete_all_prediction, delete_a_prediction, \
    get_a_prediction
from NBC.service.training_service import delete_all_training, get_all_training, get_a_training, update_a_training, \
    delete_a_trainig
from NBC.service.user_service import get_all_users, get_an_user, update_an_user, delete_an_user
from NBC.service.database_service import save_to_db
from NBC.models.user import User
from NBC.config import Config
from NBC.service.utils_service import allowed_file, get_data_length, grouping_school_type, grouping_school_city, \
    create_bar_chart, create_pie_chart, check_upload_file, predict_data


@dashboard.route('/')
@login_required
def index():
    return redirect(url_for('dashboard.alumni'))


@dashboard.route('/preprocessing', methods=["GET", "POST"])
@login_required
def preprocessing():
    if request.method == "POST":
        # check if the post request has the file part
        file = check_upload_file(request)
        if not file:
            flash('No File Selected', 'danger')
            return redirect(request.url)
        try:
            timestr = time.strftime("%Y%m%d")
            filename = timestr + "_" + secure_filename(file.filename)
            path_upload = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "preprocessing", filename)
            file.save(path_upload)
            csv = pd.read_csv(path_upload)
            # based on rule you should never modify what you are iterating over
            mylist = []
            for i, row in csv.iterrows():
                # data selection and grouping
                mylist.append({
                    'NIM': row['NIM'],
                    'Tipe Sekolah': grouping_school_type(row['Tipe Sekolah']),
                    'Gender': row['Gender'],
                    'Kota Sekolah': grouping_school_city(row['Kota Sekolah']),
                    'IPS_1': row['IPS_1'],
                    'IPS_2': row['IPS_2'],
                    'IPS_3': row['IPS_3'],
                    'IPS_4': row['IPS_4'],
                    'Keterangan Lulus': row['Keterangan Lulus']
                })
            features = ['NIM', 'Tipe Sekolah', 'Gender', 'Kota Sekolah', 'IPS_1', 'IPS_2', 'IPS_3', 'IPS_4']
            df = pd.DataFrame(mylist, columns=features)
            path_save = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "preprocessing_result",
                                     filename)
            path_download = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "preprocessing_result")
            df.to_csv(path_save, index=False, encoding='utf-8')
            return send_from_directory(directory=path_download, filename=filename, as_attachment=True)
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(url_for('dashboard.alumni'))
    return render_template('admin_preprocessing.html')


@dashboard.route('/users', methods=["GET", "POST"])
@login_required
def users():
    users = get_all_users(id=True)
    len_data = get_data_length()
    return render_template('admin_data_users.html', data=users, len_data=len_data, modal_for='user')


@dashboard.route('/create_user', methods=["GET", "POST"])
@login_required
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
        flash('User berhasil dibuat', 'success')
        return redirect(url_for('dashboard.create_user'))
    return render_template('admin_user_create.html')


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


@dashboard.route('/user/delete', methods=["POST"])
@login_required
def delete_user():
    if request.method == "POST":
        email = request.form.get('id')
        exst = get_an_user(email)
        if exst:
            delete_an_user(email)
            flash('User success dihapus.', 'success')
            return redirect(url_for('dashboard.users'))
        else:
            flash('Error: Data did not exist.', 'danger')
            return redirect(url_for('dashboard.users'))


@dashboard.route('/alumni', methods=["GET", "POST"])
@login_required
def alumni():
    len_data = get_data_length()
    alumni = get_all_alumni()
    # POST METHOD
    if request.method == "POST":
        file = check_upload_file(request)
        if not file:
            flash('No File or Wrong Extension!', 'danger')
            return redirect(request.url)
        try:
            timestr = time.strftime("%Y%m%d")
            filename = timestr + "_" + secure_filename(file.filename)
            path = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "alumni", filename)
            file.save(path)
            csv = pd.read_csv(path)
            for i, row in csv.iterrows():
                # check if the data already exist
                exst = get_an_alumni(row['NIM'])
                if exst:
                    flash('ID Terduplikasi, Tidak ada perubahan yang dilakukan.', 'danger')
                    return redirect(request.url)
            for index, row in csv.iterrows():
                data_alumni = Alumni(
                    id=row['NIM'],
                    school_type=row['Tipe Sekolah'],
                    gender=row['Gender'],
                    school_city=row['Kota Sekolah'],
                    ket_lulus=row['Keterangan Lulus']
                )
                # make ipk
                ipk = (row['IPS_1'] + row['IPS_2'] + row['IPS_3'] + row['IPS_4']) / 4
                # save the IP value
                data_nilai = Nilai(
                    id_alumni=row['NIM'],
                    semester_1=row['IPS_1'],
                    semester_2=row['IPS_2'],
                    semester_3=row['IPS_3'],
                    semester_4=row['IPS_4'],
                    ipk=ipk
                )
                save_to_db(data_alumni)
                save_to_db(data_nilai)
            flash('Successfully menambahkan data alumni', 'success')
            return redirect(request.url)
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_data_alumni.html', data=alumni, len_data=len_data, modal_for='alumni')


@dashboard.route('/alumni/create', methods=["GET", "POST"])
@login_required
def create_alumni():
    if request.method == "POST":
        try:
            # get data from form
            id = request.form.get('angkatan') + '.11.' + request.form.get('id').zfill(4)
            semester_1 = float(request.form.get('semester_1'))
            semester_2 = float(request.form.get('semester_2'))
            semester_3 = float(request.form.get('semester_3'))
            semester_4 = float(request.form.get('semester_4'))
            ipk = (semester_1 + semester_2 + semester_3 + semester_4) / 4
            # check if the nim is conflict or not
            exst = get_an_alumni(id)
            if exst:
                flash('Error: NIM Conflict!', 'danger')
                return redirect(request.url)
            # make an Alumni model object
            que_alumni = Alumni(
                id=id,
                school_type=request.form.get('school_type'),
                gender=request.form.get('gender'),
                school_city=request.form.get('school_city'),
                ket_lulus=request.form.get('ket_lulus')
            )
            que_nilai = Nilai(
                id_alumni=id,
                semester_1=semester_1,
                semester_2=semester_2,
                semester_3=semester_3,
                semester_4=semester_4,
                ipk=ipk
            )
            # insert into database
            save_to_db(que_alumni)
            save_to_db(que_nilai)
            flash('Success membuat data alumni.', 'success')
            return redirect(url_for('dashboard.create_alumni'))
        except Exception as e:
            flash('Error: {}'.format(e))
            return redirect(request.url)
    return render_template('admin_create_alumni.html')


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
                # 'parent_salary': request.form.get('parent_salary'),
                'ket_lulus': request.form.get('ket_lulus')
            }
            updated_nilai = {
                'semester_1': float(request.form.get('semester_1')),
                'semester_2': float(request.form.get('semester_2')),
                'semester_3': float(request.form.get('semester_3')),
                'semester_4': float(request.form.get('semester_4')),
                'ipk': (float(request.form.get('semester_1')) + float(request.form.get('semester_2')) + float(
                    request.form.get('semester_3')) + float(request.form.get('semester_4'))) / 4
            }
            update_a_nilai(ni, updated_nilai)
            update_an_alumni(al, updated_alumni)
            flash('Alumni berhasil diubah.', 'success')
            return redirect(url_for('dashboard.alumni'))
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_edit_alumni.html', alumni=al, nilai=ni)


@dashboard.route('/alumni/delete', methods=["POST"])
@login_required
def delete_alumni():
    if request.method == "POST":
        id = request.form.get('id')
        exst = get_an_alumni(id)
        if exst:
            delete_an_alumni(id)
            flash('Alumni berhasil dihapus.', 'success')
            return redirect(url_for('dashboard.alumni'))
        else:
            flash('Error: Data did not exist.', 'danger')
            return redirect(url_for('dashboard.alumni'))


@dashboard.route('/training', methods=["GET", "POST"])
@login_required
def training():
    training_data = get_all_training()
    len_data = get_data_length()
    if request.method == "POST":
        file = check_upload_file(request)
        if not file:
            flash('No File or Wrong Extension!', 'danger')
            return redirect(request.url)
        try:
            # should delete this or no ?
            delete_all_training()
            timestr = time.strftime("%Y%m%d")
            filename = timestr + "_" + secure_filename(file.filename)
            path = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "training", filename)
            file.save(path)
            csv = pd.read_csv(path)
            for i, row in csv.iterrows():
                # check if the data already exist
                exst = get_a_training(row['NIM'])
                if exst:
                    flash('ID Terduplikasi, Tidak ada perubahan yang dilakukan.', 'danger')
                    return redirect(request.url)
            for index, row in csv.iterrows():
                # make ipk
                ipk = (row['IPS_1'] + row['IPS_2'] + row['IPS_3'] + row['IPS_4']) / 4
                data_training = Training(
                    id=row['NIM'],
                    school_type=row['Tipe Sekolah'],
                    gender=row['Gender'],
                    school_city=row['Kota Sekolah'],
                    ket_lulus=row['Keterangan Lulus'],
                    semester_1=row['IPS_1'],
                    semester_2=row['IPS_2'],
                    semester_3=row['IPS_3'],
                    semester_4=row['IPS_4'],
                    ipk=ipk
                )
                save_to_db(data_training)
            flash('Success menambahkan data alumni', 'success')
            return redirect(url_for('dashboard.cross_validation', dt='training'))
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_data_training.html', data=training_data, len_data=len_data, modal_for='data training')


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
                'ket_lulus': request.form.get('ket_lulus'),
                'semester_1': float(request.form.get('semester_1')),
                'semester_2': float(request.form.get('semester_2')),
                'semester_3': float(request.form.get('semester_3')),
                'semester_4': float(request.form.get('semester_4')),
                'ipk': (float(request.form.get('semester_1')) + float(request.form.get('semester_2')) + float(
                    request.form.get('semester_3')) + float(request.form.get('semester_4'))) / 4
            }
            update_a_training(tr, updated_training)
            flash('Data Training berhasil di edit.', 'success')
            return redirect(url_for('dashboard.training'))
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_edit_training.html', latih=tr)


@dashboard.route('/training/delete', methods=["POST"])
@login_required
def delete_training():
    if request.method == "POST":
        id = request.form.get('id')
        exst = get_a_training(id)
        if exst:
            delete_a_trainig(id)
            flash('Data training success dihapus.', 'success')
            return redirect(url_for('dashboard.training'))
        else:
            flash('Error: Data did not exist.', 'danger')
            return redirect(url_for('dashboard.training'))


@dashboard.route('/prediction', methods=["GET", "POST"])
@login_required
def predict():
    show = False
    pred_data = get_all_prediction_result()
    len_data = get_data_length()
    if len(pred_data) > 0:
        show = True
        cnt = pred_data['result'].value_counts()
        if 'Tepat Waktu' not in cnt:
            cnt['Tepat Waktu'] = 0
        if 'Tidak Tepat Waktu' not in cnt:
            cnt['Tidak Tepat Waktu'] = 0
        labels = ['Tidak Tepat Waktu', 'Tepat Waktu']
        sizes = [cnt['Tidak Tepat Waktu'], cnt['Tepat Waktu']]
        create_pie_chart(Config.STATIC_DIRECTORY, labels, sizes)
    # from delete button
    if request.method == "POST":
        delete_all_prediction()
        flash('Data telah berhasil dihapus', 'success')
        return redirect(request.url)
    return render_template('admin_data_predict.html', data=pred_data, len_data=len_data, modal_for='prediksi',
                           show=show)


@dashboard.route('/prediction/create', methods=["GET", "POST"])
@login_required
def create_predict():
    if request.method == "POST":
        try:
            # get training data for prediction
            train_data = get_all_training()
            if train_data.empty:
                flash('Tidak ada data training, silahkan buat model terlebih dahulu.', 'danger')
                return redirect(request.url)
            if not request.form.get('id'):
                flash('Fill all empty form!', 'danger')
                return redirect(request.url)
            # get the data from frontend form
            id = request.form.get('angkatan') + '.11.' + request.form.get('id').zfill(4)
            # check if the nim already in databases
            exst = get_a_prediction(id)
            # if already in databases, return with alert error
            if exst:
                flash('Error, NIM Duplicated!', 'danger')
                return redirect(request.url)
            semester_1 = float(request.form.get('semester_1'))
            semester_2 = float(request.form.get('semester_2'))
            semester_3 = float(request.form.get('semester_3'))
            semester_4 = float(request.form.get('semester_4'))
            ipk = (semester_1 + semester_2 + semester_3 + semester_4) / 4
            # change the value into categorical
            test_data = {
                'id': id,
                'school_type': request.form.get('school_type'),
                'gender': request.form.get('gender'),
                'school_city': request.form.get('school_city'),
                'semester_1': convert_nilai(semester_1),
                'semester_2': convert_nilai(semester_2),
                'semester_3': convert_nilai(semester_3),
                'semester_4': convert_nilai(semester_4),
                'ipk': convert_nilai(ipk)
            }
            y_pred = predict_data(train_data, test_data, index=True)
            # convert the result into string
            if y_pred == 0:
                y_pred = 'Tidak Tepat Waktu'
            else:
                y_pred = 'Tepat Waktu'
            # insert into database
            que_test = Testing(
                id=test_data['id'],
                school_type=test_data['school_type'],
                gender=test_data['gender'],
                school_city=test_data['school_city'],
                # parent_salary=test_data['parent_salary'],
                semester_1=semester_1,
                semester_2=semester_2,
                semester_3=semester_3,
                semester_4=semester_4,
                ipk=ipk
            )
            que_hasil = Hasil(
                id_testing=test_data['id'],
                result=y_pred
            )
            save_to_db(que_test)
            save_to_db(que_hasil)
            flash('Success memprediksi data.', 'success')
            return redirect(url_for('dashboard.predict'))
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_create_prediksi.html')


@dashboard.route('/prediction/csv', methods=["GET", "POST"])
@login_required
def predict_csv():
    train_data = get_all_training()
    if request.method == "POST":
        try:
            file = check_upload_file(request)
            if not file:
                flash('No File or Wrong Extension!', 'danger')
                return redirect(request.url)
            timestr = time.strftime("%Y%m%d")
            filename = timestr + "_" + secure_filename(file.filename)
            path = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "prediksi", filename)
            file.save(path)
            csv = pd.read_csv(path)
            for i, row in csv.iterrows():
                exst = get_a_prediction(row['NIM'])
                if exst:
                    flash('ID Terduplikasi, Tidak ada perubahan yang dilakukan.', 'danger')
                    return redirect(request.url)
                # END FOR
            # make list of dict
            data_target = []
            for index, row in csv.iterrows():
                ipk = (row['IPS_1'] + row['IPS_2'] + row['IPS_3'] + row['IPS_4']) / 4
                # save the data into database
                # the reason is, i want to store the value with number data type, not discret.
                obj_testing = Testing(
                    id=row['NIM'],
                    school_type=row['Tipe Sekolah'],
                    gender=row['Gender'],
                    school_city=row['Kota Sekolah'],
                    semester_1=row['IPS_1'],
                    semester_2=row['IPS_2'],
                    semester_3=row['IPS_3'],
                    semester_4=row['IPS_4'],
                    ipk=ipk
                )
                save_to_db(obj_testing)
                # convert all IP and insert it to list
                data_target.append({
                    'id': row['NIM'],
                    'school_type': row['Tipe Sekolah'],
                    'gender': row['Gender'],
                    'school_city': row['Kota Sekolah'],
                    'semester_1': convert_nilai(row['IPS_1']),
                    'semester_2': convert_nilai(row['IPS_2']),
                    'semester_3': convert_nilai(row['IPS_3']),
                    'semester_4': convert_nilai(row['IPS_4']),
                    'ipk': convert_nilai(ipk)
                })
                # END FOR
            y_pred = predict_data(train_data, data_target)
            # make data frame from y pred
            y_pred = pd.DataFrame(y_pred, columns=['result'])
            y_pred['result'] = y_pred['result'].replace([0, 1], ['Tidak Tepat Waktu', 'Tepat Waktu'])
            # insert into database
            # combine the result and the data
            dfr = pd.concat([pd.DataFrame(data_target), y_pred], axis=1, sort=True)
            for i, row in dfr.iterrows():
                obj_hasil = Hasil(
                    id_testing=row['id'],
                    result=row['result']
                )
                save_to_db(obj_hasil)
            flash('Prediksi Berhasil', 'success')
            return redirect(url_for('dashboard.predict'))
        except Exception as e:
            flash('Error: {}'.format(e), 'danger')
            return redirect(request.url)
    return render_template('admin_prediksi_csv.html')


@dashboard.route('/prediction/delete', methods=["POST"])
@login_required
def delete_predict():
    if request.method == "POST":
        id = request.form.get('id')
        exst = get_a_prediction(id)
        if exst:
            delete_a_prediction(id)
            flash('Data prediksi success dihapus.', 'success')
            return redirect(url_for('dashboard.predict'))
        else:
            flash('Error: Data did not exist.', 'danger')
            return redirect(url_for('dashboard.training'))


@dashboard.route('/cross_validation/<dt>', methods=["GET", "POST"])
@login_required
def cross_validation(dt):
    try:
        # query all the data
        if dt == 'alumni':
            dtobj = get_all_alumni()
        else:
            dtobj = get_all_training()
        if dtobj.empty:
            flash('Tidak ada data.', 'danger')
            return redirect(url_for('dashboard.alumni'))
        # convert class label into number, so that it will not be encoded
        dtobj['ket_lulus'] = dtobj['ket_lulus'].replace(['Tidak Tepat Waktu', 'Tepat Waktu'], [0, 1])
        # based on you should never change what you itterating over, make a list and make a new data frame
        mydf = []
        for i, row in dtobj.iterrows():
            mydf.append({
                'id': row['id'],
                'school_type': row['school_type'],
                'gender': row['gender'],
                'school_city': row['school_city'],
                'semester_1': convert_nilai(row['semester_1']),
                'semester_2': convert_nilai(row['semester_2']),
                'semester_3': convert_nilai(row['semester_3']),
                'semester_4': convert_nilai(row['semester_4']),
                'ipk': convert_nilai(row['ipk']),
                'ket_lulus': row['ket_lulus']
            })
        df = pd.DataFrame(mydf)
        # One Hot Encoder
        enc = pd.get_dummies(df.drop(['id'], axis=1))
        # Split X, and Y
        x = np.array(enc.drop('ket_lulus', axis=1))
        y = np.array(enc['ket_lulus'])
        # determine the label for confusion matrix
        labels = np.unique(y)
        # create the model
        model = MultinomialNB()
        # initialize kfold
        kf = KFold(n_splits=10)
        # initialize confusion matrix
        cf = np.array([[0, 0], [0, 0]])
        f1 = []
        recall = []
        precision = []
        scores = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # fit the model, then predict the test fold
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            # compute confusion matrix, ROC curve and AUC
            cf += confusion_matrix(y_test, y_pred, labels=labels)
            print(cf)
            # calculate the accuracy, f1-score, recall, precision
            f1.append(f1_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred))
            scores.append(model.score(x_test, y_test))
        dfcv = pd.concat([pd.DataFrame(scores, columns=['accuracy']),
                          pd.DataFrame(precision, columns=['precision']),
                          pd.DataFrame(recall, columns=['recall']),
                          pd.DataFrame(f1, columns=['f1'])], axis=1)
        # # put the average inside a dict
        # avgitem = [dict(avg_f1=np.average(f1), avg_prec=np.average(precision),
        #                 avg_recall=np.average(recall), avg_score=np.average(scores))]
        path_model_kfold = os.path.join(Config.STATIC_DIRECTORY, 'kfold_model.png')
        path_result_kfold = os.path.join(Config.STATIC_DIRECTORY, 'kfold_result.png')
        create_bar_chart(path_model_kfold, f1, recall, precision, scores)
        if dt == 'training':
            dfcf = pd.DataFrame(cf, columns=['P_Negative', 'P_Positive'])
            create_bar_chart(path_result_kfold, f1, recall, precision, scores)
            dfcv.to_csv('current_model_cv.csv', index=False, encoding='utf-8')
            dfcf.to_csv('current_model_cf.csv', index=False, encoding='utf-8')
        # if create model button is clicked
        if request.method == "POST":
            delete_all_training()
            create_bar_chart(path_result_kfold, f1, recall, precision, scores)
            dtobj['ket_lulus'] = dtobj['ket_lulus'].replace([0, 1], ['Tidak Tepat Waktu', 'Tepat Waktu'])
            for i, row in dtobj.iterrows():
                data = Training(
                    id=row['id'],
                    school_type=row['school_type'],
                    gender=row['gender'],
                    school_city=row['school_city'],
                    semester_1=row['semester_1'],
                    semester_2=row['semester_2'],
                    semester_3=row['semester_3'],
                    semester_4=row['semester_4'],
                    ipk=row['ipk'],
                    ket_lulus=row['ket_lulus']
                )
                save_to_db(data)
            dfcf = pd.DataFrame(cf, columns=['P_Negative', 'P_Positive'])
            dfcv.to_csv('current_model_cv.csv', index=False, encoding='utf-8')
            dfcf.to_csv('current_model_cf.csv', index=False, encoding='utf-8')
            flash('Successfully create a model', 'success')
            return redirect(url_for('dashboard.index'))
        # < END POST REQUEST >
        return render_template('cross_validation.html', scores=dfcv, cf=cf, dt=dt)
    except Exception as e:
        flash('Error: {}'.format(e), 'danger')
        return redirect(url_for('dashboard.alumni'))


@dashboard.route('/current_model')
@login_required
def current_model():
    try:
        cv = pd.read_csv('current_model_cv.csv')
        cf = pd.read_csv('current_model_cf.csv')
        return render_template('admin_check_current_model.html', cv=cv, cf=cf)
    except:
        flash('Belum ada model, silahkan buat model terlebih dahulu!', 'danger')
        return redirect(url_for('dashboard.alumni'))


@dashboard.route('/manual', methods=["GET"])
@login_required
def guide():
    return render_template('admin_guide_book.html')
