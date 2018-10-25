import datetime
import os
import time
import numpy as np
import pandas as pd
import matplotlib
from sklearn.dummy import DummyClassifier
from werkzeug.utils import secure_filename

from flask import render_template, request, redirect, url_for, flash, send_from_directory
from sklearn.model_selection import KFold, StratifiedKFold
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
from NBC.service.alumni_service import get_all_alumni, get_an_alumni, update_an_alumni, delete_an_alumni, convert_nilai
from NBC.service.prediction_service import get_all_prediction_result, delete_all_prediction, delete_a_prediction
from NBC.service.training_service import delete_all_training, get_all_training, get_a_training, update_a_training, \
    delete_a_trainig
from NBC.service.user_service import get_all_users, get_an_user, update_an_user, delete_an_user
from NBC.service.database_service import save_to_db
from NBC.models.user import User
from NBC.config import Config
from NBC.service.utils_service import allowed_file, get_data_length, clean_train_discretization, grouping_school_type, \
    grouping_school_city

matplotlib.use('agg')

import matplotlib.pyplot as plt


@dashboard.route('/')
@login_required
def index():
    return redirect(url_for('dashboard.alumni'))


@dashboard.route('/preprocessing', methods=["GET", "POST"])
@login_required
def preprocessing():
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
                path_upload = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "preprocessing", filename)
                file.save(path_upload)
                csv = pd.read_csv(path_upload)
                # based on rule you should never modify what you are iterating over
                mylist = []
                for i, row in csv.iterrows():
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
                df = pd.DataFrame(mylist)
                # sort the column
                features = ['NIM', 'Tipe Sekolah', 'Gender', 'Kota Sekolah', 'IPS_1', 'IPS_2', 'IPS_3', 'IPS_4']
                df = df[features]
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
        exst = User.query.filter_by(email=email).first()
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
                path = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "alumni", filename)
                file.save(path)
                csv = pd.read_csv(path)
                for i, row in csv.iterrows():
                    # check if the data already exist
                    exst = Alumni.query.filter_by(id=row['NIM']).first()
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
        else:
            flash('Invalid file extension!', 'danger')
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
            exst = Alumni.query.filter_by(id=id).first()
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
        exst = Alumni.query.filter_by(id=id).first()
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
                # should delete this or no ?
                delete_all_training()
                timestr = time.strftime("%Y%m%d")
                filename = timestr + "_" + secure_filename(file.filename)
                path = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "training", filename)
                file.save(path)
                csv = pd.read_csv(path)
                for i, row in csv.iterrows():
                    # check if the data already exist
                    exst = Training.query.filter_by(id=row['NIM']).first()
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
        else:
            flash('Invalid file extension!', 'danger')
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
        exst = Training.query.filter_by(id=id).first()
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
    pred_data = get_all_prediction_result()
    len_data = get_data_length()
    # from delete button
    if request.method == "POST":
        delete_all_prediction()
        flash('Data telah berhasil dihapus', 'success')
        return redirect(request.url)
    return render_template('admin_data_predict.html', data=pred_data, len_data=len_data, modal_for='prediksi')


@dashboard.route('/prediction/create', methods=["GET", "POST"])
@login_required
def create_predict():
    # get training data for prediction
    train_data = get_all_training()
    if request.method == "POST":
        try:
            if not request.form.get('id'):
                flash('Fill all empty form!', 'danger')
                return redirect(request.url)
            # get the data from frontend form
            id = request.form.get('angkatan') + '.11.' + request.form.get('id')
            # check if the nim already in databases
            exst = Testing.query.filter_by(id=id).first()
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
            # make dataframe from test data
            df = pd.DataFrame(test_data, index=[0])
            # replace the train ket_lulus into number
            train_data['ket_lulus'] = train_data['ket_lulus'].replace(['Tidak Tepat Waktu', 'Tepat Waktu'], [0, 1])
            # concat the dataframe with train data, fill NaN with 0, convert ket_lulus into int type
            con = pd.concat([train_data.drop('id', axis=1), df.drop('id', axis=1)], keys=['train', 'test'],
                            sort=True).fillna(0)
            con['ket_lulus'] = con['ket_lulus'].astype(int)
            # one hot encoder all, make sure
            enc = pd.get_dummies(con)
            # split x, y, and target
            x = enc.loc['train'].drop('ket_lulus', axis=1)
            y = enc.loc['train']['ket_lulus']
            target = enc.loc['test'].drop('ket_lulus', axis=1)
            # build the model
            model = MultinomialNB()
            model.fit(x, y)
            # predict the data
            y_pred = model.predict(target)
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
                timestr = time.strftime("%Y%m%d")
                filename = timestr + "_" + secure_filename(file.filename)
                path = os.path.join(Config.ROOT_DIRECTORY, "NBC", "static", "upload", "prediksi", filename)
                file.save(path)
                csv = pd.read_csv(path)
                for i, row in csv.iterrows():
                    exst = Testing.query.filter_by(id=row['NIM']).first()
                    if exst:
                        flash('ID Terduplikasi, Tidak ada perubahan yang dilakukan.', 'danger')
                        return redirect(request.url)
                    # END FOR
                # make list of dict
                pred_list = []
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
                    pred_list.append({
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
                data_test = pd.DataFrame(pred_list)
                train_data['ket_lulus'] = train_data['ket_lulus'].replace(['Tidak Tepat Waktu', 'Tepat Waktu'], [0, 1])
                df_concat = pd.concat([train_data, data_test.drop('id', axis=1)], keys=['train', 'test'],
                                      sort=True).fillna(0)
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
        else:
            flash('Invalid file extension!', 'danger')
            return redirect(request.url)
    return render_template('admin_prediksi_csv.html')


@dashboard.route('/prediction/delete', methods=["POST"])
@login_required
def delete_predict():
    if request.method == "POST":
        id = request.form.get('id')
        exst = Testing.query.filter_by(id=id).first()
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
            dtobj = get_all_alumni(id=True)
        else:
            dtobj = get_all_training()
        # one hot encoder
        dtobj['ket_lulus'] = dtobj['ket_lulus'].replace(['Tidak Tepat Waktu', 'Tepat Waktu'], [0, 1])
        # i dont know how to chagne the value, so i make a list
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
        enc = pd.get_dummies(df.drop(['id'], axis=1))
        x = np.array(enc.drop('ket_lulus', axis=1))
        y = np.array(enc['ket_lulus'])
        labels = np.unique(y)
        # create the model
        model = MultinomialNB()
        # cross validation
        kf = KFold(n_splits=10)
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
            # calculate the accuracy, f1-score, recall, precision
            f1.append(f1_score(y_test, y_pred, average='macro'))
            recall.append(recall_score(y_test, y_pred, average='macro'))
            precision.append(precision_score(y_test, y_pred, average='macro'))
            score = model.score(x_test, y_test)
            scores.append(score)
        # put all scores inside a dict
        items = []
        for i in range(len(scores)):
            item = dict(numb=i + 1, f1=f1[i], precision=precision[i], recall=recall[i], score=scores[i])
            items.append(item)
        # put the average inside a dict
        avgitem = [dict(avg_f1=np.average(f1), avg_prec=np.average(precision),
                        avg_recall=np.average(recall), avg_score=np.average(scores))]
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        output_path_kfold = os.path.join(path, 'static/kfold.png')
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
        if dt == 'training':
            # save cross validation and confusion matrix as a csv file
            dfcv = pd.concat([pd.DataFrame(scores, columns=['accuracy']),
                              pd.DataFrame(precision, columns=['precision']),
                              pd.DataFrame(recall, columns=['recall']),
                              pd.DataFrame(f1, columns=['f1'])], axis=1)
            dfcf = pd.DataFrame(cf, columns=['P_Negative', 'P_Positive'])
            dfcv.to_csv('current_model_cv.csv', index=False, encoding='utf-8')
            dfcf.to_csv('current_model_cf.csv', index=False, encoding='utf-8')
        # if create model button is clicked
        if request.method == "POST":
            delete_all_training()
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
            # save cross validation and confusion matrix as a csv file
            dfcv = pd.concat([pd.DataFrame(scores, columns=['accuracy']),
                              pd.DataFrame(precision, columns=['precision']),
                              pd.DataFrame(recall, columns=['recall']),
                              pd.DataFrame(f1, columns=['f1'])])
            dfcf = pd.DataFrame(cf, columns=['P_Negative', 'P_Positive'])
            dfcv.to_csv('current_model_cv.csv', index=False, encoding='utf-8')
            dfcf.to_csv('current_model_cf.csv', index=False, encoding='utf-8')
            flash('Successfully create a model', 'success')
            return redirect(url_for('dashboard.index'))
        # < END POST REQUEST >
        return render_template('cross_validation.html', scores=items, cf=cf, avg=avgitem, dt=dt)
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
