import csv
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from flask import render_template, request

from NBC.config import Config
from . import examples
from NBC.service.prediction_service import pd_concat_row_csv, train_test_target_split
from NBC.service.utils_service import allowed_file, clean_train_column, clean_train_school_type, clean_train_with_mean, \
    clean_train_with_median, clean_train_reorder_column, clean_train_discret_salary, clean_train_school_city, \
    clean_train_ipk, clean_train_discretization


@examples.route('/predict_golf', methods=["GET"])
def pgolf():
    with open('golf_dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = []
        for line in reader:
            for l in line:
                data.append(l.split(','))
        data = np.array(data)
        df = pd.DataFrame(data, columns=['outlook', 'temperature', 'humidity', 'windy', 'play'])
        new_data = pd.DataFrame({
            'outlook': ['sunny'],
            'temperature': ['cool'],
            'humidity': ['high'],
            'windy': ['TRUE']
        })
        df2 = pd.concat([df, new_data], keys=['train', 'test'], sort=True).fillna(0)
        df2['play'] = df2['play'].astype(int)
        enc = pd.get_dummies(df2).astype(int)
        print(enc)
        model = MultinomialNB(alpha=0)
        x = enc.loc['train'].drop('play', axis=1)
        y = enc.loc['train']['play']
        pred_data = enc.loc['test'].drop('play', axis=1)
        model.fit(x, y)
        predict = model.predict(pred_data)
        predict_proba = model.predict_proba(pred_data)
    return render_template('predict_golf.html', predict=predict, predict_proba=predict_proba)


@examples.route('/test/mix')
def mix():
    with open('mhs_test_datasets.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = []
        for line in reader:
            for l in line:
                if l == "":
                    continue
                data.append(l.split(','))
        data = np.array(data)
        df = pd.DataFrame(data,
                          columns=['gender', 'school_type', 'school_city', 'parent_salary', 'semester_1', 'semester_2',
                                   'semester_3', 'semester_4', 'ipk', 'keterangan_lulus'])
        df['keterangan_lulus'] = df['keterangan_lulus'].astype(int)

        multiFeatures = ['gender', 'school_type', 'school_city']
        gaussFeatures = ['parent_salary', 'semester_1', 'semester_2', 'semester_3', 'semester_4', 'ipk']

        multiNB_data = df[multiFeatures + ['keterangan_lulus']]
        modelMulti = MultinomialNB(alpha=0)
        modelGauss = GaussianNB()
        df_target = pd.DataFrame({
            'gender': ['L'],
            'school_type': ['SMA'],
            'school_city': ['Dalam'],
            'parent_salary': [3500],
            'semester_1': [3.3],
            'semester_2': [3.4],
            'semester_3': [3.1],
            'semester_4': [3.2],
            'ipk': [3.3]
        })
        multiNB_dummy_target = df_target[multiFeatures]

        # one hot encoder
        concat = pd.concat([multiNB_data, multiNB_dummy_target], keys=['train', 'test'], sort=True).fillna(0)
        enc = pd.get_dummies(concat).astype(int)
        multiNB_x = enc.loc['train'].drop('keterangan_lulus', axis=1)
        multiNB_y = enc.loc['train']['keterangan_lulus']

        multiNB_target = enc.loc['test'].drop('keterangan_lulus', axis=1)
        gaussNB_target = df_target[gaussFeatures]

        gaussianNB_x = df[gaussFeatures]
        gaussianNB_y = df['keterangan_lulus']

        modelMulti.fit(multiNB_x, multiNB_y)
        modelGauss.fit(gaussianNB_x, gaussianNB_y)

        multipred = modelMulti.predict_proba(multiNB_target)
        gauss_pred = modelGauss.predict_proba(gaussNB_target)
        # if predict_proba, times the result
        # if predict_log_proba, add the result, then find the exponent from it
        result_pred = (multipred * gauss_pred)
        res = np.argmax(result_pred)

        # cross validation
        gauss_cv = cross_val_score(modelGauss, gaussianNB_x, gaussianNB_y, cv=5)
        multi_cv = cross_val_score(modelMulti, multiNB_x, multiNB_y, cv=5)
        total_cv = gauss_cv + multi_cv

        return render_template('multigaus.html', prob=multipred, prob2=gauss_pred, result=res, total_cv=total_cv,
                               probtot=result_pred)


@examples.route('/test/multinomial')
def ptest_multi():
    with open('mhs_test_datasets_multinomial.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = []
        for line in reader:
            for l in line:
                if l == "":
                    continue
                data.append(l.split(','))
        data = np.array(data)
        features_x = ['gender', 'school_type', 'school_city', 'parent_salary', 'semester_1', 'semester_2',
                      'semester_3', 'semester_4', 'ipk']
        features_y = ['keterangan_lulus']
        df_target = pd.DataFrame({
            'gender': ['L'],
            'school_type': ['SMA'],
            'school_city': ['Dalam'],
            'parent_salary': ['Rendah'],
            'semester_1': ['B'],
            'semester_2': ['B'],
            'semester_3': ['B'],
            'semester_4': ['B'],
            'ipk': ['B']
        })
        df = pd.DataFrame(data, columns=features_x + features_y)
        df[features_y] = df[features_y].astype(int)
        concat = pd_concat_row_csv(df, df_target)
        encode = pd.get_dummies(concat).astype(int)
        x, y, target = train_test_target_split(encode)
        print(target)
        model = MultinomialNB(alpha=0)
        model.fit(x, y)
        result = model.predict(target)
        prob = model.predict_proba(target)

        return render_template('multinomial.html', prob=prob, result=result)


@examples.route('/csv_maker', methods=["GET", "POST"])
def csv_maker():
    if request.method == "POST":
        path = Config.ROOT_DIRECTORY
        file = os.path.join(path, "Data Penelitian - Sheet3.csv")
        df = clean_train_column(file)
        df['Tipe Sekolah'] = clean_train_school_type(df['Tipe Sekolah'])
        df['Gaji Orang Tua'] = clean_train_discret_salary(df['Gaji Orang Tua'])
        df['Gaji Orang Tua'] = clean_train_with_median(df['Gaji Orang Tua'])
        df = clean_train_reorder_column(df)
        df['Kota Sekolah'] = clean_train_school_city(df['Kota Sekolah'])
        df = clean_train_ipk(df)
        df = clean_train_discretization(df, ['IPS_1', 'IPS_2', 'IPS_3', 'IPS_4', 'IPK'])
        df['Gaji Orang Tua'] = df['Gaji Orang Tua'].astype(str)
        df['Gaji Orang Tua'] = df['Gaji Orang Tua'].replace(['1', '2', '3', '4', '5', '6'],
                                                            ['Sangat Rendah', 'Rendah', 'Cukup Rendah', 'Cukup Tinggi',
                                                             'Tinggi', 'Sangat Tinggi'])
        df.to_csv('Data Result - Sheet3.csv', encoding='utf-8', index=False)
        return render_template('csv_maker.html')
    return render_template('csv_maker.html')
