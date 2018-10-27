import os

import pandas as pd
import matplotlib
import numpy as np

from NBC.config import Config
from NBC.models.alumni import Alumni
from NBC.models.testing import Testing
from NBC.models.training import Training
from NBC.models.user import User

matplotlib.use('agg')

import matplotlib.pyplot as plt

def clean_train_column(filename):
    """
    Make data frame from csv file. Suit the column with what we need for making alumni table
    Also, change th column name so its readable
    """
    col_names = ['NIM', 'Gender', 'Tipe Sekolah', 'Kota Sekolah', 'IPS_1', 'IPS_2', 'IPS_3', 'IPS_4']
    df = pd.read_csv(filename, names=col_names, skiprows=1)
    return df


def clean_train_school_type(df):
    """
    Make the school type for 3 type categories: SMK, SMA, Lain
    :param df: Data Frame for column school type
    :return: Pandas Data Frame
    """
    for i in range(len(df)):
        if 'smk' in df.iloc[i].lower():
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'SMK')
        elif 'sma' in df.iloc[i].lower():
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'SMA')
        else:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'Lain')
    return df


def clean_train_with_mean(df):
    df = df.replace('NaN', df.mean())
    return df


def clean_train_discret_salary(df):
    df = df.fillna('-1')
    for i in range(len(df)):
        if 'Kurang dari' in df.iloc[i]:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], '1')
        elif 'Rp. 999,999' in df.iloc[i]:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], '2')
        elif '1,000,000' in df.iloc[i]:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], '3')
        elif '2,000,000' in df.iloc[i]:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], '4')
        elif '5,000,000' in df.iloc[i]:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], '5')
        elif 'Lebih dari' in df.iloc[i]:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], '6')
        else:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'NaN')
    return df


def clean_train_with_median(df):
    df = df.replace('NaN', df.median().astype(int))
    return df


def clean_train_reorder_column(df):
    """
    Reordering the column
    :param df:
    :return:
    """
    features_alumni = ['NIM', 'Tipe Sekolah', 'Gender', 'Kota Sekolah', 'IPS_1', 'IPS_2', 'IPS_3',
                       'IPS_4']
    return df[features_alumni]


def clean_train_school_city(df):
    for i in range(len(df)):
        if 'yogya' in df.iloc[i].lower():
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'Dalam Kota')
        else:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'Luar Kota')
    return df

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

def clean_train_ipk(df):
    # multiple the ips 1 until ips 4 then assign to ipk
    ipk = (df['IPS_1'] + df['IPS_2'] + df['IPS_3'] + df['IPS_4']) / 4
    df_ipk = pd.DataFrame(ipk, columns=['IPK'])
    df = pd.concat([df, df_ipk], axis=1)
    return df


def clean_train_discretization(df, list_col):
    # change the value of IP and IPK to discret [A, B, C, D, E]. Change based on BukuPanduanAkademik
    for data in list_col:
        for i in range(len(df)):
            if df[data].iloc[i] >= 3.5:
                df[data].iloc[i] = 'A'
            elif df[data].iloc[i] >= 3:
                df[data].iloc[i] = 'B'
            elif df[data].iloc[i] >= 2.5:
                df[data].iloc[i] = 'C'
            elif df[data].iloc[i] >= 2:
                df[data].iloc[i] = 'D'
            else:
                df[data].iloc[i] = 'E'
    return df


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
    ax.set_title('Stratified 10 Fold Cross Validation')
    ax.set_xticks(ind + bar_width / 2)
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    ax.legend()
    fig.set_size_inches(15, 10)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.clf()


def create_pie_chart(path, labels, sizes):
    filename = 'piechart.png'
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(os.path.join(path, filename))
    plt.clf()
