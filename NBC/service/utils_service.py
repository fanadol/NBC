import pandas as pd

from NBC.config import Config


def clean_train_column(filename):
    """
    Make data frame from csv file. Suit the column with what we need for making alumni table
    Also, change th column name so its readable
    """
    col_names = ['NIM', 'Gender', 'Tipe Sekolah', 'Kota Sekolah', 'Tanggal Lulus', 'Lama Studi', 'Keterangan Lulus',
                 'IPS_1', 'IPS_2', 'IPS_3', 'IPS_4', 'Gaji Orang Tua']
    df = pd.read_csv(filename, names=col_names, skiprows=1)
    return df.drop(['Tanggal Lulus', 'Lama Studi'], axis=1)


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
    features_alumni = ['NIM', 'Tipe Sekolah', 'Gender', 'Kota Sekolah', 'Gaji Orang Tua', 'Keterangan Lulus',
                       'IPS_1', 'IPS_2', 'IPS_3', 'IPS_4']
    return df[features_alumni]


def clean_train_school_city(df):
    for i in range(len(df)):
        if 'yogya' in df.iloc[i].lower():
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'Dalam Kota')
        else:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'Luar Kota')
    return df


def clean_train_ipk(df):
    # multiple the ips 1 until ips 4 then assign to ipk
    ipk = (df['IPS_1'] + df['IPS_2'] + df['IPS_3'] + df['IPS_4']) / 4
    df_ipk = pd.DataFrame(ipk, columns=['IPK'])
    df = pd.concat([df, df_ipk], axis=1)
    return df


def clean_train_discretization(df):
    # change the value of IP and IPK to discret [A, B, C, D, E]. Change based on BukuPanduanAkademik
    ip = ['IPS_1', 'IPS_2', 'IPS_3', 'IPS_4', 'IPK']
    for data in ip:
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
