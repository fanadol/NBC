import pandas as pd

from NBC.config import Config


def clean_train_column(filename):
    """
    Make data frame from csv file. Suit the column with what we need for making alumni table
    Also, change th column name so its readable
    """
    col_names = ['NIM', 'Gender', 'Asal Sekolah', 'Kota Sekolah', 'Tanggal Lulus', 'Lama Studi', 'Keterangan Lulus',
                 'IPS_1', 'IPS_2', 'IPS_3', 'IPS_4', 'Gaji Orang Tua']
    df = pd.read_csv(filename, names=col_names)
    return df.drop(['Tanggal Lulus', 'Lama Studi'])


def clean_train_school_type(df):
    """
    Make the school type for 3 type categories: SMK, SMA, Lain
    :param df: Data Frame for column school type
    :return: Pandas Data Frame
    """
    for i in range(len(df)):
        if 'smk' in df['Asal Sekolah'].iloc[i].lower():
            df.iloc[i] = df.iloc[i].replace(df.iloc[i], 'SMK')
        elif 'sma' in df['Asal Sekolah'].iloc[i].lower():
            df.iloc[i] = df.iloc[i].replace(df.iloc[i].values, 'SMA')
        else:
            df.iloc[i] = df.iloc[i].replace(df.iloc[i].values, 'Lain')
    return df


def clean_train_with_mean(df):
    df['Penghasilan Orang Tua'] = df['Penghasilan Orang Tua'].replace('NULL', df['Penghasilan Orang Tua'].mean())
    return df['Penghasilan Orang Tua']


def clean_train_with_median(df):
    df['Penghasilan Orang Tua'] = df['Penghasilan Orang Tua'].replace('NULL', df['Penghasilan Orang Tua'].median())
    return df['Penghasilan Orang Tua']


def clean_train_reorder_column(df):
    """
    Reordering the column
    :param df:
    :return:
    """
    features_alumni = ['NIM', 'Asal Sekolah', 'Gender', 'Kota Sekolah', 'Gaji Orang Tua', 'Keterangan Lulus',
                       'IPS_1', 'IPS_2', 'IPS_3', 'IPS_4']
    return df[features_alumni]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSION
