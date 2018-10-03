import numpy as np
import pandas as pd

from NBC.models.hasil import Hasil
from NBC.models.testing import Testing


def get_a_prediction(id):
    return Testing.query.filter_by(id_mahasiswa=id).first()


def get_all_prediction_result(id=False):
    """
    Query prediction result from database, then change it into pandas dataframe.
    :param id = boolean:
    determine whether need id column or not
    :return pandas.DataFrame without Object Alumni, and id column:
    """
    id_feature = ['id']
    selected_features = ['TS', 'JK', 'KS', 'GO', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPK', 'Hasil']
    prediction = Testing.query.join(Hasil, Testing.id == Hasil.id_alumni).add_columns(Testing.id, Testing.tipe_sekolah,
                                                                                      Testing.gender,
                                                                                      Testing.kota_sekolah,
                                                                                      Testing.gaji_ortu,
                                                                                      Testing.semester_1,
                                                                                      Testing.semester_2,
                                                                                      Testing.semester_3,
                                                                                      Testing.semester_4, Testing.ipk,
                                                                                      Hasil.result).all()
    df = pd.DataFrame(prediction)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty and not id:
        return pd.DataFrame(prediction, columns=selected_features)
    elif df.empty and id:
        return pd.DataFrame(prediction, columns=id_feature + selected_features)
    # if data frame not empty
    else:
        if not id:
            return df.drop(['Mahasiswa', 'id'], axis=1)
        else:
            return df.drop('Mahasiswa', axis=1)


def pd_concat_row(data1, data2):
    return pd.concat([data1, data2.drop('id', axis=1)], keys=['train', 'test'], sort=True).fillna(0)


def pd_concat_row_csv(data1, data2):
    return pd.concat([data1, data2], keys=['train', 'test'], sort=True).fillna(0)


def train_test_target_split(enc):
    """
    Split the One Hot Encoder into x, y, and target data for predict purposes
    :param enc:
    :return zip(x, y, target):
    """
    x = np.array(enc.loc['train'].drop('keterangan_lulus', axis=1))
    y = np.array(enc.loc['train']['keterangan_lulus'])
    target = np.array(enc.loc['test'].drop('keterangan_lulus', axis=1))
    return x, y, target
