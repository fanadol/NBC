import numpy as np
import pandas as pd

from NBC import db
from NBC.models.hasil import Hasil
from NBC.models.testing import Testing


def get_a_prediction(id):
    return Testing.query.filter_by(id=id).first()


def get_all_prediction_result():
    """
    Query prediction result from database, then change it into pandas dataframe.
    determine whether need id column or not
    :return pandas.DataFrame without Object Alumni, and id column:
    """
    selected_features = ['id', 'school_type', 'gender', 'school_city', 'semester_1', 'semester_2',
                         'semester_3', 'semester_4', 'ipk', 'result']
    prediction = Testing.query.join(Hasil, Testing.id == Hasil.id_testing).add_columns(Testing.id, Testing.school_type,
                                                                                       Testing.gender,
                                                                                       Testing.school_city,
                                                                                       Testing.semester_1,
                                                                                       Testing.semester_2,
                                                                                       Testing.semester_3,
                                                                                       Testing.semester_4, Testing.ipk,
                                                                                       Hasil.result).all()
    df = pd.DataFrame(prediction)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty:
        return pd.DataFrame([], columns=selected_features)
        # if data frame not empty
    else:
        return df[selected_features]


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


def delete_a_prediction(id):
    Hasil.query.filter_by(id_testing=id).delete()
    Testing.query.filter_by(id=id).delete()
    db.session.commit()


def delete_all_prediction():
    Hasil.query.delete()
    Testing.query.delete()
    db.session.commit()
