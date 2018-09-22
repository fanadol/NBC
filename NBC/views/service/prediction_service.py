import numpy as np
import pandas as pd

from NBC import db
from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai
from NBC.views.models.testing import Testing


def get_all_testing():
    return Testing.query.all()


def get_an_id(id):
    return Testing.query.filter_by(id_mahasiswa=id).first()


def get_all_testing_result():
    testing = Mahasiswa.query.join(Nilai, Mahasiswa.id == Nilai.id_mahasiswa).join(Testing,
                                                                                   Mahasiswa.id == Testing.id_mahasiswa) \
        .add_columns(Mahasiswa.id,
                     Mahasiswa.name,
                     Mahasiswa.school_type,
                     Mahasiswa.gender,
                     Mahasiswa.school_city,
                     Mahasiswa.parent_salary,
                     Nilai.semester_1,
                     Nilai.semester_2,
                     Nilai.semester_3,
                     Nilai.semester_4,
                     Nilai.ipk,
                     Testing.hasil).all()
    return testing


def pd_concat_row(data1, data2):
    return pd.concat([data1, data2.drop('id', axis=1)], keys=['train', 'test'], sort=True).fillna(0)


def train_test_target_split(enc):
    """
    Split the One Hot Encoder into x, y, and target data for predict purposes
    :param enc:
    :return zip(x, y, target):
    """
    x = np.array(enc.loc['train'].drop('keterangan_lulus', axis=1))
    y = np.array(enc.loc['train']['keterangan_lulus'])
    target = np.array(enc.loc['test'].drop('keterangan_lulus', axis=1))
    return list(zip(x, y, target))


def save_to_db(data):
    db.session.add(data)
    db.session.commit()
