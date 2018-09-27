import pandas as pd

from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai


def get_all_mahasiswa(id=False):
    """
    Query mahasiswa from database, then change it into pandas dataframe.
    :param id = boolean:
    determine whether need id column or not
    :return pandas.DataFrame without Object Mahasiswa, and id column:
    """
    id_feature = ['id']
    selected_features = ['Nama', 'TS', 'KS', 'JK', 'GO', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPK']
    mahasiswa = Mahasiswa.query.join(Nilai, Mahasiswa.id == Nilai.id_mahasiswa).add_columns(Mahasiswa.id,
                                                                                            Mahasiswa.name,
                                                                                            Mahasiswa.school_type,
                                                                                            Mahasiswa.gender,
                                                                                            Mahasiswa.school_city,
                                                                                            Mahasiswa.parent_salary,
                                                                                            Nilai.semester_1,
                                                                                            Nilai.semester_2,
                                                                                            Nilai.semester_3,
                                                                                            Nilai.semester_4,
                                                                                            Nilai.ipk) \
        .filter(Mahasiswa.lulus == "false").all()
    df = pd.DataFrame(mahasiswa)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty and not id:
        return pd.DataFrame(mahasiswa, columns=selected_features)
    elif df.empty and id:
        return pd.DataFrame(mahasiswa, columns=id_feature + selected_features)
    # if data frame not empty
    else:
        if not id:
            return df.drop(['Mahasiswa', 'id'], axis=1)
        else:
            return df.drop('Mahasiswa', axis=1)
