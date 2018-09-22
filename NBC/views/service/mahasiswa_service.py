import pandas as pd

from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai


def get_all_mahasiswa():
    """
    Query mahasiswa from database, then change it into pandas dataframe.
    :return pandas.DataFrame without Object Mahasiswa, and id column:
    """
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
    return df.drop(['Mahasiswa', 'id'], axis=1)

def get_mahasiswa():
    """
    Query mahasiswa from database, then change it into pandas dataframe.
    :return pandas.DataFrame without Object Mahasiswa, and id column:
    """
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
    return mahasiswa