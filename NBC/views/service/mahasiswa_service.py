import pandas as pd

from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai


def get_all_mahasiswa():
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

def get_pandas_mahasiswa():
    mhs = get_all_mahasiswa()

    raw_data = {
        'id': [],
        'school_type': [],
        'gender': [],
        'school_city': [],
        'parent_salary': [],
        'semester_1': [],
        'semester_2': [],
        'semester_3': [],
        'semester_4': [],
        'ipk': []
    }

    for data in mhs:
        raw_data['id'] += [data.id]
        raw_data['school_type'] += [data.school_type]
        raw_data['gender'] += [data.gender]
        raw_data['school_city'] += [data.school_city]
        raw_data['parent_salary'] += [data.parent_salary]
        raw_data['semester_1'] += [data.semester_1]
        raw_data['semester_2'] += [data.semester_2]
        raw_data['semester_3'] += [data.semester_3]
        raw_data['semester_4'] += [data.semester_4]
        raw_data['ipk'] += [data.ipk]

    df = pd.DataFrame(raw_data, columns=['id', 'school_type', 'gender', 'school-city', 'parent_salary', 'semester_1',
                                         'semester_2', 'semester_3', 'semester_4', 'ipk'])
    return df
