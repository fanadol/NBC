import pandas as pd

from NBC.views.models.alumni import Alumni
from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai


def get_all_alumni():
    alumni = Alumni.query.join(Mahasiswa, Alumni.id_mahasiswa == Mahasiswa.id).join(Nilai,
                                                                                    Mahasiswa.id == Nilai.id_mahasiswa) \
        .add_columns(Alumni.id, Alumni.id_mahasiswa, Mahasiswa.name, Mahasiswa.school_type, Mahasiswa.gender,
                     Mahasiswa.school_city,
                     Mahasiswa.parent_salary, Nilai.semester_1, Nilai.semester_2, Nilai.semester_3,
                     Nilai.semester_4, Nilai.ipk, Alumni.keterangan_lulus) \
        .all()
    return alumni


def get_pandas_alumni():
    alumni = get_all_alumni()

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
        'ipk': [],
        'keterangan_lulus': []
    }

    for data in alumni:
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
        if data.keterangan_lulus.lower() == "tepat waktu":
            raw_data['keterangan_lulus'] += [1]
        elif data.keterangan_lulus.lower() == 'tidak tepat waktu':
            raw_data['keterangan_lulus'] += [0]
        else:
            return "Error, wrong data of data"

    df = pd.DataFrame(raw_data, columns=['id', 'school_type', 'gender', 'school-city', 'parent_salary', 'semester_1',
                                         'semester_2', 'semester_3', 'semester_4', 'ipk', 'keterangan_lulus'])

    return df
