import pandas as pd
from sklearn_pandas import gen_features

from NBC.views.models.alumni import Alumni
from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai


def get_all_alumni():
    alumni = Alumni.query.join(Mahasiswa, Alumni.id_mahasiswa == Mahasiswa.id).join(Nilai,
                                                                                    Mahasiswa.id == Nilai.id_mahasiswa) \
        .add_columns(Alumni.id_mahasiswa, Mahasiswa.name, Mahasiswa.school_type, Mahasiswa.gender,
                     Mahasiswa.school_city,
                     Mahasiswa.parent_salary, Nilai.semester_1, Nilai.semester_2, Nilai.semester_3,
                     Nilai.semester_4, Nilai.ipk, Alumni.keterangan_lulus) \
        .all()
    # clean the noise in the data
    # for i in range(len(alumni)):
    #     alumni[i] = alumni[i][1:]
    return alumni


def get_pandas_alumni():
    alumni = get_all_alumni()

    raw_data = {
        'school_type': [],
        'gender': [],
        'school-city': [],
        'parent_salary': [],
        'semester_1': [],
        'semester_2': [],
        'semester_3': [],
        'semester_4': [],
        'ipk': [],
        'keterangan_lulus': []
    }

    for data in alumni:
        raw_data['school_type'] += [data.school_type]
        raw_data['gender'] += [data.gender]
        raw_data['school-city'] += [data.school_city]
        raw_data['parent_salary'] += [data.parent_salary]
        raw_data['semester_1'] += [data.semester_1]
        raw_data['semester_2'] += [data.semester_2]
        raw_data['semester_3'] += [data.semester_3]
        raw_data['semester_4'] += [data.semester_4]
        raw_data['ipk'] += [data.ipk]
        raw_data['keterangan_lulus'] += [data.keterangan_lulus]

    df = pd.DataFrame(raw_data, columns=['school_type', 'gender', 'school-city', 'parent_salary', 'semester_1',
                                         'semester_2', 'semester_3', 'semester_4', 'ipk', 'keterangan_lulus'])

    return df

# # determine all the features
# school_type_features = []
# gender_features = []
# school_city_features = []
# parent_salary_features = []
# semester_1_features = []
# semester_2_features = []
# semester_3_features = []
# semester_4_features = []
# ipk_features = []
# keterangan_lulus_features = []
#
# for data in alumni:
#     school_type_features.append(data.school_type)
#     gender_features.append(data.gender)
#     school_city_features.append(data.school_city)
#     parent_salary_features.append(data.parent_salary)
#     semester_1_features.append(data.semester_1)
#     semester_2_features.append(data.semester_2)
#     semester_3_features.append(data.semester_3)
#     semester_4_features.append(data.semester_4)
#     ipk_features.append(data.ipk)
#     keterangan_lulus_features.append(data.keterangan_lulus)