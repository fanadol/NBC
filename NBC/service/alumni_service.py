import pandas as pd

from NBC import db
from NBC.models.alumni import Alumni
from NBC.models.nilai import Nilai


def get_an_alumni(id):
    return Alumni.query.filter_by(id=id).first()


def get_all_alumni(id=False):
    """
    Query alumni from database, then change it into pandas dataframe.
    :param id = boolean:
    determine whether need id column or not
    :return pandas.DataFrame without Object Alumni, and id column:
    """
    id_feature = ['id']
    selected_features = ['school_type', 'gender', 'school_city', 'parent_salary', 'semester_1', 'semester_2',
                         'semester_3', 'semester_4', 'ipk', 'ket_lulus']
    alumni = Alumni.query.join(Nilai, Alumni.id == Nilai.id_alumni) \
        .add_columns(Alumni.id, Alumni.school_type, Alumni.gender, Alumni.school_city,
                     Alumni.parent_salary, Nilai.semester_1, Nilai.semester_2, Nilai.semester_3,
                     Nilai.semester_4, Nilai.ipk, Alumni.ket_lulus).all()
    df = pd.DataFrame(alumni)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty and not id:
        return pd.DataFrame([], columns=selected_features)
    elif df.empty and id:
        return pd.DataFrame([], columns=id_feature + selected_features)
    # if data frame not empty
    else:
        if not id:
            return df[selected_features]
        else:
            return df[id_feature + selected_features]


def update_an_alumni(obj, updatedObj):
    # obj.id = updatedObj['id']
    obj.school_type = updatedObj['school_type']
    obj.gender = updatedObj['gender']
    obj.school_city = updatedObj['school_city']
    obj.parent_salary = updatedObj['parent_salary']
    obj.ket_lulus = updatedObj['ket_lulus']
    db.session.commit()


def delete_an_alumni(id):
    Nilai.query.filter_by(id_alumni=id).delete()
    Alumni.query.filter_by(id=id).delete()
    db.session.commit()


def convert_nilai(ip):
    if ip >= 3.5:
        nilai = 'A'
    elif ip >= 3:
        nilai = 'B'
    elif ip >= 2.5:
        nilai = 'C'
    elif ip >= 2:
        nilai = 'D'
    else:
        nilai = 'E'
    return nilai
