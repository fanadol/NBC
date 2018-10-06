import pandas as pd

from NBC import db
from NBC.models.training import Training


def delete_all_training():
    Training.query.delete()
    db.session.commit()


def get_a_training(id):
    return Training.query.filter_by(nim=id).first()


def get_all_training():
    selected_features = ['NIM', 'school_type', 'gender', 'school_city', 'parent_salary', 'semester_1', 'semester_2',
                         'semester_3', 'semester_4', 'ipk', 'ket_lulus']
    datas = Training.query.all()
    frame = []
    for data in datas:
        frame.append({'id': data.id,
                      'NIM': data.nim,
                      'school_type': data.school_type,
                      'gender': data.gender,
                      'school_city': data.school_city,
                      'parent_salary': data.parent_salary,
                      'semester_1': data.semester_1,
                      'semester_2': data.semester_2,
                      'semester_3': data.semester_3,
                      'semester_4': data.semester_4,
                      'ipk': data.ipk,
                      'ket_lulus': data.ket_lulus})
    df = pd.DataFrame(frame)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty:
        return pd.DataFrame([], columns=selected_features)
    return df[selected_features]


def update_a_training(obj, updatedObj):
    obj.school_type = updatedObj['school_type']
    obj.gender = updatedObj['gender']
    obj.school_city = updatedObj['school_city']
    obj.parent_salary = updatedObj['parent_salary']
    obj.semester_1 = updatedObj['semester_1']
    obj.semester_2 = updatedObj['semester_2']
    obj.semester_3 = updatedObj['semester_3']
    obj.semester_4 = updatedObj['semester_4']
    obj.ipk = updatedObj['ipk']
    obj.ket_lulus = updatedObj['ket_lulus']
    db.session.commit()