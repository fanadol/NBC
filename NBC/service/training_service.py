import pandas as pd

from NBC import db
from NBC.models.training import Training


def delete_all_training():
    Training.query.delete()
    db.session.commit()


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
