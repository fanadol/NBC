import pandas as pd

from NBC.models.alumni import Alumni
from NBC.models.nilai import Nilai


def get_all_alumni(id=False):
    """
    Query alumni from database, then change it into pandas dataframe.
    :param id = boolean:
    determine whether need id column or not
    :return pandas.DataFrame without Object Alumni, and id column:
    """
    id_feature = ['id']
    selected_features = ['TS', 'JK', 'KS', 'GO', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPK', 'keterangan_lulus']
    alumni = Alumni.query.join(Alumni, Alumni.id_Alumni == Alumni.id).join(Nilai,
                                                                           Alumni.id == Nilai.id_Alumni) \
        .add_columns(Alumni.id, Alumni.school_type, Alumni.gender, Alumni.school_city,
                     Alumni.parent_salary, Nilai.semester_1, Nilai.semester_2, Nilai.semester_3,
                     Nilai.semester_4, Nilai.ipk, Alumni.keterangan_lulus).all()
    df = pd.DataFrame(alumni)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty and not id:
        return pd.DataFrame(alumni, columns=selected_features)
    elif df.empty and id:
        return pd.DataFrame(alumni, columns=id_feature + selected_features)
    # if data frame not empty
    else:
        if not id:
            return df.drop(['Alumni', 'id'], axis=1)
        else:
            return df.drop(['Alumni'], axis=1)
