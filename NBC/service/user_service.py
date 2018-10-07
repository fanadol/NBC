import pandas as pd

from NBC import db
from NBC.models.user import User


def get_an_user(id):
    return User.query.filter_by(id=id).first()


def get_all_users(id=False):
    """
    Query user from database, then change it into pandas dataframe.
    :param id: bool
    determine whether need id column or not
    :return pandas.DataFrame without Object Users:
    """
    id_feature = ['id']
    selected_features = ['email', 'first_name', 'last_name', 'registered_on', 'phone_number', 'role']
    users = User.query.all()
    frame = []
    if users:
        for user in users:
            frame.append({'id': user.id,
                          'email': user.email,
                          'hash_password': user.hash_password,
                          'first_name': user.first_name,
                          'last_name': user.last_name,
                          'registered_on': user.registered_on,
                          'phone_number': user.phone_number,
                          'role': user.role})
    df = pd.DataFrame(frame)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty and not id:
        return pd.DataFrame(frame, columns=selected_features)
    elif df.empty and id:
        return pd.DataFrame(frame, columns=id_feature + selected_features)
    # if data frame not empty
    else:
        if not id:
            return df[selected_features]
        else:
            return df[id_feature + selected_features]


def update_an_user(obj, updatedObj):
    obj.email = updatedObj['email']
    obj.first_name = updatedObj['first_name']
    obj.last_name = updatedObj['last_name']
    obj.phone_number = updatedObj['phone_number']
    obj.admin = updatedObj['admin']
    db.session.commit()


def delete_an_user(email):
    User.query.filter_by(email=email).delete()
    db.session.commit()
