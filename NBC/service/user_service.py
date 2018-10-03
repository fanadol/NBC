import pandas as pd

from NBC.models.user import User


def get_all_users(id=False):
    """
    Query user from database, then change it into pandas dataframe.
    :param id = boolean:
    determine whether need id column or not
    :return pandas.DataFrame without Object Users:
    """
    id_feature = ['id']
    selected_features = ['public_id', 'email', 'first_name', 'last_name', 'registered_on', 'phone_number',
                         'admin']
    users = User.query.all()
    frame = []
    if users:
        for user in users:
            frame.append({'id': user.id,
                          'public_id': user.public_id,
                          'email': user.email,
                          'hash_password': user.hash_password,
                          'first_name': user.first_name,
                          'last_name': user.last_name,
                          'registered_on': user.registered_on,
                          'phone_number': user.phone_number,
                          'admin': user.admin})
    df = pd.DataFrame(frame)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty and not id:
        return pd.DataFrame(frame, columns=selected_features)
    elif df.empty and id:
        return pd.DataFrame(frame, columns=id_feature + selected_features)
    # if data frame not empty
    else:
        if not id:
            return df.drop(['id', 'hash_password'], axis=1)
        else:
            return df.drop('hash_password', axis=1)
