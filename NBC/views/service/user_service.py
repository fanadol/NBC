import pandas as pd

from NBC.views.models.user import User


def get_all_users(id=False):
    """
    Query user from database, then change it into pandas dataframe.
    :param id = boolean:
    determine whether need id column or not
    :return pandas.DataFrame without Object Users:
    """
    id_feature = ['id']
    selected_features = ['public_id', 'username', 'first_name', 'last_name', 'registered_on', 'phone_number',
                         'admin']
    user = User.query.all()
    df = pd.DataFrame(user)
    # create empty data frame with columns for DataFrame.to_html()
    if df.empty and not id:
        return pd.DataFrame(user, columns=selected_features)
    elif df.empty and id:
        return pd.DataFrame(user, columns=id_feature + selected_features)
    # if data frame not empty
    else:
        if not id:
            return df.drop(['id', 'hash_password'], axis=1)
        else:
            return df.drop('hash_password', axis=1)
