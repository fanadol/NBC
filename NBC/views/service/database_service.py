from NBC import db


def save_to_db(data):
    db.session.add(data)
    db.session.commit()
