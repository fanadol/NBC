from NBC import db
from NBC.models.training import Training


def delete_all_training():
    Training.query.delete()
    db.session.commit()


def get_all_training():
    return Training.query.all()
