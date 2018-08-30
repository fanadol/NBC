from NBC import db


class Training(db.Model):
    __tablename__ = 'data_training'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id_alumni = db.Column(db.Integer, db.ForeignKey('alumni.id'))
