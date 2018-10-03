from NBC import db


class Hasil(db.Model):
    __tablename__ = 'hasil'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id_testing = db.Column(db.String(11), db.ForeignKey('testing.id'))
    result = db.Column(db.String(20), nullable=False)
