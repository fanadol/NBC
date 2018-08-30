from NBC import db


class Hasil(db.Model):
    __tablename__ = 'hasil'

    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    id_testing = db.Column(db.Integer, db.ForeignKey('data_testing.id'))
    hasil = db.Column(db.String(15), nullable=False)
