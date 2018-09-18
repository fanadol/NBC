from NBC import db


class Testing(db.Model):
    __tablename__ = 'data_testing'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id_mahasiswa = db.Column(db.Integer, db.ForeignKey('mahasiswa.id'))
    hasil = db.Column(db.String(20), nullable=False)
