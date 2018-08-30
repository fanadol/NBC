from NBC import db


class Alumni(db.Model):
    __tablename__ = 'alumni'

    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    id_mahasiswa = db.Column(db.Integer, db.ForeignKey('mahasiswa.id'))
    keterangan_lulus = db.Column(db.String(20), nullable=False)
    testing = db.relationship('Training', backref='alumni', lazy=True)