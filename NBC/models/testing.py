from NBC import db


class Testing(db.Model):
    __tablename__ = 'testing'

    id = db.Column(db.String(11), primary_key=True)
    tipe_sekolah = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    kota_sekolah = db.Column(db.String(10), nullable=False)
    gaji_ortu = db.Column(db.String(10), nullable=False)
    semester_1 = db.Column(db.String(1), nullable=False)
    semester_2 = db.Column(db.String(1), nullable=False)
    semester_3 = db.Column(db.String(1), nullable=False)
    semester_4 = db.Column(db.String(1), nullable=False)
    ipk = db.Column(db.String(1), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    result = db.relationship('Hasil', backref='testing', lazy=True)
