from NBC import db


class Mahasiswa(db.Model):
    __tablename__ = 'mahasiswa'

    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    school_type = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    school_city = db.Column(db.String(10), nullable=False)
    parent_salary = db.Column(db.String(10), nullable=False)
    lulus = db.Column(db.Boolean, default=False)
    nilai = db.relationship('Nilai', backref='mahasiswa', lazy=True)
    alumnus = db.relationship('Alumni', backref='mahasiswa', lazy=True)
    testing = db.relationship('Testing', backref='mahasiswa', lazy=True)
