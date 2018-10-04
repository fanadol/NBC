from NBC import db


class Alumni(db.Model):
    __tablename__ = 'alumni'

    id = db.Column(db.String(11), primary_key=True)
    school_type = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    school_city = db.Column(db.String(10), nullable=False)
    parent_salary = db.Column(db.String(20), nullable=False)
    ket_lulus = db.Column(db.String(20), nullable=False)
    nilai = db.relationship('Nilai', backref='alumni', lazy=True)
