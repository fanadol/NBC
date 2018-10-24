from NBC import db


class Testing(db.Model):
    __tablename__ = 'testing'

    id = db.Column(db.String(11), primary_key=True)
    school_type = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    school_city = db.Column(db.String(10), nullable=False)
    semester_1 = db.Column(db.Float, nullable=False)
    semester_2 = db.Column(db.Float, nullable=False)
    semester_3 = db.Column(db.Float, nullable=False)
    semester_4 = db.Column(db.Float, nullable=False)
    ipk = db.Column(db.Float, nullable=False)
    result = db.relationship('Hasil', backref='testing', lazy=True, uselist=False)
