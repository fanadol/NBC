from NBC import db


class Training(db.Model):
    __tablename__ = 'training'

    id = db.Column(db.String(11), primary_key=True)
    school_type = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    school_city = db.Column(db.String(10), nullable=False)
    semester_1 = db.Column(db.Float, nullable=False)
    semester_2 = db.Column(db.Float, nullable=False)
    semester_3 = db.Column(db.Float, nullable=False)
    semester_4 = db.Column(db.Float, nullable=False)
    ipk = db.Column(db.Float, nullable=False)
    ket_lulus = db.Column(db.String(20), nullable=False)
