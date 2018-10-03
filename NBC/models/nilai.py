from NBC import db


class Nilai(db.Model):
    __tablename__ = 'nilai'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id_alumni = db.Column(db.String(11), db.ForeignKey('alumni.id'))
    semester_1 = db.Column(db.String(1), nullable=False)
    semester_2 = db.Column(db.String(1), nullable=False)
    semester_3 = db.Column(db.String(1), nullable=False)
    semester_4 = db.Column(db.String(1), nullable=False)
    ipk = db.Column(db.String(1), nullable=False)
