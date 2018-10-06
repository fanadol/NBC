from NBC import db
from NBC.models.nilai import Nilai


def get_a_nilai(id):
    return Nilai.query.filter_by(id_alumni=id).first()

def update_a_nilai(obj, updatedObj):
    obj.semester_1 = updatedObj['semester_1']
    obj.semester_2 = updatedObj['semester_2']
    obj.semester_3 = updatedObj['semester_3']
    obj.semester_4 = updatedObj['semester_4']
    obj.ipk = updatedObj['ipk']
    db.session.commit()