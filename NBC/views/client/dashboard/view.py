from flask import render_template

from . import dashboard
from NBC.views.models.alumni import Alumni
from NBC.views.models.mahasiswa import Mahasiswa
from NBC.views.models.nilai import Nilai
from NBC.views.models.user import User


@dashboard.route('/')
def index():
    return render_template('dashboard.html')


@dashboard.route('/users', methods=["GET", "POST"])
def users():
    users = User.query.all()
    return render_template('users.html', users=users)


@dashboard.route('/alumni', methods=["GET"])
def alumni():
    alumni = Alumni.query.join(Mahasiswa, Alumni.id_mahasiswa == Mahasiswa.id).join(Nilai,
                                                                                    Mahasiswa.id == Nilai.id_mahasiswa) \
        .add_columns(Alumni.id_mahasiswa, Mahasiswa.name, Mahasiswa.school_type, Mahasiswa.gender,
                     Mahasiswa.school_city,
                     Mahasiswa.parent_salary, Nilai.semester_1, Nilai.semester_2, Nilai.semester_3,
                     Nilai.semester_4, Nilai.ipk, Alumni.keterangan_lulus) \
        .all()
    return render_template('alumni.html', alumni=alumni)


@dashboard.route('/mahasiswa', methods=["GET", "POST"])
def mahasiswa():
    mahasiswa = Mahasiswa.query.join(Nilai, Mahasiswa.id == Nilai.id_mahasiswa).add_columns(Mahasiswa.id,
                                                                                            Mahasiswa.name,
                                                                                            Mahasiswa.school_type,
                                                                                            Mahasiswa.gender,
                                                                                            Mahasiswa.school_city,
                                                                                            Mahasiswa.parent_salary,
                                                                                            Mahasiswa.lulus,
                                                                                            Nilai.semester_1,
                                                                                            Nilai.semester_2,
                                                                                            Nilai.semester_3,
                                                                                            Nilai.semester_4,
                                                                                            Nilai.ipk)\
                                                                                            .filter(Mahasiswa.lulus==False).all()
    return render_template('mahasiswa.html', mahasiswa=mahasiswa)
