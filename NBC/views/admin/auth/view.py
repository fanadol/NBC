import datetime
from flask import render_template, redirect, flash, url_for, request
from flask_login import login_user, login_required, logout_user

from . import auth
from NBC.models.user import User
from NBC.service.database_service import save_to_db


@auth.route('/admin/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        first_name = request.form.get('firstname')
        last_name = request.form.get('lastname')
        phone = request.form.get('phone')
        if not email or not password or not first_name or not last_name:
            flash('Please fill all form!', 'danger')
            return redirect(url_for('dashboard.create_user'))
        if phone:
            phone = '+62' + phone
        data = User(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            registered_on=datetime.datetime.utcnow(),
            phone_number=phone,
            role='Admin'
        )
        save_to_db(data)
        flash('User berhasil dibuat', 'success')
        return redirect(url_for('auth.login'))
    return render_template('register.html')


@auth.route('/admin/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            if user.check_admin(user.role):
                login_user(user)
                flash('You successfully logged in', 'success')
                return redirect(url_for('dashboard.alumni'))
            else:
                flash('Sorry, you are not admin, please login from user page!', 'danger')
                return redirect(url_for('auth.login'))
        else:
            flash('Username or password does not match.', 'danger')
            return redirect(url_for('auth.login'))
    return render_template('login.html')


@auth.route('/admin/logout')
@login_required
def logout():
    logout_user()
    flash('Successfully logged out', 'success')
    return redirect(url_for('auth.login'))
