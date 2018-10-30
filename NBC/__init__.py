from flask import Flask, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt

from NBC.config import config_by_name

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please login first before accessing this page.'
login_manager.login_message_category = 'danger'
flask_bcrypt = Bcrypt()


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    db.init_app(app)
    login_manager.init_app(app)
    flask_bcrypt.init_app(app)

    @app.route('/')
    def index():
        return redirect(url_for('dashboard.alumni'))

    from NBC.views.admin.dashboard import dashboard as dashboard_bp
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')

    from NBC.views.admin.auth import auth as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    return app
