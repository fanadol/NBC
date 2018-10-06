from flask_login import UserMixin

from NBC import db, flask_bcrypt, login_manager


class User(db.Model, UserMixin):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(100), nullable=False)
    hash_password = db.Column(db.String(100), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    registered_on = db.Column(db.DateTime, nullable=False)
    phone_number = db.Column(db.String(25))
    role = db.Column(db.String(5), nullable=False, default=False)

    @property
    def password(self):
        raise AttributeError('password: write-only field')

    @password.setter
    def password(self, password):
        self.hash_password = flask_bcrypt.generate_password_hash(password).decode('utf-8')

    def check_admin(self, role):
        return role == True

    def check_password(self, password):
        return flask_bcrypt.check_password_hash(self.hash_password, password)

    def __repr__(self):
        return "User {}".format(self.email)


# flask login special function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
