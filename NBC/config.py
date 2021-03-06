import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ['NBC_KEY']
    DEBUG = True
    ROOT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
    STATIC_DIRECTORY = os.path.join(ROOT_DIRECTORY, "NBC", "static")
    ALLOWED_EXTENSION = ['csv']
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:jankenpo@localhost:5432/nbc'


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfgi(Config):
    DEBUG = False


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False


config_by_name = dict(
    dev=DevelopmentConfig,
    prod=ProductionConfgi,
    test=TestingConfig
)
