from flask import Blueprint

examples = Blueprint('examples', __name__)

from . import view
