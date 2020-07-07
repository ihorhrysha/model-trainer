from flask import Blueprint, redirect


# building the API
bp = Blueprint('front', __name__, template_folder='templates')

@bp.route('/')
@bp.route('/index')
def index():
    return redirect('api')
