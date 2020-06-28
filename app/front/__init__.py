from flask import Blueprint, render_template


# building the API
routes = Blueprint('front', __name__, template_folder='templates')


@routes.route('/')
@routes.route('/index')
def index():
    user = {'username': 'Ihor'}
    return render_template('index.html', title='Home', user=user)
