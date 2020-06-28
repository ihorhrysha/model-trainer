import os
# Patern folder
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(object):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'flask_dev_main.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    RESTPLUS_SWAGGER_UI_DOC_EXPANSION = 'list'
    RESTPLUS_VALIDATE = True
    RESTPLUS_MASK_SWAGGER = False
    RESTPLUS_ERROR_404_HELP = False

    FLASK_DEBUG = True  # Do not use debug mode in production

    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GBQ_CRED_PATH') or \
        os.path.join(basedir, 'bigquery.cred.json')
