import os
import unittest

from app import create_app

from app.database import db, model, task

from flask_migrate import Migrate
# from flask_script import Manager

app = create_app()
app.app_context().push()

# manager = Manager(app)
migrate = Migrate(app, db)


@app.cli.group()
def utils():
    """Custom utils commands."""
    pass


@utils.command()
def test():
    """Runs the unit tests."""
    tests = unittest.TestLoader().discover('app/test', pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful():
        return 0
    return 1


if __name__ == '__main__':
    app.run()
