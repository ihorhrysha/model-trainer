
import os
import unittest

def register(app):
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