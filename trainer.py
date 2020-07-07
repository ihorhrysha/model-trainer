from app import create_app, cli

app = create_app()
cli.register(app)