
### Init project(dev)
```bash
python -m venv venv

source venv/bin/activate

make install

python manage.py db upgrade

make run
```

goto http://localhost:5000/api/