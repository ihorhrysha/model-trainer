
[![Build Status](https://travis-ci.org/ihorhrysha/model-trainer.svg?branch=master)](https://travis-ci.org/ihorhrysha/model-trainer)

### Init project(dev)
```bash
python -m venv venv

source venv/bin/activate

make install

export FLASK_APP=manage.py

flask db upgrade

flask run
```

goto http://localhost:5000/api/