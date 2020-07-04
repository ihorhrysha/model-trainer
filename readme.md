
[![Build Status](https://travis-ci.org/ihorhrysha/model-trainer.svg?branch=master)](https://travis-ci.org/ihorhrysha/model-trainer)

### Init project(dev)
```bash
python -m venv venv

source venv/bin/activate

make install

export FLASK_APP=manage.py

flask db upgrade
```


### Install redis
##### For Ubuntu([due to this tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04-ru})):
Install redis 
```bash
sudo apt update
sudo apt install redis-server
```
Change conf file (supervised no -> supervised systemd)
```bash
sudo nano /etc/redis/redis.conf

...
# If you run Redis from upstart or systemd, Redis can interact with your
# supervision tree. Options:
#   supervised no      - no supervision interaction
#   supervised upstart - signal upstart by putting Redis into SIGSTOP mode
#   supervised systemd - signal systemd by writing READY=1 to $NOTIFY_SOCKET
#   supervised auto    - detect upstart or systemd method based on
#                        UPSTART_JOB or NOTIFY_SOCKET environment variables
# Note: these supervision methods only signal "process is ready."
#       They do not enable continuous liveness pings back to your supervisor.
supervised systemd
...
```
From the root of the project start a worker:
```bash
rq worker training-tasks
```
To see existing redis keys (queues, jobs, workers etc.) launch redis-cli 
```bash
redis-cli
> keys "*"
```


### Run project

```bash
flask run
```

goto http://localhost:5000/api/