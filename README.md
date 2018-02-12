# tensorflow-rnn-server
Let's use flask to serve some RNN time-series model


## Installation

`pip3 install flask importlib`

## Development Server launch

#### On Linux
`export FLASK_APP=app.py && flask run`

## Production Server launch

#### On Ubuntu
`/usr/local/bin/gunicorn app:app`

## Tests

in the root of the project:

`python3 src/rnn_time_series_server_tests.py`
