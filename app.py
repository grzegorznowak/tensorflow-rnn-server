from flask  import Flask, request, jsonify, session
from src    import rnn_time_series_server as rnn

import numpy as np


app = Flask(__name__)
app.debug = True
app.secret_key = 'development'

@app.route('/')
def index():
    return "Hello"

@app.route('/random')
def random():
    return str(int(np.random.normal(0.0, 7))) 

@app.route('/prediction')
def predict():
    current_observation  = rnn.ObservationData(rnn.raw_observation_to_list(request.args.get('observation')))

    try:
        rnn.append_observation_to_db(current_observation)
        prediction = rnn.make_prediction()

        return prediction


    except Exception as error:
        return str("error: control check didn't pass")
    
    # return """we are {} observations away from updating the model; or something""".format(5-len(session['observations']))
    
@app.route('/echo')
def echo():
    try:
        values = rnn.raw_observation_to_list(request.args.get('observation'))
    except Exception as e:
        print(e)
        return "malformed input"
    else:    
        return jsonify(values)

if __name__ == '__main__':
    app.run()
    
