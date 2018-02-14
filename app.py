from flask  import Flask, request, jsonify, session
from src    import rnn_time_series_server as rnn

import numpy as np
import tensorflow as tf


app = Flask(__name__)
app.debug = True
app.secret_key = 'development'

sess      = tf.Session()
new_saver = tf.train.import_meta_graph('/tmp/time_series/run-20180213203103-9-GRU_300_2_10_5/model_final_9.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('/tmp/time_series/run-20180213203103-9-GRU_300_2_10_5'))
X_placeholder             = tf.get_default_graph().get_tensor_by_name("X:0")
outputs_op                = tf.get_default_graph().get_tensor_by_name("outputs:0")
initial_state_placeholder = tf.get_default_graph().get_tensor_by_name("initial_state_placeholder:0")
is_training_placeholder   = tf.get_default_graph().get_tensor_by_name("is_training:0")
keep_prob                 = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
final_state_op            = tf.get_default_graph().get_tensor_by_name("rnn/while/Exit_3:0")


all_vars = tf.get_collection('vars')
for v in all_vars:
    print(v)

print(initial_state_placeholder)

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
        prediction = rnn.make_prediction(sess, outputs_op, initial_state_placeholder,
                                         X_placeholder, is_training_placeholder, keep_prob, final_state_op)

        print(prediction)
        return str(prediction)


    except Exception as error:
        print(error)
        return str(error)
    
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
    
