from flask import Flask, request, jsonify, session
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
    values = [int(x) for x in request.args.get('observation').split(',')]
    if 'observations' not in session:
        session['observations'] = [values]
    elif len(session['observations']) < 5:
        session['observations'].append(values)
    elif len(session['observations']) == 5:
        # DO SOME ML MAGIC
        return jsonify(session.pop('observations'))
    
    # return """we are {} observations away from updating the model; or something""".format(5-len(session['observations']))
    return str(0)
    
@app.route('/test_parsing_input')
def prase_test():
    try:
        values = [float(x) if '.' in x else int(x) for x in request.args.get('observation').split(',')]
    except Exception as e:
        print(e)
        return "malformed input"
    else:    
        return jsonify(values)

if __name__ == '__main__':
    app.run()
    
