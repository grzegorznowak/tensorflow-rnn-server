import os
import numpy as np

# TODO: this is just a stub of a proper backend storage (which we don't actually need at this stage).
# all we need is a quick turnaround with a working PoC that will show us how good/bad evaluations from the model are
database       = []
next_rnn_state = None

BATCH_SIZE         = 10
OBSERVATIONS_COUNT = 6
STEPS_COUNT        = 5
RNN_NEURONS        = 300  # this is a property of a model, need to be variablized somehow - rather hard, tbd.
RNN_LAYERS         = 2    # this is a property of a model, need to be variablized somehow - rather hard, tbd.

class ObservationData:
    def __init__(self, list):
        self.open    = list[0]
        self.high    = list[1]
        self.low     = list[2]
        self.close   = list[3]
        self.volume  = list[4]
        self.time    = list[5]
        self.control = list[6]


def load_module_method_from_path(absolutePath, moduleName, nameOfTheFactoryMethod):
    import importlib.util
    spec = importlib.util.spec_from_file_location(moduleName, absolutePath)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    method_to_call = getattr(foo, nameOfTheFactoryMethod)
    return method_to_call


def zero_state():
    return np.random.rand(BATCH_SIZE, RNN_NEURONS * RNN_LAYERS)


def raw_observation_to_list(observations_string_raw):
    return [float(x) if '.' in x else int(x) for x in observations_string_raw.split(',')]


def is_it_a_new_day(observation):
    return observation.control == 0


def append_observation_to_db(observation):
    global database
    global next_rnn_state

    if is_it_a_new_day(observation) or next_rnn_state is None:
        database = []
        next_rnn_state = zero_state()

    else:
        previous_observation = db_last_element()
        if previous_observation is not None:  # which is possible if we went onto a new batch
            validate_observations(previous_observation, observation)


    database.append(observation)
    return database


def db_last_element():
    global database
    if not database:
        return None
    else:
        return database[-1]


def get_db():
    global database
    return database


def flush_db():
    global database
    database = []

"""Need to get a feel of the data, if they make sense in general"""
def validate_observations(observation_prev, observation_next):
    print(observation_prev.control)
    print(observation_next.control)
    if observation_prev.control > observation_next.control:
        raise Exception('Wrong ordering of observation data')
    if observation_next.control != observation_prev.control + 1:
        raise Exception('Observation data skipped an entry')


"""Calculate model response using data from db"""
def make_prediction(sess, outputs_op, initial_state_placeholder,
                    X_placeholder, is_training_placeholder, keep_prob, final_state_op):

    global next_rnn_state
    db = get_db()
    db_len = len(db)

    labels  = maybe_fill_batch_with_sparse_vectors(unpack_labels(get_db()), BATCH_SIZE, STEPS_COUNT, OBSERVATIONS_COUNT)

    outputs, new_states = sess.run([outputs_op, final_state_op], feed_dict={X_placeholder: labels,
                                               keep_prob: 1,
                                               initial_state_placeholder: next_rnn_state,
                                               is_training_placeholder:False})


    # TODO: BATCH_SIZE must be configurable somewhere sometime
    # reprime RNN state every BATCH_SIZE full data batches
    if db_len == STEPS_COUNT:

        next_rnn_state = new_states
        flush_db()
        print("db flushed")

    return  round(np.transpose(outputs[0])[0][db_len - 1], 0)


"""Adds extra zero vectors up to batch_size"""
def maybe_fill_batch_with_sparse_vectors(batch, batch_size, steps_count, observations_count):
    out_batch = batch.copy()
    out_batch.resize((batch_size, steps_count, observations_count))
    return out_batch


"""convert observation data back to arrays for convenience of the model"""
def unpack_labels(observationDataList):
    return np.array(list(map(lambda d: [d.open, d.high, d.low, d.close, d.volume, d.time], observationDataList)))

# def make_a_prediction(previous_step_data):
#     prediction_batch = create_a_data_batch(next_step_tensor)
#     next_rnn_state   = get_next_rnn_state(previous_step_data)