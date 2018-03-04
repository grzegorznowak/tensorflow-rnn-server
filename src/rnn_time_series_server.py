import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# TODO: this is just a stub of a proper backend storage (which we don't actually need at this stage).
# all we need is a quick turnaround with a working PoC that will show us how good/bad evaluations from the model are
database       = []
next_rnn_state = None
scaler         = None
scaling_range  = 60

BATCH_SIZE         = 62
OBSERVATIONS_COUNT = 5
STEPS_COUNT        = 1
RNN_NEURONS        = 200  # this is a property of a model, need to be variablized somehow - rather hard, tbd.
RNN_LAYERS         = 1    # this is a property of a model, need to be variablized somehow - rather hard, tbd.
#
# class ObservationData:
#     def __init__(self, list):
#         self.open    = list[0]
#         self.high    = list[1]
#         self.low     = list[2]
#         self.close   = list[3]
#         self.volume  = list[4]
#         self.time    = list[5]
#         self.control = list[6]
#

def load_module_method_from_path(absolutePath, moduleName, nameOfTheFactoryMethod):
    import importlib.util
    spec = importlib.util.spec_from_file_location(moduleName, absolutePath)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    method_to_call = getattr(foo, nameOfTheFactoryMethod)
    return method_to_call


def zero_state(initial_state_op_c, initial_state_op_h):
    return tuple(initial_state_op_c, initial_state_op_h)
  #  return np.zeros((BATCH_SIZE, RNN_NEURONS * RNN_LAYERS))
   # return np.random.randn(BATCH_SIZE, RNN_NEURONS * RNN_LAYERS)


def raw_observation_to_list(observations_string_raw):
    return [float(x) if '.' in x else int(x) for x in observations_string_raw.split(',')]


def is_it_a_new_day(observation):
    return get_control_value_from_observation(observation) == 0


# def append_observation_to_db(observation, initial_state_op_c, initial_state_op_h):
#     global database
#     global next_rnn_state
#
#     if is_it_a_new_day(observation) or next_rnn_state is None:
#         database = []
#         next_rnn_state = zero_state(initial_state_op_c, initial_state_op_h)
#
#     else:
#         previous_observation = db_last_element()
#         if previous_observation is not None:  # which is possible if we went onto a new batch
#             validate_observations(previous_observation, observation)
#
#
#     database.append(observation)
#     return database


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

    if get_control_value_from_observation(observation_prev) > get_control_value_from_observation(observation_next):
        raise Exception('Wrong ordering of observation data')
    if get_control_value_from_observation(observation_next) != get_control_value_from_observation(observation_prev) + 1:
        raise Exception('Observation data skipped an entry')


"""Calculate model response using data from db"""
def make_prediction(current_observation, sess, outputs_op, initial_state_placeholder,
                    X_placeholder, is_training_placeholder, keep_prob, final_state_op_c, final_state_op_h):

    global next_rnn_state
    global database
    global scaler
    global scaling_range

    def execute_rnn(observation, initial_state):
        Xs  = maybe_fill_batch_with_sparse_vectors(observation, BATCH_SIZE, STEPS_COUNT, OBSERVATIONS_COUNT)
        if initial_state is None:
            outputs, new_states_c, new_states_h = sess.run([outputs_op, final_state_op_c, final_state_op_h], feed_dict={X_placeholder: Xs,
                                                                                keep_prob: 1,
                                                                                is_training_placeholder:False})
        else:
            outputs, new_states_c, new_states_h = sess.run([outputs_op, final_state_op_c, final_state_op_h], feed_dict={X_placeholder: Xs,
                                                                                    keep_prob: 1,
                                                                                    initial_state_placeholder: initial_state,
                                                                                    is_training_placeholder:False})
        print("outputs: ",outputs[0][0])
        #TODO: BATCH_SIZE must be configurable somewhere sometime
        #reprime RNN state every BATCH_SIZE full data batches


        return [[new_states_c, new_states_h]], outputs[0][0].argsort()[-1:][::-1]
        #return  np.argmax(outputs[0][0])


    if is_it_a_new_day(current_observation):
        database       = []
        next_rnn_state = None
        scaler         = None
        return 0
    else:
        if len(database) > 0:
            validate_observations(database[-1], current_observation)  # do not validate if a first sample for a day

        database.append(current_observation)
        if len(database) == scaling_range:
            observations = np.array(drop_control_values(database))
            scaler = MinMaxScaler((-1, 1))
            scaler.fit(observations)
            print(current_observation)
            print(scaler.transform(drop_control_values([current_observation])))
            database           = database[1:]  # drop first element so we scale over sliding window of `scaling_range` elements
            next_state, result = execute_rnn(scaler.transform(drop_control_values([current_observation]))[0], next_rnn_state)
            next_rnn_state     = next_state
            return result
        else: # not enough data in db to setup scaler and do any predictions
            return 0



"""return the same list without control values in it"""
def drop_control_values(list):
    return [list_item[:-1] for list_item in list]

"""Adds extra zero vectors up to batch_size"""
def maybe_fill_batch_with_sparse_vectors(batch, batch_size, steps_count, observations_count):
    out_batch = batch.copy()
    out_batch.resize((batch_size, steps_count, observations_count))
    return out_batch


def get_control_value_from_observation(observation):
    return observation[-1]
# """convert observation data back to arrays for convenience of the model"""
# def unpack_labels(observationDataList):
#     return np.array(list(map(lambda d: [d.open, d.high, d.low, d.close, d.volume, d.time], observationDataList)))

# def make_a_prediction(previous_step_data):
#     prediction_batch = create_a_data_batch(next_step_tensor)
#     next_rnn_state   = get_next_rnn_state(previous_step_data)