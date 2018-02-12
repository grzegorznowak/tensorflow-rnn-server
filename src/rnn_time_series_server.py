import os

# TODO: this is just a stub of a proper backend storage (which we don't actually need at this stage).
# all we need is a quick turnaround with a working PoC that will show us how good/bad evaluations from the model are
database       = []
next_rnn_state = None

BATCH_SIZE = 5
OBSERVATIONS_COUNT = 6

class ObservationData():
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


# TODO: just a stub, need to populate
zero_state     = load_module_method_from_path(os.path.dirname(os.path.abspath(__file__))+'/stub_module.py', 'stub_module', 'justAStubFunctionForATest')


def raw_observation_to_list(observations_string_raw):
    return [float(x) if '.' in x else int(x) for x in observations_string_raw.split(',')]


def is_it_a_new_day(observation):
    return observation.control == 0


def append_observation_to_db(observation):
    global database
    global next_rnn_state

    if is_it_a_new_day(observation):
        database = [observation]
        next_rnn_state = zero_state()

    else:
        previous_observation = db_last_element()
        validate_observations(previous_observation, observation)


    # TODO: 5 must be configurable somewhere sometime
    if len(databasae) == BATCH_SIZE:
        # after every 5 batches, flush the database and update the state used to calculate responses against
        prediction, next_rnn_state = make_prediction()
        database = []

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


"""Need to get a feel of the data, if they make sense in general"""
def validate_observations(observation_prev, observation_next):
    if observation_prev.control > observation_next.control:
        raise Exception('Wrong ordering of observation data')
    if observation_next.control > observation_prev.control+1:
        raise Exception('Observation data skipped an entry')


"""Calculate model response using data from db"""
def make_prediction(sess):

    labels = maybe_fill_batch_with_sparse_vectors(unpack_labels(get_db()), BATCH_SIZE)

    output = 10; #stub
    state = [1, 1, 1] #stub

    return output, state

"""Adds extra zero vectors up to batch_size"""
def maybe_fill_batch_with_sparse_vectors(batch, batch_size):
    if len(batch) < batch_size:
        return batch

    return batch


"""convert observation data back to arrays for convenience of the model"""
def unpack_labels(observationDataList):
    return np.array(map(lambda d: [d.open, d.high, d.low, d.close, d.volume, d.time], observationDataList))

# def make_a_prediction(previous_step_data):
#     prediction_batch = create_a_data_batch(next_step_tensor)
#     next_rnn_state   = get_next_rnn_state(previous_step_data)