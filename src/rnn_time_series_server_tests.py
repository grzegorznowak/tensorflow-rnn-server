import unittest
import requests
import rnn_time_series_server as rnn
import os
import numpy as np
from numpy.testing import assert_array_equal

class RNNTimeSeriesServerTestRequests(unittest.TestCase):


    def test_predict(self):
        response = requests.get('http://localhost:5000/prediction?observation=10,-10,0,100,9999,9999,1')
        self.assertIs(type(response.json()), int)


    def test_parsing_input(self):
        response = requests.get('http://localhost:5000/echo?observation=10,-10,0,100,9999,0.9999,1')
        self.assertEqual(response.json(), [10, -10, 0, 100, 9999, 0.9999,1])


    def test_failsafe_when_next_observation_isnt_actually_next(self):
        requests.get('http://localhost:5000/prediction?observation=10,-10,0,100,9999,0.9999,1')
        response1 = requests.get('http://localhost:5000/prediction?observation=10,-10,0,100,9999,0.9999,3')  # this should be 2, 3 means we skipped a data
        self.assertEqual(response1, "error: control check didn't pass")

        requests.get('http://localhost:5000/prediction?observation=10,-10,0,100,9999,0.9999,4')
        response2 = requests.get('http://localhost:5000/prediction?observation=10,-10,0,100,9999,0.9999,1')  # this should be 5
        self.assertEqual(response2, "error: control check didn't pass")

        requests.get('http://localhost:5000/prediction?observation=10,-10,0,100,9999,0.9999,2')
        response3 = requests.get('http://localhost:5000/prediction?observation=10,-10,0,100,9999,0.9999,0')  # this is actually OK, since new day
        self.assertNotEqual(response3, "error: control check didn't pass")
        self.assertIs(type(response3.json()), int)


    def test_random_response(self):
        response = requests.get('http://localhost:5000/random')
        self.assertIs(type(response.json()), int)


class RNNTimeSeriesServerTestCore(unittest.TestCase):

    def test_model_importing(self):
        stubResponse = rnn.load_module_method_from_path(os.path.dirname(os.path.abspath(__file__))+'/stub_module.py', 'stub_module', 'justAStubFunctionForATest')
        self.assertEqual(stubResponse(), 1234)

    def test_observation_parsing(self):
        result = rnn.raw_observation_to_list("10,-10,0,100,9999,0.9999,1")
        self.assertEqual(result, [10, -10, 0, 100, 9999, 0.9999,1])

    def test_unpacking_observation_data(self):
        unpacked_labels = rnn.unpack_labels([rnn.ObservationData([1,1,1,1,1,0.1, 1]), rnn.ObservationData([1,1,1,1,1,0.1, 2])])
        assert_array_equal(unpacked_labels, np.array([[1,1,1,1,1,0.1], [1,1,1,1,1,0.1]]))

    def test_filling_batch(self):
        output1 = rnn.maybe_fill_batch_with_sparse_vectors(np.array([1,1,1,1,1,0.1]), 5, 6)
        assert_array_equal(output1, np.array([[1,1,1,1,1,0.1], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]))

        output2 = rnn.maybe_fill_batch_with_sparse_vectors(np.array([[1,1,1,1,1,0.1], [1,1,1,1,1,0.1]]), 5, 6)
        assert_array_equal(output2, np.array([[1,1,1,1,1,0.1], [1,1,1,1,1,0.1], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]))

        output3 = rnn.maybe_fill_batch_with_sparse_vectors(np.array([[1,1,1,1,1,0.1], [1,1,1,1,1,0.1], [1,1,1,1,1,0.1], [1,1,1,1,1,0.1], [1,1,1,1,1,0.1]]), 5, 6)
        assert_array_equal(output3, np.array([[1,1,1,1,1,0.1], [1,1,1,1,1,0.1], [1,1,1,1,1,0.1], [1,1,1,1,1,0.1], [1,1,1,1,1,0.1]]))

if __name__ == "__main__":
    unittest.main()

