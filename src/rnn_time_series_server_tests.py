import unittest
import requests
import rnn_time_series_server
import os

class RNNTimeSeriesServerTestRequests(unittest.TestCase):

    def test_random_response(self):
        response = requests.get('http://localhost:5000/random')
        self.assertIs(response.data, int)


class RNNTimeSeriesServerTestCore(unittest.TestCase):

    def test_model_importing(self):
        stubResponse = rnn_time_series_server.load_model_from_path(os.path.dirname(os.path.abspath(__file__))+'/stub_module.py', 'stub_module', 'justAStubFunctionForATest')
        self.assertEqual(stubResponse, 1234)

if __name__ == "__main__":
    unittest.main()