import unittest
import requests
import rnn_time_series_server
import os

class RNNTimeSeriesServerTestRequests(unittest.TestCase):


    def test_parsing_input(self):
        response = requests.get('http://localhost:5000/prediction?observation=10,-10,0,100,9999,0.9999')
        self.assertIs(response.data, int)


    def test_parsing_input(self):
        response = requests.get('http://localhost:5000/test_parsing_input?observation=10,-10,0,100,9999,0.9999')
        self.assertEqual(response.data, [10, -10, 0, 9999, 0.9999])


    def test_random_response(self):
        response = requests.get('http://localhost:5000/random')
        self.assertIs(response.data, int)


class RNNTimeSeriesServerTestCore(unittest.TestCase):

    def test_model_importing(self):
        stubResponse = rnn_time_series_server.load_model_from_path(os.path.dirname(os.path.abspath(__file__))+'/stub_module.py', 'stub_module', 'justAStubFunctionForATest')
        self.assertEqual(stubResponse, 1234)

if __name__ == "__main__":
    unittest.main()

