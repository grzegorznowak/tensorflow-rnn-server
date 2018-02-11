import unittest
import requests


class RNNTimeSeriesServerTestRequests(unittest.TestCase):

    def test_random_response(self):
        response = requests.get('http://localhost:5000/random')
        self.assertIs(response.data, int)


if __name__ == "__main__":
    unittest.main()