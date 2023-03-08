import unittest
import json

import numpy as np
from lmfit.models import LinearModel, GaussianModel

from peak_prophet_server.fitting import FitManager


class TestFitting(unittest.TestCase):
    def test_fitting(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = GaussianModel(prefix='p0_')
        params.update(peak1_model.make_params(amplitude=10, center=2, sigma=0.3))

        peak2_model = GaussianModel(prefix='p1_')
        params.update(peak2_model.make_params(amplitude=10, center=4, sigma=0.3))

        model = background_model + peak1_model + peak2_model

        num_points = 501
        pattern_x = np.linspace(0, 10, num_points)
        pattern_y = model.eval(params, x=pattern_x) + np.random.normal(0, 0.2, num_points)

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'Amplitude', 'value': 9},
                        {'name': 'Center', 'value': 2.2},
                        {'name': 'FWHM', 'value': 0.3}
                    ]
                },
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'Amplitude', 'value': 10.5},
                        {'name': 'Center', 'value': 3.9},
                        {'name': 'FWHM', 'value': 0.1}
                    ]
                }
            ],
            'background': {
                'type': 'linear',
                'parameters': [
                    {'name': 'intercept', 'value': 1},
                    {'name': 'slope', 'value': 0.2}
                ]
            }
        }

        input_request = json.dumps(input_dict)

        fit_manager = FitManager()
        fit_manager.process_request(input_request)
