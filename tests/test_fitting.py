import unittest
import json

import numpy as np
from lmfit.models import LinearModel, GaussianModel, LorentzianModel, PseudoVoigtModel

from peak_prophet_server.data_reader import convert_fwhm_to_sigma
from peak_prophet_server.fitting import FitManager


class TestFitting(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.num_points = 501
        self.pattern_x = np.linspace(0, 10, self.num_points)
        self.error = 0.1
        self.error_array = np.random.normal(0, self.error, self.num_points)
        self.bkg_dict = {
            'type': 'linear',
            'parameters': [
                {'name': 'intercept', 'value': 1},
                {'name': 'slope', 'value': 0.2}
            ]
        }

    async def test_fitting_single_gaussian(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = GaussianModel(prefix='p0_')
        params.update(peak1_model.make_params(amplitude=10, center=2, sigma=convert_fwhm_to_sigma(0.2)))

        model = background_model + peak1_model

        pattern_y = model.eval(params, x=self.pattern_x) + self.error_array

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': self.pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 2.2},
                        {'name': 'fwhm', 'value': 0.2}
                    ]
                },
            ],

            'background': self.bkg_dict
        }

        fit_result = await self.fit(input_dict)

        self.compare_background_results(fit_result['background'])

        expected_peak_data = [
            {
                'type': 'gaussian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 2},
                    {'name': 'fwhm', 'value': 0.2}
                ]
            },
        ]

        self.compare_peak_results(fit_result['peaks'], expected_peak_data)

    async def test_fitting_single_lorentzian(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = LorentzianModel(prefix='p0_')
        params.update(peak1_model.make_params(amplitude=10, center=2, sigma=0.2 * 0.5))
        model = background_model + peak1_model

        pattern_y = model.eval(params, x=self.pattern_x) + self.error_array

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': self.pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'lorentzian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 2.2},
                        {'name': 'fwhm', 'value': 0.2}
                    ]
                },
            ],

            'background': self.bkg_dict
        }

        fit_result = await self.fit(input_dict)

        self.compare_background_results(fit_result['background'])

        expected_peak_data = [
            {
                'type': 'lorentzian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 2},
                    {'name': 'fwhm', 'value': 0.2}
                ]
            },
        ]

        self.compare_peak_results(fit_result['peaks'], expected_peak_data)

    async def test_fitting_single_pseudovoigt(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = PseudoVoigtModel(prefix='p0_')
        params.update(peak1_model.make_params(
            amplitude=10,
            center=2,
            sigma=0.2 * 0.5,
            fraction=0.3))

        model = background_model + peak1_model

        pattern_y = model.eval(params, x=self.pattern_x) + self.error_array

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': self.pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'pseudovoigt',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 2.2},
                        {'name': 'fwhm', 'value': 0.2},
                        {'name': 'fraction', 'value': 0.5}
                    ]
                },
            ],
            'background': self.bkg_dict
        }

        fit_result = await self.fit(input_dict)

        self.compare_background_results(fit_result['background'])

        expected_peak_data = [
            {
                'type': 'pseudovoigt',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 2},
                    {'name': 'fwhm', 'value': 0.2},
                    {'name': 'fraction', 'value': 0.3}
                ]
            },
        ]

        self.compare_peak_results(fit_result['peaks'], expected_peak_data)

    async def test_fit_two_gaussians(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = GaussianModel(prefix='p0_')
        params.update(peak1_model.make_params(amplitude=10, center=2, sigma=convert_fwhm_to_sigma(0.2)))

        peak2_model = GaussianModel(prefix='p1_')
        params.update(peak2_model.make_params(amplitude=10, center=5, sigma=convert_fwhm_to_sigma(0.3)))

        model = background_model + peak1_model + peak2_model

        pattern_y = model.eval(params, x=self.pattern_x) + self.error_array

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': self.pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 2.2},
                        {'name': 'fwhm', 'value': 0.2}
                    ]
                },
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 5.1},
                        {'name': 'fwhm', 'value': 0.3}
                    ]
                },
            ],

            'background': self.bkg_dict
        }

        fit_result = await self.fit(input_dict)

        self.compare_background_results(fit_result['background'])

        expected_peak_data = [
            {
                'type': 'gaussian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 2},
                    {'name': 'fwhm', 'value': 0.2}
                ]
            }, {
                'type': 'gaussian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 5},
                    {'name': 'fwhm', 'value': 0.3}
                ]
            },
        ]

        self.compare_peak_results(fit_result['peaks'], expected_peak_data)

    async def test_fit_gaussian_and_lorentzian(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = GaussianModel(prefix='p0_')
        params.update(peak1_model.make_params(amplitude=10, center=2, sigma=convert_fwhm_to_sigma(0.2)))

        peak2_model = LorentzianModel(prefix='p1_')
        params.update(peak2_model.make_params(amplitude=10, center=5, sigma=0.3 * 0.5))

        model = background_model + peak1_model + peak2_model

        pattern_y = model.eval(params, x=self.pattern_x) + self.error_array

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': self.pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 2.2},
                        {'name': 'fwhm', 'value': 0.2}
                    ]
                },
                {
                    'type': 'lorentzian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 5.1},
                        {'name': 'fwhm', 'value': 0.3}
                    ]
                },
            ],

            'background': self.bkg_dict
        }

        fit_result = await self.fit(input_dict)

        self.compare_background_results(fit_result['background'])

        expected_peak_data = [
            {
                'type': 'gaussian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 2},
                    {'name': 'fwhm', 'value': 0.2}
                ]
            }, {
                'type': 'lorentzian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 5},
                    {'name': 'fwhm', 'value': 0.3}
                ]
            },
        ]

        self.compare_peak_results(fit_result['peaks'], expected_peak_data)

    async def test_fit_gaussian_and_pseudovoigt(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = GaussianModel(prefix='p0_')
        params.update(peak1_model.make_params(amplitude=10, center=2, sigma=convert_fwhm_to_sigma(0.2)))

        peak2_model = PseudoVoigtModel(prefix='p1_')
        params.update(peak2_model.make_params(amplitude=10, center=5, sigma=0.3 * 0.5, fraction=0.3))

        model = background_model + peak1_model + peak2_model

        pattern_y = model.eval(params, x=self.pattern_x) + self.error_array

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': self.pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 2.2},
                        {'name': 'fwhm', 'value': 0.2}
                    ]
                },
                {
                    'type': 'pseudovoigt',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 5.1},
                        {'name': 'fwhm', 'value': 0.3},
                        {'name': 'fraction', 'value': 0.4}
                    ]
                },
            ],

            'background': self.bkg_dict
        }

        fit_result = await self.fit(input_dict)

        self.compare_background_results(fit_result['background'])

        expected_peak_data = [
            {
                'type': 'gaussian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 2},
                    {'name': 'fwhm', 'value': 0.2}
                ]
            }, {
                'type': 'pseudovoigt',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 5},
                    {'name': 'fwhm', 'value': 0.3},
                    {'name': 'fraction', 'value': 0.3}
                ]
            },
        ]

        self.compare_peak_results(fit_result['peaks'], expected_peak_data)

    async def test_fit_gaussian_and_lorentzian_and_pseudovoigt(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = GaussianModel(prefix='p0_')
        params.update(peak1_model.make_params(amplitude=10, center=2, sigma=convert_fwhm_to_sigma(0.2)))

        peak2_model = LorentzianModel(prefix='p1_')
        params.update(peak2_model.make_params(amplitude=10, center=5, sigma=0.4 * 0.5))

        peak3_model = PseudoVoigtModel(prefix='p2_')
        params.update(peak3_model.make_params(amplitude=10, center=8, sigma=0.3 * 0.5, fraction=0.3))

        model = background_model + peak1_model + peak2_model + peak3_model

        pattern_y = model.eval(params, x=self.pattern_x) + self.error_array

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': self.pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 2.2},
                        {'name': 'fwhm', 'value': 0.2}
                    ]
                },
                {
                    'type': 'lorentzian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 5},
                        {'name': 'fwhm', 'value': 0.2}
                    ]
                },
                {
                    'type': 'pseudovoigt',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': 8},
                        {'name': 'fwhm', 'value': 0.3},
                        {'name': 'fraction', 'value': 0.4}
                    ]
                },
            ],

            'background': self.bkg_dict
        }

        fit_result = await self.fit(input_dict)

        self.compare_background_results(fit_result['background'])

        expected_peak_data = [
            {
                'type': 'gaussian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 2},
                    {'name': 'fwhm', 'value': 0.2}
                ]
            },
            {
                'type': 'lorentzian',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 5},
                    {'name': 'fwhm', 'value': 0.4}
                ]
            },
            {
                'type': 'pseudovoigt',
                'parameters': [
                    {'name': 'amplitude', 'value': 10},
                    {'name': 'center', 'value': 8},
                    {'name': 'fwhm', 'value': 0.3},
                    {'name': 'fraction', 'value': 0.3}
                ]
            },
        ]

        self.compare_peak_results(fit_result['peaks'], expected_peak_data)

    async def test_failing_fit(self):
        background_model = LinearModel(prefix='bkg_')
        params = background_model.make_params(intercept=1, slope=0.2)

        peak1_model = GaussianModel(prefix='p0_')
        params.update(peak1_model.make_params(amplitude=3, center=1, sigma=convert_fwhm_to_sigma(0.2)))

        peak2_model = LorentzianModel(prefix='p1_')
        params.update(peak2_model.make_params(amplitude=7, center=5, sigma=0.4 * 0.5))

        peak3_model = PseudoVoigtModel(prefix='p2_')
        params.update(peak3_model.make_params(amplitude=5, center=8, sigma=0.3 * 0.5, fraction=0.3))

        pattern_model = background_model + peak1_model + peak2_model + peak3_model

        pattern_y = pattern_model.eval(params, x=self.pattern_x) + self.error_array

        input_dict = {
            'pattern': {
                'name': 'test',
                'x': self.pattern_x.tolist(),
                'y': pattern_y.tolist()
            },
            'peaks': [
                {
                    'type': 'gaussian',
                    'parameters': [
                        {'name': 'amplitude', 'value': 10.5},
                        {'name': 'center', 'value': i * 0.5},
                        {'name': 'fwhm', 'value': 0.2}
                    ]
                } for i in range(30)],
            'background': self.bkg_dict
        }

        import asyncio

        input_request = json.dumps(input_dict)
        fit_manager = FitManager("TEST-SID")
        loop = asyncio.get_event_loop()

        def stop_fit():
            print("Stopping fit")
            fit_manager.stop = True

        loop.call_later(1, stop_fit)
        fit_response = await fit_manager.process_request(input_request)

        self.assertEqual(fit_response['success'], False)

    async def fit(self, input_dict):
        input_request = json.dumps(input_dict)

        fit_manager = FitManager("TEST-SID")
        fit_response = await fit_manager.process_request(input_request)

        fit_result = fit_response['result']
        self.assertTrue(fit_response['success'])
        self.assertEqual(fit_response['message'], 'Fit succeeded.')

        return fit_result

    def compare_peak_results(self, peak_result, expected_peak_data):
        self.assertEqual(len(peak_result), len(expected_peak_data))
        for i, peak in enumerate(peak_result):
            self.assertEqual(peak['type'], expected_peak_data[i]['type'])
            for j, param in enumerate(peak['parameters']):
                self.assertEqual(param['name'], expected_peak_data[i]['parameters'][j]['name'])
                self.assertAlmostEqual(param['value'], expected_peak_data[i]['parameters'][j]['value'], delta=0.1)

    def compare_background_results(self, background_result):
        self.assertEqual(background_result['type'], 'linear')
        self.assertEqual(background_result['parameters'][0]['name'], 'intercept')
        self.assertAlmostEqual(background_result['parameters'][0]['value'], 1, delta=0.1)
        self.assertEqual(background_result['parameters'][1]['name'], 'slope')
        self.assertAlmostEqual(background_result['parameters'][1]['value'], 0.2, delta=0.01)
