import unittest

import numpy as np
from lmfit import CompositeModel
from lmfit.models import LinearModel, QuadraticModel, PolynomialModel, GaussianModel, LorentzianModel, \
    PseudoVoigtModel

from peak_prophet_server.data_reader import read_background, read_pattern, read_peaks, read_peak, read_data


class TestDataReader(unittest.TestCase):
    def test_read_linear_background(self):
        input_dict = \
            {'parameters': [
                {'name': 'slope', 'value': 3, 'vary': True, 'min': None, 'max': None},
                {'name': 'intercept', 'value': 1, 'vary': True, 'min': None, 'max': None}
            ],
                'type': 'linear'}
        background_model, background_parameter = read_background(input_dict)
        self.assertIsInstance(background_model, LinearModel)
        self.assertEqual(background_parameter['bkg_slope'].value, 3)
        self.assertEqual(background_parameter['bkg_intercept'].value, 1)

    def test_read_quadratic_background(self):
        input_dict = \
            {'parameters':
                 [{'name': 'a', 'value': 0, 'vary': True, 'min': None, 'max': None},
                  {'name': 'b', 'value': 2, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c', 'value': 3, 'vary': True, 'min': None, 'max': None}],
             'type': 'quadratic'}
        background_model, background_parameter = read_background(input_dict)
        self.assertIsInstance(background_model, QuadraticModel)
        self.assertEqual(background_parameter['bkg_a'].value, 0)
        self.assertEqual(background_parameter['bkg_b'].value, 2)
        self.assertEqual(background_parameter['bkg_c'].value, 3)

    def test_read_polynomial_background_with_3_params(self):
        input_dict = \
            {'parameters':
                 [{'name': 'c0', 'value': 0, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c1', 'value': 2, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c2', 'value': 3, 'vary': True, 'min': None, 'max': None}],
             'type': 'polynomial',
             'degree': 2}
        background_model, background_parameter = read_background(input_dict)
        self.assertIsInstance(background_model, PolynomialModel)

        self.assertEqual(background_parameter['bkg_c0'].value, 0)
        self.assertEqual(background_parameter['bkg_c1'].value, 2)
        self.assertEqual(background_parameter['bkg_c2'].value, 3)

    def test_read_polynomial_background_with_7_params(self):
        input_dict = \
            {'parameters':
                 [{'name': 'c0', 'value': 0, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c1', 'value': 1, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c2', 'value': 2, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c3', 'value': 3, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c4', 'value': 4, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c5', 'value': 5, 'vary': True, 'min': None, 'max': None},
                  {'name': 'c6', 'value': 6, 'vary': True, 'min': None, 'max': None}],
             'type': 'polynomial',
             'degree': 6}
        background_model, background_parameter = read_background(input_dict)
        self.assertIsInstance(background_model, PolynomialModel)

        for i in range(7):
            self.assertEqual(background_parameter[f'bkg_c{i}'].value, i)

    def test_read_pattern(self):
        input_dict = \
            {'name': 'test_pattern',
             'x': [1, 2, 3, 4, 5],
             'y': [1, 2, 3, 4, 5]}
        pattern = read_pattern(input_dict)
        self.assertEqual(pattern.x, [1, 2, 3, 4, 5])
        self.assertEqual(pattern.y, [1, 2, 3, 4, 5])

    def test_read_peaks(self):
        input_dict = \
            [{"type": "Gaussian",
              "parameters": [
                  {"name": "center", "value": 1, 'vary': True, 'min': None, 'max': None},
                  {"name": "fwhm", "value": 0.5, 'vary': True, 'min': None, 'max': None},
                  {"name": "amplitude", "value": 10, 'vary': True, 'min': None, 'max': None}],
              },
             {"type": "Lorentzian",
              "parameters": [
                  {"name": "center", "value": 3, 'vary': True, 'min': None, 'max': None},
                  {"name": "fwhm", "value": 1, 'vary': False, 'min': None, 'max': None},
                  {"name": "amplitude", "value": 10, 'vary': True, 'min': None, 'max': None}],
              },
             {"type": "PseudoVoigt",
              "parameters": [
                  {"name": "center", "value": 6, 'vary': True, 'min': None, 'max': None},
                  {"name": "fwhm", "value": 2, 'vary': True, 'min': None, 'max': None},
                  {"name": "amplitude", "value": 10, 'vary': True, 'min': None, 'max': None},
                  {"name": "fraction", "value": 0.3, 'vary': True, 'min': None, 'max': None}],
              },
             ]

        peaks, parameters = read_peaks(input_dict)
        self.assertEqual(len(peaks), 3)
        self.assertIsInstance(peaks[0], GaussianModel)
        self.assertIsInstance(peaks[1], LorentzianModel)
        self.assertEqual(parameters[1]['p1_fwhm'].vary, False)
        self.assertIsInstance(peaks[2], PseudoVoigtModel)
        self.assertEqual(len(parameters), 3)

    def test_read_gaussian_peak(self):
        input_dict = \
            {"type": "Gaussian",
             "parameters": [
                 {"name": "center", "value": 1, 'vary': True, 'min': None, 'max': None},
                 {"name": "fwhm", "value": 0.5, 'vary': True, 'min': None, 'max': 500},
                 {"name": "amplitude", "value": 10, 'vary': True, 'min': 0, 'max': None}]
             }
        peak, parameters = read_peak(input_dict, prefix='test_')
        self.assertIsInstance(peak, GaussianModel)
        self.assertEqual(parameters['test_center'].value, 1)
        self.assertTrue(np.isclose(parameters['test_fwhm'].value, 0.5))
        self.assertEqual(parameters['test_amplitude'].value, 10)
        self.assertEqual(parameters['test_fwhm'].min, -np.inf)
        self.assertEqual(parameters['test_fwhm'].max, 500)
        self.assertEqual(parameters['test_amplitude'].min, 0)

    def test_read_lorentzian_peak(self):
        input_dict = \
            {"type": "Lorentzian",
             "parameters": [
                 {"name": "center", "value": 1, 'vary': True, 'min': None, 'max': None},
                 {"name": "fwhm", "value": 0.5, 'vary': True, 'min': None, 'max': None},
                 {"name": "amplitude", "value": 10, 'vary': True, 'min': None, 'max': None}]
             }
        peak, parameters = read_peak(input_dict, prefix='lor_')
        self.assertIsInstance(peak, LorentzianModel)
        self.assertEqual(parameters['lor_center'].value, 1)
        self.assertEqual(parameters['lor_fwhm'].value, 0.5)
        self.assertEqual(parameters['lor_amplitude'].value, 10)

    def test_read_pseudovoigt_peak(self):
        input_dict = \
            {"type": "PseudoVoigt",
             "parameters": [
                 {"name": "center", "value": 1, 'vary': True, 'min': None, 'max': None},
                 {"name": "fwhm", "value": 0.5, 'vary': True, 'min': None, 'max': None},
                 {"name": "amplitude", "value": 10, 'vary': True, 'min': None, 'max': None},
                 {"name": "fraction", "value": 0.5, 'vary': True, 'min': None, 'max': None}]
             }
        peak, parameters = read_peak(input_dict, prefix='pv_')
        self.assertIsInstance(peak, PseudoVoigtModel)
        self.assertEqual(parameters['pv_center'].value, 1)
        self.assertEqual(parameters['pv_fwhm'].value, 0.5)
        self.assertEqual(parameters['pv_amplitude'].value, 10)
        self.assertEqual(parameters['pv_fraction'].value, 0.5)

    def test_read_gaussian_peak_with_fixed_parameters(self):
        input_dict = \
            {"type": "Gaussian",
             "parameters": [
                 {"name": "center", "value": 1, "vary": False, 'min': None, 'max': None},
                 {"name": "fwhm", "value": 0.5, "vary": False, 'min': None, 'max': None},
                 {"name": "amplitude", "value": 10, "vary": True, 'min': None, 'max': None}]
             }
        peak, parameters = read_peak(input_dict, prefix='test_')
        self.assertIsInstance(peak, GaussianModel)
        self.assertEqual(parameters['test_center'].value, 1)
        self.assertEqual(parameters['test_center'].vary, False)
        self.assertTrue(np.isclose(parameters['test_fwhm'].value, 0.5))
        self.assertEqual(parameters['test_fwhm'].vary, False)
        self.assertEqual(parameters['test_amplitude'].value, 10)
        self.assertEqual(parameters['test_amplitude'].vary, True)

    def test_read_data(self):
        input_dict = \
            {'name': 'test_data',
             'peaks': [{"type": "Gaussian",
                        "parameters": [
                            {"name": "center", "value": 1, 'vary': True, 'min': None, 'max': None},
                            {"name": "fwhm", "value": 0.5, 'vary': True, 'min': None, 'max': None},
                            {"name": "amplitude", "value": 10, 'vary': True, 'min': None, 'max': None}],
                        },
                       {"type": "Gaussian",
                        "parameters": [
                            {"name": "center", "value": 3, 'vary': True, 'min': None, 'max': None},
                            {"name": "fwhm", "value": 1, 'vary': True, 'min': None, 'max': None},
                            {"name": "amplitude", "value": 10, 'vary': True, 'min': None, 'max': None}],
                        },
                       ],
             'background': {'type': 'linear',
                            'parameters': [{'name': 'intercept', 'value': 0.5, 'vary': True, 'min': None, 'max': None},
                                           {'name': 'slope', 'value': 1, 'vary': True, 'min': None, 'max': None}]},
             'pattern': {'name': 'test_pattern',
                         'x': [1, 2, 3, 4, 5],
                         'y': [1, 2, 3, 4, 5]}
             }

        pattern, model, params = read_data(input_dict)

        self.assertEqual(pattern.x, [1, 2, 3, 4, 5])
        self.assertEqual(pattern.y, [1, 2, 3, 4, 5])
        self.assertIsInstance(model, CompositeModel)
        self.assertIsInstance(model.left.left, LinearModel)
        self.assertIsInstance(model.left.right, GaussianModel)
        self.assertIsInstance(model.right, GaussianModel)
        self.assertEqual(len(params.valuesdict()), 12)
        params_dict = params.valuesdict()
        self.assertEqual(params_dict['bkg_intercept'], 0.5)
        self.assertEqual(params_dict['bkg_slope'], 1)
        self.assertEqual(params_dict['p0_center'], 1)
        self.assertEqual(params_dict['p1_center'], 3)
