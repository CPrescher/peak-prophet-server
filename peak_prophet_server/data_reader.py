import numpy as np

from lmfit.models import LinearModel, QuadraticModel, PolynomialModel, GaussianModel, LorentzianModel, PseudoVoigtModel
from lmfit import Parameters

from peak_prophet_server.pattern import Pattern


def read_data(data_dict):
    """
    Read the data from the input dictionary and return the pattern, model and parameters
    :param data_dict: the input dictionary containing the pattern, peaks and background
    :return: pattern, model, parameters
    :rtype: (Pattern, Model, Parameters)
    """
    pattern = read_pattern(data_dict['pattern'])
    peaks, peaks_parameters = read_peaks(data_dict['peaks'])
    bkg_model, bkg_params = read_background(data_dict['background'])
    model = bkg_model
    for peak in peaks:
        model += peak

    params = bkg_params
    for peak_params in peaks_parameters:
        params.update(peak_params)

    return pattern, model, params


def read_background(background_dict):
    """
    Read the background from the input dictionary and return the model and parameters
    :param background_dict: dictionary containing the background type and parameters
    :return: background model, background parameters
    :rtype: (Model, Parameters)
    """
    parameter_values = {p['name']: p['value'] for p in background_dict['parameters']}

    match background_dict['type']:
        case 'linear':
            model = LinearModel(prefix='bkg_')
            params = model.make_params(intercept=parameter_values['intercept'],
                                       slope=parameter_values['slope'])
            return model, params
        case 'quadratic':
            model = QuadraticModel(prefix='bkg_')
            params = model.make_params(a=parameter_values['a'],
                                       b=parameter_values['b'],
                                       c=parameter_values['c'])
            return model, params
        case 'polynomial':
            model = PolynomialModel(degree=background_dict['degree'], prefix='bkg_')
            params = model.make_params()
            for i in range(background_dict['degree'] + 1):
                params[f'bkg_c{i}'].set(value=parameter_values[f'c{i}'])
            return model, params
        case _:
            raise ValueError('Unknown background type')


def read_pattern(pattern_dict):
    """
    Read the pattern from the input dictionary.
    :param pattern_dict: dictionary containing the pattern x and y values
    :return: the extracted pattern
    :rtype: Pattern
    """
    return Pattern(x=pattern_dict['x'], y=pattern_dict['y'])


def read_peaks(peaks_list):
    """
    Read the peaks from the input dictionary peak part.
    :param peaks_list:
    :return: list of peak models, list of parameters
    :rtype: (list[Model], list[Parameters])
    """
    peaks = []
    parameters = []
    for i, peak_dict in enumerate(peaks_list):
        peak, peak_params = read_peak(peak_dict, f'p{i}_')
        parameters.append(peak_params)
        peaks.append(peak)
    return peaks, parameters


def read_peak(peak_dict, prefix=''):
    """
    Read a single peak from the input dictionary.
    :param peak_dict: dictionary containing the peak type and parameters
    :param prefix: used for the lmfit model prefix for this particular pexis (e.g. p0_), prevents name clashes
    :return: peak model, parameters
    :rtype: (Model, Parameters)
    """
    parameter_values = {p['name']: p['value'] for p in peak_dict['parameters']}

    match peak_dict['type'].lower():
        case 'gaussian':
            model = GaussianModel(prefix=prefix)
            params = model.make_params(amplitude=parameter_values['amplitude'],
                                       center=parameter_values['center'],
                                       sigma=convert_fwhm_to_sigma(parameter_values['fwhm']))
            return model, params
        case 'lorentzian':
            model = LorentzianModel(prefix=prefix)
            params = model.make_params(amplitude=parameter_values['amplitude'],
                                       center=parameter_values['center'],
                                       sigma=parameter_values['fwhm'] * 0.5)
            return model, params
        case 'pseudovoigt':
            model = PseudoVoigtModel(prefix=prefix)
            params = model.make_params(amplitude=parameter_values['amplitude'],
                                       center=parameter_values['center'],
                                       sigma=parameter_values['fwhm'] / 2,
                                       fraction=parameter_values['fraction'])
            return model, params
        case _:
            raise ValueError(f'Unknown peak type: {peak_dict["type"]}')


def convert_fwhm_to_sigma(fwhm):
    return fwhm / (2 * (2 * np.log(2)) ** 0.5)
