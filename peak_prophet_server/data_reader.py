import numpy as np

from lmfit.models import LinearModel, QuadraticModel, PolynomialModel, GaussianModel, LorentzianModel, PseudoVoigtModel

from peak_prophet_server.pattern import Pattern


def read_bkg(bkg_dict):
    parameter_values = {p['name']: p['value'] for p in bkg_dict['parameters']}

    match bkg_dict['type']:
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
            model = PolynomialModel(degree=bkg_dict['degree'], prefix='bkg_')
            params = model.make_params()
            for i in range(bkg_dict['degree'] + 1):
                params[f'bkg_c{i}'].set(value=parameter_values[f'c{i}'])
            return model, params
        case _:
            raise ValueError('Unknown background type')


def read_pattern(pattern_dict):
    return Pattern(x=pattern_dict['x'], y=pattern_dict['y'])


def read_peaks(peaks_list):
    peaks = []
    parameters = []
    for i, peak_dict in enumerate(peaks_list):
        peak, peak_params = read_peak(peak_dict, f'p{i}_')
        parameters.append(peak_params)
        peaks.append(peak)
    return peaks, parameters


def read_peak(peak_dict, prefix=''):
    parameter_values = {p['name']: p['value'] for p in peak_dict['parameters']}

    match peak_dict['type']:
        case 'Gaussian':
            model = GaussianModel(prefix=prefix)
            params = model.make_params(amplitude=parameter_values['Amplitude'],
                                       center=parameter_values['Position'],
                                       sigma=convert_fwhm_to_sigma(parameter_values['FWHM']))
            return model, params
        case 'Lorentzian':
            model = LorentzianModel(prefix=prefix)
            params = model.make_params(amplitude=parameter_values['Amplitude'],
                                       center=parameter_values['Position'],
                                       sigma=parameter_values['FWHM'] * 0.5)
            return model, params
        case 'PseudoVoigt':
            model = PseudoVoigtModel(prefix=prefix)
            params = model.make_params(amplitude=parameter_values['Amplitude'],
                                       center=parameter_values['Position'],
                                       sigma=parameter_values['FWHM']/2,
                                       fraction=parameter_values['Eta'])
            return model, params


def convert_fwhm_to_sigma(fwhm):
    return fwhm / (2 * (2 * np.log(2)) ** 0.5)
