import asyncio
import json

from .data_reader import read_data


class FitManager:
    data_dict = None
    current_progress = None
    result = None

    def __init__(self, sio=None):
        self.sio = sio

    async def process_request(self, request):
        self.data_dict = json.loads(request)
        pattern, model, params = read_data(self.data_dict)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.fit, pattern, model, params)
        out = self.result

        background_result = create_background_output(self.data_dict['background'], out.params)
        peaks_result = create_peaks_output(self.data_dict['peaks'], out.params)

        return {
            'success': out.success,
            'message': 'Fitting successful',
            'result': {
                'background': background_result,
                'peaks': peaks_result,
                'chi2': out.chisqr,
                'redchi': out.redchi,
                'nfev': out.nfev,
            }
        }

    def fit(self, pattern, model, params):
        self.result = model.fit(pattern.y, params, x=pattern.x, iter_cb=self.iter_cb)

    def iter_cb(self, params, iter, resid, *args, **kwargs):
        print("iter_cb: ", iter)
        if self.sio is None:
            print("sio is None")
            return

        self.current_progress = {
            'iter': iter,
            'resid': resid.tolist(),
            'result': {
                'background': create_background_output(self.data_dict['background'], params),
                'peaks': create_peaks_output(self.data_dict['peaks'], params),
            }
        }


def create_background_output(background_input, params):
    output = {
        'type': background_input['type'],
        'parameters': []
    }
    for param in background_input['parameters']:
        output['parameters'].append({
            'name': param['name'],
            'value': params[f'bkg_{param["name"]}'].value,
            'error': params[f'bkg_{param["name"]}'].stderr
        })
    return output


def create_peaks_output(peaks_input, params):
    output = []
    for i, peak in enumerate(peaks_input):
        output.append({
            'type': peak['type'],
            'parameters': []
        })
        for param in peak['parameters']:
            output[i]['parameters'].append({
                'name': param['name'],
                'value': params[f'p{i}_{param["name"].lower()}'].value,
                'error': params[f'p{i}_{param["name"].lower()}'].stderr
            })
    return output
