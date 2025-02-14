import asyncio
import json
import numpy as np

from .data_reader import read_data


class FitManager:
    data_dict = None
    current_progress = None
    result = None
    stop = False
    pattern = None

    def __init__(self, sid=None):
        self.sid = sid

    async def process_request(self, request):
        self.data_dict = json.loads(request)
        self.pattern, model, params = read_data(self.data_dict)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.fit, self.pattern, model, params)
        out = self.result
        print(self.sid, "fit finished")

        background_result = create_background_output(
            self.data_dict["background"], out.params
        )
        peaks_result = create_peaks_output(self.data_dict["peaks"], out.params)

        return {
            "success": out.success,
            "message": out.message,
            "chi2": out.chisqr,
            "red_chi2": out.redchi,
            "nfev": out.nfev,
            "result": {
                "background": background_result,
                "peaks": peaks_result,
            },
        }

    def fit(self, pattern, model, params):
        self.result = model.fit(pattern.y, params, x=pattern.x, iter_cb=self.iter_cb)

    def iter_cb(self, params, iter, resid, *args, **kwargs):
        if self.sid is None:
            print("sid is None")
            return

        chi2 = np.sum(resid**2)
        red_chi2 = chi2 / (len(self.pattern.y) - 1)

        self.current_progress = {
            "iter": iter,
            "resid": resid.tolist(),
            "chi2": chi2,
            "red_chi2": red_chi2,
            "result": {
                "background": create_background_output(
                    self.data_dict["background"], params
                ),
                "peaks": create_peaks_output(self.data_dict["peaks"], params),
            },
        }
        return self.stop


def create_background_output(background_input, params):
    output = {"type": background_input["type"], "parameters": []}
    for param in background_input["parameters"]:
        output["parameters"].append(
            {
                "name": param["name"],
                "value": params[f'bkg_{param["name"]}'].value,
                "error": params[f'bkg_{param["name"]}'].stderr,
                "vary": params[f'bkg_{param["name"]}'].vary,
            }
        )

    return output


def create_peaks_output(peaks_input, params):
    output = []
    for i, peak in enumerate(peaks_input):
        output.append({"type": peak["type"], "parameters": []})
        for param in peak["parameters"]:
            output[i]["parameters"].append(
                {
                    "name": param["name"],
                    "value": params[f'p{i}_{param["name"].lower()}'].value,
                    "error": params[f'p{i}_{param["name"].lower()}'].stderr,
                    "vary": params[f'p{i}_{param["name"].lower()}'].vary,
                }
            )
            if param["name"] == "fwhm":
                output[i]["parameters"][-1]["vary"] = params[f"p{i}_sigma"].vary
    return output
