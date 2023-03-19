from peak_prophet_server.fitting import FitManager
from .util import run_coroutine


def connect_events(sio):
    @sio.on('connect')
    async def connect(sid, _):
        print(sid, 'connected!')
        await sio.save_session(sid, {'fit_manager': FitManager(sio)})
        return sid

    @sio.on('fit')
    async def fit(sid, data):
        print(sid, 'fitting')
        session = await sio.get_session(sid)
        fit_manager = session['fit_manager']
        result = await fit_manager.process_request(data)

        run_coroutine(sio.emit('result', result))

    @sio.on('request_progress')
    async def get_progress(sid, _):
        session = await sio.get_session(sid)
        await sio.emit('progress', session['fit_manager'].current_progress)

    @sio.on('disconnect')
    async def disconnect(sid):
        print(sid, 'disconnected!')
