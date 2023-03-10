from peak_prophet_server.fitting import FitManager
from .util import run_coroutine


def connect_events(sio, session_manager):
    @sio.on('connect')
    def connect(sid, data):
        session_manager.sessions[sid] = {}
        print(sid, 'connected!')
        return sid

    @sio.on('fit')
    def fit(sid, data):
        print(sid, 'fitting')
        fit_manager = FitManager()
        result = fit_manager.process_request(data)
        run_coroutine(
            sio.emit('fit_result', result, room=sid)
        )

    @sio.on('disconnect')
    def disconnect(sid):
        print(sid, 'disconnected!')
        del session_manager.sessions[sid]
