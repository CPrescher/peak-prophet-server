from peak_prophet_server.fitting import FitManager


def connect_events(sio):
    @sio.on('connect')
    async def connect(sid, _):
        print(sid, 'connected!')
        await sio.save_session(sid, {'fit_manager': FitManager(sid)})
        return sid

    @sio.on('fit')
    async def fit(sid, data):
        print(sid, 'fitting')
        session = await sio.get_session(sid)
        fit_manager = session['fit_manager']
        result = await fit_manager.process_request(data)
        return result

    @sio.on('stop')
    async def stop(sid):
        print(sid, 'stopping')
        session = await sio.get_session(sid)
        fit_manager = session['fit_manager']
        fit_manager.stop = True

    @sio.on('request_progress')
    async def get_progress(sid):
        session = await sio.get_session(sid)
        return session['fit_manager'].current_progress

    @sio.on('disconnect')
    async def disconnect(sid):
        session = await sio.get_session(sid)
        fit_manager = session['fit_manager']
        fit_manager.stop = True
        print(sid, 'disconnected!')
