def connect_events(sio, session_manager):
    @sio.on('connect')
    def connect(sid, data):
        session_manager.sessions[sid] = {}
        print(sid, 'connected!')
        return sid

    @sio.on('disconnect')
    def disconnect(sid):
        print(sid, 'disconnected!')
        del session_manager.sessions[sid]
