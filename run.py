import socketio
from sanic import Sanic
from peak_prophet_server.sessions import session_manager as sm
from peak_prophet_server.sio_events import connect_events

sio = socketio.AsyncServer(async_mode='sanic', cors_allowed_origins="*")
connect_events(sio, sm)


app = Sanic("Peak-Prophet-Server")
sio.attach(app)

if __name__ == '__main__':
    app.run()

