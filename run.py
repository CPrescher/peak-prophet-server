
import uvicorn
import socketio
from peak_prophet_server.sessions import session_manager as sm
from peak_prophet_server.sio_events import connect_events


############################################
# OLD WAY to start server:
# from sanic import Sanic
# sio = socketio.AsyncServer(async_mode='sanic', cors_allowed_origins="*")
# connect_events(sio, sm)
#
#
# app = Sanic("Peak-Prophet-Server")
# sio.attach(app)
#
# if __name__ == '__main__':
#     app.run()

############################################


sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
connect_events(sio, sm)

app = socketio.ASGIApp(sio)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)


