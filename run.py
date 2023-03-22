
import uvicorn
import socketio
from peak_prophet_server.sio_events import connect_events


############################################
# OLD WAY to start server:
# from sanic import Sanic
# sio = socketio.AsyncServer(async_mode='sanic', cors_allowed_origins="*")
# connect_events(sio)
#
#
# app = Sanic("Peak-Prophet-Server")
# sio.attach(app)
#
# if __name__ == '__main__':
#     app.run()

############################################


sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins=
   ["https://peakprophet.com", "http://peakprophet.com/",
       "https://peakprophet.web.app", "http://localhost:4200"])
connect_events(sio)

app = socketio.ASGIApp(sio)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8009)


