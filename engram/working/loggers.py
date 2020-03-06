import cv2
import imutils
import keyboard  # for keylogs
import time
import numpy as np
import math

def select(logger,stream_start=0):
    selection = {
        "KEYLOGGER": KeyLogger,
        "CAMERA": Camera,
        "FLOW": Flow
    }
    # Get the function from switcher dictionary
    func = selection.get(logger, lambda: "Invalid event parser")
    # Execute the function
    return func(stream_start)


class KeyLogger(object):

    def __init__(self,stream_start = 0):
        self.stream_start = stream_start
        self.log = []
        self.times = []
        self.categories = []

    def manage(self):
        print('Keylogger Managed')

    def callback(self, event):
        """
        This callback is invoked whenever a keyboard event is occured
        (i.e when a key is released in this example)
        """
        name = event.name
        if len(name) > 1:
            # not a character, special key (e.g ctrl, alt, etc.)
            # uppercase with []
            if name == "space":
                # " " instead of "space"
                name = " "
            elif name == "enter":
                # add a new line whenever an ENTER is pressed
                name = "[ENTER]\n"
            elif name == "decimal":
                name = "."
            else:
                # replace spaces with underscores
                name = name.replace(" ", "_")
                name = f"[{name.upper()}]"

        self.categories = np.unique(self.categories.append(name))
        self.log.append(name)
        self.times.append([time.time()-self.stream_start])

    def start(self):
        # start the keylogger
        keyboard.on_release(callback=self.callback)

    def pull(self,full=False):
        if self.log:
            if full:
                val = self.log
            else:
                val = self.log[-1]
        else:
            val = []
            
        return val


class Camera(object):

    def __init__(self,stream_start=0):
        self.cam = cv2.VideoCapture(0)
        self.vis = None
        self.log = []
        self.times = []
        self.stream_start = stream_start
        self.categories = []
        first = self.get_frame(style='BW')
        self.log.append(first[0])
        self.times.append(first[1])

    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def manage(self):
        print('Camera Managed')

    def get_frame(self,style='BW',log=False):
        success, frame = self.cam.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=800,height=600)
        if style == 'BW':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            ret, frame = cv2.imencode('.jpg', frame)

        t = time.time() - self.stream_start

        if log:
            self.log.append(frame)
            self.times.append([t])

        return frame,t
        # return prev.tobytes()

    def pull(self):
        return self.log[-1]

    def push(self,frame,t):
        self.log.append(frame)
        self.times.append([t])


class Flow(object):

    def __init__(self,stream_start=0):
        self.camera = Camera(stream_start)
        self.flow = []
        self.log = []
        self.times = []
        self.stream_start = stream_start
        self.categories = []

    def manage(self):
        self.derive()
        self.label()
        self.show()
        print('Flow Managed')

    def derive(self):
        # Derive Optical Flow from Webcam
        gray,t = self.camera.get_frame(style='BW',log=False)
        self.flow = cv2.calcOpticalFlowFarneback(
            self.camera.pull(), gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.camera.push(gray,t)


    def label(self):
        h, w = self.camera.pull().shape[:2]

        THRESHOLD = 5
        step = w / 50

        x_labels = ['R', 'L', '-']
        y_labels = ['U', 'D', '-']

        self.categories = [x_labels]

        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = self.flow[y, x].T
        mag = np.sqrt((fx ** 2) + (fy ** 2))

        # Only keep flow vectors that are over the threshold
        overThresh = mag > THRESHOLD
        fx_to_draw = fx * overThresh
        x = x.astype('float32')
        x[fx_to_draw == 0] = math.nan
        fx[fx_to_draw == 0] = math.nan
        y = y.astype('float32')
        fy_to_draw = fy * overThresh
        fy[fy_to_draw == 0] = math.nan
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        self.lines = np.int32(lines + 0.5)

        x_comp = np.nanmean(fx)
        y_comp = np.nanmean(fy)
        event = []
        if x_comp > THRESHOLD:
            event.append(x_labels[0])
        elif x_comp < -THRESHOLD:
            event.append(x_labels[1])
        else:
            event.append(x_labels[2])
        if y_comp < -THRESHOLD:
            event.append(y_labels[0])
        elif y_comp > THRESHOLD:
            event.append(y_labels[1])
        else:
            event.append(y_labels[2])

        self.log.append(event)
        self.times = self.camera.times
        return event

    def show(self, name='ENGRAM',prediction=None):

        h, w = self.camera.pull().shape[:2]
        img = np.zeros((h, w, 1), dtype="uint8")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        SIZE = 5
        STROKE = 7
        font = cv2.FONT_HERSHEY_SIMPLEX
        message = self.log[-1][0] + ' | ' + self.log[-1][1]

        # Compare prediction to actual results
        if prediction is None:
            prediction = self.log[-1][0]
        else:
            if prediction == self.log[-1][0]:
                cv2.polylines(img, self.lines, 0, (236, 206, 131))
            else:
                cv2.polylines(img, self.lines, 0, (131, 131, 236))

        # get boundary of this text
        textsize = cv2.getTextSize(message, font, SIZE, STROKE)[0]

        # get coords based on boundary
        textX = (w - textsize[0]) / 2
        textY = (h + textsize[1]) / 2

        cv2.putText(img, message, (int(textX), int(textY)), font, SIZE, (236, 206, 131), STROKE, cv2.LINE_AA)

        # Create Window
        #cv2.namedWindow(name, cv2.WINDOW_FREERATIO)
        #cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(name, img)