from __future__ import division
import vispy


# Note : All files currently copied from Vispy examples.
# See http://vispy.org/gallery.html for similar work.



def select(name):
    selection = {
        "galaxy": galaxy,
        "fireworks": fireworks,
        "boids": boids,
        "realtimesignals" : realtimesignals,
        "brain":brain,
        "sandbox":sandbox,
        "graphical":graph,
        "interactiveqt":interact,
        "atom":atom,
        "oscilloscope":oscilloscope,
        "engram":engram
    }
    # Get the function from switcher dictionary
    func = selection.get(name, lambda: "Invalid event parser")
    # Execute the function
    return func()

def engram():
    print('Engram plot not created yet')
    
def oscilloscope():
    # Copyright (c) Vispy Development Team. All Rights Reserved.
    # Distributed under the (new) BSD License. See LICENSE.txt for more info.
    """
    An oscilloscope, spectrum analyzer, and spectrogram.

    This demo uses pyaudio to record data from the microphone. If pyaudio is not
    available, then a signal will be generated instead.
    """

    import threading
    import atexit
    import numpy as np
    from vispy import app, scene, gloo, visuals
    from vispy.util.filter import gaussian_filter

    try:
        import pyaudio

        class MicrophoneRecorder(object):
            def __init__(self, rate=44100, chunksize=1024):
                self.rate = rate
                self.chunksize = chunksize
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunksize,
                                        stream_callback=self.new_frame)
                self.lock = threading.Lock()
                self.stop = False
                self.frames = []
                atexit.register(self.close)

            def new_frame(self, data, frame_count, time_info, status):
                data = np.fromstring(data, 'int16')
                with self.lock:
                    self.frames.append(data)
                    if self.stop:
                        return None, pyaudio.paComplete
                return None, pyaudio.paContinue

            def get_frames(self):
                with self.lock:
                    frames = self.frames
                    self.frames = []
                    return frames

            def start(self):
                self.stream.start_stream()

            def close(self):
                with self.lock:
                    self.stop = True
                self.stream.close()
                self.p.terminate()

    except ImportError:
        class MicrophoneRecorder(object):
            def __init__(self):
                self.chunksize = 1024
                self.rate = rate = 44100
                t = np.linspace(0, 10, rate*10)
                self.data = (np.sin(t * 10.) * 0.3).astype('float32')
                self.data += np.sin((t + 0.3) * 20.) * 0.15
                self.data += gaussian_filter(np.random.normal(size=self.data.shape)
                                            * 0.2, (0.4, 8))
                self.data += gaussian_filter(np.random.normal(size=self.data.shape)
                                            * 0.005, (0, 1))
                self.data += np.sin(t * 1760 * np.pi)  # 880 Hz
                self.data = (self.data * 2**10 - 2**9).astype('int16')
                self.ptr = 0

            def get_frames(self):
                if self.ptr + 1024 > len(self.data):
                    end = 1024 - (len(self.data) - self.ptr)
                    frame = np.concatenate((self.data[self.ptr:], self.data[:end]))
                else:
                    frame = self.data[self.ptr:self.ptr+1024]
                self.ptr = (self.ptr + 1024) % (len(self.data) - 1024)
                return [frame]

            def start(self):
                pass


    class Oscilloscope(scene.ScrollingLines):
        """A set of lines that are temporally aligned on a trigger.

        Data is added in chunks to the oscilloscope, and each new chunk creates a
        new line to draw. Older lines are slowly faded out until they are removed.

        Parameters
        ----------
        n_lines : int
            The maximum number of lines to draw.
        line_size : int
            The number of samples in each line.
        dx : float
            The x spacing between adjacent samples in a line.
        color : tuple
            The base color to use when drawing lines. Older lines are faded by
            decreasing their alpha value.
        trigger : tuple
            A set of parameters (level, height, width) that determine how triggers
            are detected.
        parent : Node
            An optional parent scenegraph node.
        """
        def __init__(self, n_lines=100, line_size=1024, dx=1e-4,
                    color=(20, 255, 50), trigger=(0, 0.002, 1e-4), parent=None):

            self._trigger = trigger  # trigger_level, trigger_height, trigger_width

            # lateral positioning for trigger
            self.pos_offset = np.zeros((n_lines, 3), dtype=np.float32)

            # color array to fade out older plots
            self.color = np.empty((n_lines, 4), dtype=np.ubyte)
            self.color[:, :3] = [list(color)]
            self.color[:, 3] = 0
            self._dim_speed = 0.01 ** (1 / n_lines)

            self.frames = []  # running list of recently received frames
            self.plot_ptr = 0

            scene.ScrollingLines.__init__(self, n_lines=n_lines,
                                        line_size=line_size, dx=dx,
                                        color=self.color,
                                        pos_offset=self.pos_offset,
                                        parent=parent)
            self.set_gl_state('additive', line_width=2)

        def new_frame(self, data):
            self.frames.append(data)

            # see if we can discard older frames
            while len(self.frames) > 10:
                self.frames.pop(0)

            if self._trigger is None:
                dx = 0
            else:
                # search for next trigger
                th = int(self._trigger[1])  # trigger window height
                tw = int(self._trigger[2] / self._dx)  # trigger window width
                thresh = self._trigger[0]

                trig = np.argwhere((data[tw:] > thresh + th) &
                                (data[:-tw] < thresh - th))
                if len(trig) > 0:
                    m = np.argmin(np.abs(trig - len(data) / 2))
                    i = trig[m, 0]
                    y1 = data[i]
                    y2 = data[min(i + tw * 2, len(data) - 1)]
                    s = y2 / (y2 - y1)
                    i = i + tw * 2 * (1-s)
                    dx = i * self._dx
                else:
                    # default trigger at center of trace
                    # (optionally we could skip plotting instead, or place this
                    # after the most recent trace)
                    dx = self._dx * len(data) / 2.

            # if a trigger was found, add new data to the plot
            self.plot(data, -dx)

        def plot(self, data, dx=0):
            self.set_data(self.plot_ptr, data)

            np.multiply(self.color[..., 3], 0.98, out=self.color[..., 3],
                        casting='unsafe')
            self.color[self.plot_ptr, 3] = 50
            self.set_color(self.color)
            self.pos_offset[self.plot_ptr] = (dx, 0, 0)
            self.set_pos_offset(self.pos_offset)

            self.plot_ptr = (self.plot_ptr + 1) % self._data_shape[0]


    rolling_tex = """
    float rolling_texture(vec2 pos) {
        if( pos.x < 0 || pos.x > 1 || pos.y < 0 || pos.y > 1 ) {
            return 0.0f;
        }
        vec2 uv = vec2(mod(pos.x+$shift, 1), pos.y);
        return texture2D($texture, uv).r;
    }
    """

    cmap = """
    vec4 colormap(float x) {
        x = x - 1e4;
        return vec4(x/5e6, x/2e5, x/1e4, 1);
    }
    """


    class ScrollingImage(scene.Image):
        def __init__(self, shape, parent):
            self._shape = shape
            self._color_fn = visuals.shaders.Function(rolling_tex)
            self._ctex = gloo.Texture2D(np.zeros(shape+(1,), dtype='float32'),
                                        format='luminance', internalformat='r32f')
            self._color_fn['texture'] = self._ctex
            self._color_fn['shift'] = 0
            self.ptr = 0
            scene.Image.__init__(self, method='impostor', parent=parent)
            # self.set_gl_state('additive', cull_face=False)
            self.shared_program.frag['get_data'] = self._color_fn
            cfun = visuals.shaders.Function(cmap)
            self.shared_program.frag['color_transform'] = cfun

        @property
        def size(self):
            return self._shape

        def roll(self, data):
            data = data.reshape(data.shape[0], 1, 1)

            self._ctex[:, self.ptr] = data
            self._color_fn['shift'] = (self.ptr+1) / self._shape[1]
            self.ptr = (self.ptr + 1) % self._shape[1]
            self.update()

        def _prepare_draw(self, view):
            if self._need_vertex_update:
                self._build_vertex_data()

            if view._need_method_update:
                self._update_method(view)

    global fft_frames, scope, spectrum, mic
    mic = MicrophoneRecorder()
    n_fft_frames = 8
    fft_samples = mic.chunksize * n_fft_frames

    win = scene.SceneCanvas(keys='interactive', show=True, fullscreen=True)
    grid = win.central_widget.add_grid()

    view3 = grid.add_view(row=0, col=0, col_span=2, camera='panzoom',
                        border_color='grey')
    image = ScrollingImage((1 + fft_samples // 2, 4000), parent=view3.scene)
    image.transform = scene.LogTransform((0, 10, 0))
    # view3.camera.rect = (0, 0, image.size[1], np.log10(image.size[0]))
    view3.camera.rect = (3493.32, 1.85943, 605.554, 1.41858)

    view1 = grid.add_view(row=1, col=0, camera='panzoom', border_color='grey')
    view1.camera.rect = (-0.01, -0.6, 0.02, 1.2)
    gridlines = scene.GridLines(color=(1, 1, 1, 0.5), parent=view1.scene)
    scope = Oscilloscope(line_size=mic.chunksize, dx=1.0/mic.rate,
                        parent=view1.scene)

    view2 = grid.add_view(row=1, col=1, camera='panzoom', border_color='grey')
    view2.camera.rect = (0.5, -0.5e6, np.log10(mic.rate/2), 5e6)
    lognode = scene.Node(parent=view2.scene)
    lognode.transform = scene.LogTransform((10, 0, 0))
    gridlines2 = scene.GridLines(color=(1, 1, 1, 1), parent=lognode)

    spectrum = Oscilloscope(line_size=1 + fft_samples // 2, n_lines=10,
                            dx=mic.rate/fft_samples,
                            trigger=None, parent=lognode)


    mic.start()

    window = np.hanning(fft_samples)

    fft_frames = []


    def update(ev):
        global fft_frames, scope, spectrum, mic
        data = mic.get_frames()
        for frame in data:
            # import scipy.ndimage as ndi
            # frame -= ndi.gaussian_filter(frame, 50)
            # frame -= frame.mean()

            scope.new_frame(frame)

            fft_frames.append(frame)
            if len(fft_frames) >= n_fft_frames:
                cframes = np.concatenate(fft_frames) * window
                fft = np.abs(np.fft.rfft(cframes)).astype('float32')
                fft_frames.pop(0)

                spectrum.new_frame(fft)
                image.roll(fft)


    timer = app.Timer(interval='auto', connect=update)
    timer.start()


    app.run()

def atom():
    # vispy: gallery 30
    # -----------------------------------------------------------------------------
    # Copyright (c) Vispy Development Team. All Rights Reserved.
    # Distributed under the (new) BSD License. See LICENSE.txt for more info.
    # -----------------------------------------------------------------------------
    # Author: Nicolas P .Rougier
    # Date:   06/03/2014
    # Abstract: Fake electrons orbiting
    # Keywords: Sprites, atom, particles
    # -----------------------------------------------------------------------------

    import numpy as np
    from vispy import gloo
    from vispy import app
    from vispy.util.transforms import perspective, translate, rotate

    # Create vertices
    n, p = 100, 150
    data = np.zeros(p * n, [('a_position', np.float32, 2),
                            ('a_color',    np.float32, 4),
                            ('a_rotation', np.float32, 4)])
    trail = .5 * np.pi
    data['a_position'][:, 0] = np.resize(np.linspace(0, trail, n), p * n)
    data['a_position'][:, 0] += np.repeat(np.random.uniform(0, 2 * np.pi, p), n)
    data['a_position'][:, 1] = np.repeat(np.linspace(0, 2 * np.pi, p), n)

    data['a_color'] = 1, 1, 1, 1
    data['a_color'] = np.repeat(
        np.random.uniform(0.75, 1.00, (p, 4)).astype(np.float32), n, axis=0)
    data['a_color'][:, 3] = np.resize(np.linspace(0, 1, n), p * n)

    data['a_rotation'] = np.repeat(
        np.random.uniform(0, 2 * np.pi, (p, 4)).astype(np.float32), n, axis=0)


    vert = """
    #version 120
    uniform mat4 u_model;
    uniform mat4 u_view;
    uniform mat4 u_projection;
    uniform float u_size;
    uniform float u_clock;

    attribute vec2 a_position;
    attribute vec4 a_color;
    attribute vec4 a_rotation;
    varying vec4 v_color;

    mat4 build_rotation(vec3 axis, float angle)
    {
        axis = normalize(axis);
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;
        return mat4(oc * axis.x * axis.x + c,
                    oc * axis.x * axis.y - axis.z * s,
                    oc * axis.z * axis.x + axis.y * s,
                    0.0,
                    oc * axis.x * axis.y + axis.z * s,
                    oc * axis.y * axis.y + c,
                    oc * axis.y * axis.z - axis.x * s,
                    0.0,
                    oc * axis.z * axis.x - axis.y * s,
                    oc * axis.y * axis.z + axis.x * s,
                    oc * axis.z * axis.z + c,
                    0.0,
                    0.0, 0.0, 0.0, 1.0);
    }


    void main (void) {
        v_color = a_color;

        float x0 = 1.5;
        float z0 = 0.0;

        float theta = a_position.x + u_clock;
        float x1 = x0*cos(theta) + z0*sin(theta);
        float y1 = 0.0;
        float z1 = (z0*cos(theta) - x0*sin(theta))/2.0;

        mat4 R = build_rotation(a_rotation.xyz, a_rotation.w);
        gl_Position = u_projection * u_view * u_model * R * vec4(x1,y1,z1,1);
        gl_PointSize = 8.0 * u_size * sqrt(v_color.a);
    }
    """

    frag = """
    #version 120
    varying vec4 v_color;
    varying float v_size;
    void main()
    {
        float d = 2*(length(gl_PointCoord.xy - vec2(0.5,0.5)));
        gl_FragColor = vec4(v_color.rgb, v_color.a*(1-d));
    }
    """


    # ------------------------------------------------------------ Canvas class ---
    class Canvas(app.Canvas):

        def __init__(self):
            app.Canvas.__init__(self, keys='interactive', size=(800, 800))

            self.translate = 6.5
            self.program = gloo.Program(vert, frag)
            self.view = translate((0, 0, -self.translate))
            self.model = np.eye(4, dtype=np.float32)
            self.projection = np.eye(4, dtype=np.float32)
            self.apply_zoom()

            self.program.bind(gloo.VertexBuffer(data))
            self.program['u_model'] = self.model
            self.program['u_view'] = self.view
            self.program['u_size'] = 5 / self.translate

            self.theta = 0
            self.phi = 0
            self.clock = 0
            self.stop_rotation = False

            gloo.set_state('translucent', depth_test=False)
            self.program['u_clock'] = 0.0

            self._timer = app.Timer('auto', connect=self.on_timer, start=True)

            self.show()

        def on_key_press(self, event):
            if event.text == ' ':
                self.stop_rotation = not self.stop_rotation

        def on_timer(self, event):
            if not self.stop_rotation:
                self.theta += .05
                self.phi += .05
                self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                                    rotate(self.phi, (0, 1, 0)))
                self.program['u_model'] = self.model
            self.clock += np.pi / 100
            self.program['u_clock'] = self.clock
            self.update()

        def on_resize(self, event):
            self.apply_zoom()

        def on_mouse_wheel(self, event):
            self.translate += event.delta[1]
            self.translate = max(2, self.translate)
            self.view = translate((0, 0, -self.translate))
            self.program['u_view'] = self.view
            self.program['u_size'] = 5 / self.translate
            self.update()

        def on_draw(self, event):
            gloo.clear('black')
            self.program.draw('points')

        def apply_zoom(self):
            width, height = self.physical_size
            gloo.set_viewport(0, 0, width, height)
            self.projection = perspective(45.0, width / float(height), 1.0, 1000.0)
            self.program['u_projection'] = self.projection


    c = Canvas()
    app.run()

def interact():
    # vispy: testskip
    # -----------------------------------------------------------------------------
    # Copyright (c) Vispy Development Team. All Rights Reserved.
    # Distributed under the (new) BSD License. See LICENSE.txt for more info.
    # -----------------------------------------------------------------------------
    # Abstract: show mesh primitive
    # Keywords: cone, arrow, sphere, cylinder, qt
    # -----------------------------------------------------------------------------

    """
    Test the fps capability of Vispy with meshdata primitive
    """
    try:
        from sip import setapi
        setapi("QVariant", 2)
        setapi("QString", 2)
    except ImportError:
        pass

    from PyQt5 import QtCore, QtWidgets
    import sys

    import numpy as np
    from vispy import app, gloo
    from vispy.util.transforms import perspective, translate, rotate
    from vispy.geometry import meshdata as md
    from vispy.geometry import generation as gen

    OBJECT = {'sphere': [('rows', 3, 1000, 'int', 3),
                        ('cols', 3, 1000, 'int', 3),
                        ('radius', 0.1, 10, 'double', 1.0)],
            'cylinder': [('rows', 4, 1000, 'int', 4),
                        ('cols', 4, 1000, 'int', 4),
                        ('radius', 0.1, 10, 'double', 1.0),
                        ('radius Top.', 0.1, 10, 'double', 1.0),
                        ('length', 0.1, 10, 'double', 1.0)],
            'cone': [('cols', 3, 1000, 'int', 3),
                    ('radius', 0.1, 10, 'double', 1.0),
                    ('length', 0.1, 10, 'double', 1.0)],
            'arrow': [('rows', 4, 1000, 'int', 4),
                        ('cols', 4, 1000, 'int', 4),
                        ('radius', 0.01, 10, 'double', 0.1),
                        ('length', 0.1, 10, 'double', 1.0),
                        ('cone_radius', 0.1, 10, 'double', 0.2),
                        ('cone_length', 0.0, 10., 'double', 0.3)]}

    vert = """
    // Uniforms
    // ------------------------------------
    uniform   mat4 u_model;
    uniform   mat4 u_view;
    uniform   mat4 u_projection;
    uniform   vec4 u_color;

    // Attributes
    // ------------------------------------
    attribute vec3 a_position;
    attribute vec3 a_normal;
    attribute vec4 a_color;

    // Varying
    // ------------------------------------
    varying vec4 v_color;

    void main()
    {
        v_color = a_color * u_color;
        gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    }
    """


    frag = """
    // Varying
    // ------------------------------------
    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;
    }
    """

    DEFAULT_COLOR = (0, 1, 1, 1)
    # -----------------------------------------------------------------------------


    class MyMeshData(md.MeshData):
        """ Add to Meshdata class the capability to export good data for gloo """
        def __init__(self, vertices=None, faces=None, edges=None,
                    vertex_colors=None, face_colors=None):
            md.MeshData.__init__(self, vertices=None, faces=None, edges=None,
                                vertex_colors=None, face_colors=None)

        def get_glTriangles(self):
            """
            Build vertices for a colored mesh.
                V  is the vertices
                I1 is the indices for a filled mesh (use with GL_TRIANGLES)
                I2 is the indices for an outline mesh (use with GL_LINES)
            """
            vtype = [('a_position', np.float32, 3),
                    ('a_normal', np.float32, 3),
                    ('a_color', np.float32, 4)]
            vertices = self.get_vertices()
            normals = self.get_vertex_normals()
            faces = np.uint32(self.get_faces())

            edges = np.uint32(self.get_edges().reshape((-1)))
            colors = self.get_vertex_colors()

            nbrVerts = vertices.shape[0]
            V = np.zeros(nbrVerts, dtype=vtype)
            V[:]['a_position'] = vertices
            V[:]['a_normal'] = normals
            V[:]['a_color'] = colors

            return V, faces.reshape((-1)), edges.reshape((-1))
    # -----------------------------------------------------------------------------


    class ObjectParam(object):
        """
        OBJECT parameter test
        """
        def __init__(self, name, list_param):
            self.name = name
            self.list_param = list_param
            self.props = {}
            self.props['visible'] = True
            for nameV, minV, maxV, typeV, iniV in list_param:
                self.props[nameV] = iniV
    # -----------------------------------------------------------------------------


    class ObjectWidget(QtWidgets.QWidget):
        """
        Widget for editing OBJECT parameters
        """
        signal_objet_changed = QtCore.pyqtSignal(ObjectParam, name='objectChanged')

        def __init__(self, parent=None, param=None):
            super(ObjectWidget, self).__init__(parent)

            if param is None:
                self.param = ObjectParam('sphere', OBJECT['sphere'])
            else:
                self.param = param

            self.gb_c = QtWidgets.QGroupBox(u"Hide/Show %s" % self.param.name)
            self.gb_c.setCheckable(True)
            self.gb_c.setChecked(self.param.props['visible'])
            self.gb_c.toggled.connect(self.update_param)

            lL = []
            self.sp = []
            gb_c_lay = QtWidgets.QGridLayout()
            for nameV, minV, maxV, typeV, iniV in self.param.list_param:
                lL.append(QtWidgets.QLabel(nameV, self.gb_c))
                if typeV == 'double':
                    self.sp.append(QtWidgets.QDoubleSpinBox(self.gb_c))
                    self.sp[-1].setDecimals(2)
                    self.sp[-1].setSingleStep(0.1)
                    self.sp[-1].setLocale(QtCore.QLocale(QtCore.QLocale.English))
                elif typeV == 'int':
                    self.sp.append(QtWidgets.QSpinBox(self.gb_c))
                self.sp[-1].setMinimum(minV)
                self.sp[-1].setMaximum(maxV)
                self.sp[-1].setValue(iniV)

            # Layout
            for pos in range(len(lL)):
                gb_c_lay.addWidget(lL[pos], pos, 0)
                gb_c_lay.addWidget(self.sp[pos], pos, 1)
                # Signal
                self.sp[pos].valueChanged.connect(self.update_param)

            self.gb_c.setLayout(gb_c_lay)

            vbox = QtWidgets.QVBoxLayout()
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(self.gb_c)
            hbox.addStretch(1)
            vbox.addLayout(hbox)
            vbox.addStretch(1)

            self.setLayout(vbox)

        def update_param(self, option):
            """
            update param and emit a signal
            """
            self.param.props['visible'] = self.gb_c.isChecked()
            keys = map(lambda x: x[0], self.param.list_param)
            for pos, nameV in enumerate(keys):
                self.param.props[nameV] = self.sp[pos].value()
            # emit signal
            self.signal_objet_changed.emit(self.param)
    # -----------------------------------------------------------------------------


    class Canvas(app.Canvas):

        def __init__(self,):
            app.Canvas.__init__(self)
            self.size = 800, 600
            # fovy, zfar params
            self.fovy = 45.0
            self.zfar = 10.0
            width, height = self.size
            self.aspect = width / float(height)

            self.program = gloo.Program(vert, frag)

            self.model = np.eye(4, dtype=np.float32)
            self.projection = np.eye(4, dtype=np.float32)
            self.view = translate((0, 0, -5.0))

            self.program['u_model'] = self.model
            self.program['u_view'] = self.view

            self.theta = 0
            self.phi = 0
            self.visible = True

            self._timer = app.Timer(1.0 / 60)
            self._timer.connect(self.on_timer)
            self._timer.start()

        # ---------------------------------
            gloo.set_clear_color((1, 1, 1, 1))
            gloo.set_state('opaque')
            gloo.set_polygon_offset(1, 1)

        # ---------------------------------
        def on_timer(self, event):
            self.theta += .5
            self.phi += .5
            self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                                rotate(self.phi, (0, 1, 0)))
            self.program['u_model'] = self.model
            self.update()

        # ---------------------------------
        def on_resize(self, event):
            width, height = event.size
            self.size = event.size
            gloo.set_viewport(0, 0, width, height)
            self.aspect = width / float(height)
            self.projection = perspective(self.fovy, width / float(height), 1.0,
                                        self.zfar)
            self.program['u_projection'] = self.projection

        # ---------------------------------
        def on_draw(self, event):
            gloo.clear()
            if self.visible:
                # Filled mesh
                gloo.set_state(blend=False, depth_test=True,
                            polygon_offset_fill=True)
                self.program['u_color'] = 1, 1, 1, 1
                self.program.draw('triangles', self.filled_buf)

                # Outline
                gloo.set_state(blend=True, depth_test=True,
                            polygon_offset_fill=False)
                gloo.set_depth_mask(False)
                self.program['u_color'] = 0, 0, 0, 1
                self.program.draw('lines', self.outline_buf)
                gloo.set_depth_mask(True)

        # ---------------------------------
        def set_data(self, vertices, filled, outline):
            self.filled_buf = gloo.IndexBuffer(filled)
            self.outline_buf = gloo.IndexBuffer(outline)
            self.vertices_buff = gloo.VertexBuffer(vertices)
            self.program.bind(self.vertices_buff)
            self.update()
    # -----------------------------------------------------------------------------


    class MainWindow(QtWidgets.QMainWindow):

        def __init__(self):
            QtWidgets.QMainWindow.__init__(self)

            self.resize(700, 500)
            self.setWindowTitle('vispy example ...')

            self.list_object = QtWidgets.QListWidget()
            self.list_object.setAlternatingRowColors(True)
            self.list_object.itemSelectionChanged.connect(self.list_objectChanged)

            self.list_object.addItems(list(OBJECT.keys()))
            self.props_widget = ObjectWidget(self)
            self.props_widget.signal_objet_changed.connect(self.update_view)

            self.splitter_v = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self.splitter_v.addWidget(self.list_object)
            self.splitter_v.addWidget(self.props_widget)

            self.canvas = Canvas()
            self.canvas.create_native()
            self.canvas.native.setParent(self)
            self.canvas.measure_fps(0.1, self.show_fps)

            # Central Widget
            splitter1 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter1.addWidget(self.splitter_v)
            splitter1.addWidget(self.canvas.native)

            self.setCentralWidget(splitter1)

            # FPS message in statusbar:
            self.status = self.statusBar()
            self.status_label = QtWidgets.QLabel('...')
            self.status.addWidget(self.status_label)

            self.mesh = MyMeshData()
            self.update_view(self.props_widget.param)

        def list_objectChanged(self):
            row = self.list_object.currentIndex().row()
            name = self.list_object.currentIndex().data()
            if row != -1:
                self.props_widget.deleteLater()
                self.props_widget = ObjectWidget(self, param=ObjectParam(name,
                                                OBJECT[name]))
                self.splitter_v.addWidget(self.props_widget)
                self.props_widget.signal_objet_changed.connect(self.update_view)
                self.update_view(self.props_widget.param)

        def show_fps(self, fps):
            nbr_tri = self.mesh.n_faces
            msg = "FPS - %0.2f and nbr Tri %s " % (float(fps), int(nbr_tri))
            # NOTE: We can't use showMessage in PyQt5 because it causes
            #       a draw event loop (show_fps for every drawing event,
            #       showMessage causes a drawing event, and so on).
            self.status_label.setText(msg)

        def update_view(self, param):
            cols = param.props['cols']
            radius = param.props['radius']
            if param.name == 'sphere':
                rows = param.props['rows']
                mesh = gen.create_sphere(cols, rows, radius=radius)
            elif param.name == 'cone':
                length = param.props['length']
                mesh = gen.create_cone(cols, radius=radius, length=length)
            elif param.name == 'cylinder':
                rows = param.props['rows']
                length = param.props['length']
                radius2 = param.props['radius Top.']
                mesh = gen.create_cylinder(rows, cols, radius=[radius, radius2],
                                        length=length)
            elif param.name == 'arrow':
                length = param.props['length']
                rows = param.props['rows']
                cone_radius = param.props['cone_radius']
                cone_length = param.props['cone_length']
                mesh = gen.create_arrow(rows, cols, radius=radius, length=length,
                                        cone_radius=cone_radius,
                                        cone_length=cone_length)
            else:
                return

            self.canvas.visible = param.props['visible']
            self.mesh.set_vertices(mesh.get_vertices())
            self.mesh.set_faces(mesh.get_faces())
            colors = np.tile(DEFAULT_COLOR, (self.mesh.n_vertices, 1))
            self.mesh.set_vertex_colors(colors)
            vertices, filled, outline = self.mesh.get_glTriangles()
            self.canvas.set_data(vertices, filled, outline)

    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    appQt.exec_()

def graph():
    """
    Plot clusters of data points and a graph of connections
    """
    from vispy import app, scene, color
    import numpy as np

    # Initialize arrays for position, color, edges, and types for each point in
    # the graph.
    npts = 400
    nedges = 900
    ngroups = 7
    np.random.seed(127396)
    pos = np.empty((npts, 2), dtype='float32')
    colors = np.empty((npts, 3), dtype='float32')
    edges = np.empty((nedges, 2), dtype='uint32')
    types = np.empty(npts, dtype=int)

    # Assign random starting positions
    pos[:] = np.random.normal(size=pos.shape, scale=4.)

    # Assign each point to a group
    grpsize = npts // ngroups
    ptr = 0
    typ = 0
    while ptr < npts:
        size = np.random.random() * grpsize + grpsize // 2
        types[int(ptr):int(ptr+size)] = typ
        typ += 1
        ptr = ptr + size

    # Randomly select connections, with higher connection probability between 
    # points in the same group
    conn = []
    connset = set()
    while len(conn) < nedges:
        i, j = np.random.randint(npts, size=2)
        if i == j:
            continue
        p = 0.7 if types[i] == types[j] else 0.01
        if np.random.random() < p:
            if (i, j) in connset:
                continue
            connset.add((i, j))
            connset.add((j, i))
            conn.append([i, j])
    edges[:] = conn

    # Assign colors to each point based on its type
    cmap = color.get_colormap('cubehelix')
    typ_colors = np.array([cmap.map(x)[0, :3] for x in np.linspace(0.2, 0.8, typ)])
    colors[:] = typ_colors[types]

    # Add some RGB noise and clip
    colors *= 1.1 ** np.random.normal(size=colors.shape)
    colors = np.clip(colors, 0, 1)


    # Display the data
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    view.camera.aspect = 1

    lines = scene.Line(pos=pos, connect=edges, antialias=False, method='gl',
                    color=(1, 1, 1, 0.2), parent=view.scene)
    markers = scene.Markers(pos=pos, face_color=colors, symbol='o',
                            parent=view.scene)

    view.camera.set_range()

    i = 1


    def update(ev):
        global pos, edges, lines, markers, view, force, dist, i
        
        dx = np.empty((npts, npts, 2), dtype='float32')
        dx[:] = pos[:, np.newaxis, :]
        dx -= pos[np.newaxis, :, :]

        dist = (dx**2).sum(axis=2)**0.5
        dist[dist == 0] = 1.
        ndx = dx / dist[..., np.newaxis]
        
        force = np.zeros((npts, npts, 2), dtype='float32')
        
        # all points push away from each other
        force -= 0.1 * ndx / dist[..., np.newaxis]**2
        
        # connected points pull toward each other
        # pulsed force helps to settle faster:    
        s = 0.1
        #s = 0.05 * 5 ** (np.sin(i/20.) / (i/100.))
        
        #s = 0.05 + 1 * 0.99 ** i
        mask = np.zeros((npts, npts, 1), dtype='float32')
        mask[edges[:, 0], edges[:, 1]] = s
        mask[edges[:, 1], edges[:, 0]] = s
        force += dx * dist[..., np.newaxis] * mask
        
        # points do not exert force on themselves
        force[np.arange(npts), np.arange(npts)] = 0
        
        force = force.sum(axis=0)
        pos += np.clip(force, -3, 3) * 0.09
        
        lines.set_data(pos=pos)
        markers.set_data(pos=pos, face_color=colors)
        
        i += 1


    timer = app.Timer(interval=0, connect=update, start=True)


    app.run()

def sandbox():
    """
    A GLSL sandbox application based on the spinning cube. Requires PySide
    or PyQt5.
    """

    import numpy as np
    from vispy import app, gloo
    from vispy.io import read_mesh, load_data_file, load_crate
    from vispy.util.transforms import perspective, translate, rotate

    try:
        from PyQt5.QtGui import QFont
        from PyQt5.QtWidgets import (QWidget, QPlainTextEdit, QLabel, QPushButton,
                                    QHBoxLayout, QVBoxLayout)
    except ImportError:
        from PyQt4.QtGui import (QWidget, QPlainTextEdit, QFont, QLabel,
                                QPushButton, QHBoxLayout, QVBoxLayout)

    VERT_CODE = """
    uniform   mat4 u_model;
    uniform   mat4 u_view;
    uniform   mat4 u_projection;

    attribute vec3 a_position;
    attribute vec2 a_texcoord;

    varying vec2 v_texcoord;

    void main()
    {
        v_texcoord = a_texcoord;
        gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
        //gl_Position = vec4(a_position,1.0);
    }
    """


    FRAG_CODE = """
    uniform sampler2D u_texture;
    varying vec2 v_texcoord;

    void main()
    {
        float ty = v_texcoord.y;
        float tx = sin(ty*50.0)*0.01 + v_texcoord.x;
        gl_FragColor = texture2D(u_texture, vec2(tx, ty));
    }
    """


    # Read cube data
    positions, faces, normals, texcoords = \
        read_mesh(load_data_file('orig/cube.obj'))
    colors = np.random.uniform(0, 1, positions.shape).astype('float32')

    faces_buffer = gloo.IndexBuffer(faces.astype(np.uint16))


    class Canvas(app.Canvas):

        def __init__(self, **kwargs):
            app.Canvas.__init__(self, size=(400, 400), **kwargs)

            self.program = gloo.Program(VERT_CODE, FRAG_CODE)

            # Set attributes
            self.program['a_position'] = gloo.VertexBuffer(positions)
            self.program['a_texcoord'] = gloo.VertexBuffer(texcoords)

            self.program['u_texture'] = gloo.Texture2D(load_crate())

            # Handle transformations
            self.init_transforms()

            self.apply_zoom()

            gloo.set_clear_color((1, 1, 1, 1))
            gloo.set_state(depth_test=True)

            self._timer = app.Timer('auto', connect=self.update_transforms)
            self._timer.start()

            self.show()

        def on_resize(self, event):
            self.apply_zoom()

        def on_draw(self, event):
            gloo.clear()
            self.program.draw('triangles', faces_buffer)

        def init_transforms(self):
            self.theta = 0
            self.phi = 0
            self.view = translate((0, 0, -5))
            self.model = np.eye(4, dtype=np.float32)
            self.projection = np.eye(4, dtype=np.float32)

            self.program['u_model'] = self.model
            self.program['u_view'] = self.view

        def update_transforms(self, event):
            self.theta += .5
            self.phi += .5
            self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                                rotate(self.phi, (0, 1, 0)))
            self.program['u_model'] = self.model
            self.update()

        def apply_zoom(self):
            gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
            self.projection = perspective(45.0, self.size[0] /
                                        float(self.size[1]), 2.0, 10.0)
            self.program['u_projection'] = self.projection


    class TextField(QPlainTextEdit):

        def __init__(self, parent):
            QPlainTextEdit.__init__(self, parent)
            # Set font to monospaced (TypeWriter)
            font = QFont('')
            font.setStyleHint(font.TypeWriter, font.PreferDefault)
            font.setPointSize(8)
            self.setFont(font)


    class MainWindow(QWidget):

        def __init__(self):
            QWidget.__init__(self, None)

            self.setMinimumSize(600, 400)

            # Create two labels and a button
            self.vertLabel = QLabel("Vertex code", self)
            self.fragLabel = QLabel("Fragment code", self)
            self.theButton = QPushButton("Compile!", self)
            self.theButton.clicked.connect(self.on_compile)

            # Create two editors
            self.vertEdit = TextField(self)
            self.vertEdit.setPlainText(VERT_CODE)
            self.fragEdit = TextField(self)
            self.fragEdit.setPlainText(FRAG_CODE)

            # Create a canvas
            self.canvas = Canvas(parent=self)

            # Layout
            hlayout = QHBoxLayout(self)
            self.setLayout(hlayout)
            vlayout = QVBoxLayout()
            #
            hlayout.addLayout(vlayout, 1)
            hlayout.addWidget(self.canvas.native, 1)
            #
            vlayout.addWidget(self.vertLabel, 0)
            vlayout.addWidget(self.vertEdit, 1)
            vlayout.addWidget(self.fragLabel, 0)
            vlayout.addWidget(self.fragEdit, 1)
            vlayout.addWidget(self.theButton, 0)

            self.show()

        def on_compile(self):
            vert_code = str(self.vertEdit.toPlainText())
            frag_code = str(self.fragEdit.toPlainText())
            self.canvas.program.set_shaders(vert_code, frag_code)
            # Note how we do not need to reset our variables, they are
            # re-set automatically (by gloo)



    app.create()
    m = MainWindow()
    app.run()
def brain():
    # vispy: gallery 2
    # Copyright (c) Vispy Development Team. All Rights Reserved.
    # Distributed under the (new) BSD License. See LICENSE.txt for more info.

    """
    3D brain mesh viewer.
    """

    from timeit import default_timer
    import numpy as np

    from vispy import gloo
    from vispy import app
    from vispy.util.transforms import perspective, translate, rotate
    from vispy.io import load_data_file

    brain = np.load(load_data_file('brain/brain.npz', force_download='2014-09-04'))
    data = brain['vertex_buffer']
    faces = brain['index_buffer']

    VERT_SHADER = """
    #version 120
    uniform mat4 u_model;
    uniform mat4 u_view;
    uniform mat4 u_projection;
    uniform vec4 u_color;

    attribute vec3 a_position;
    attribute vec3 a_normal;
    attribute vec4 a_color;

    varying vec3 v_position;
    varying vec3 v_normal;
    varying vec4 v_color;

    void main()
    {
        v_normal = a_normal;
        v_position = a_position;
        v_color = a_color * u_color;
        gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    }
    """

    FRAG_SHADER = """
    #version 120
    uniform mat4 u_model;
    uniform mat4 u_view;
    uniform mat4 u_normal;

    uniform vec3 u_light_intensity;
    uniform vec3 u_light_position;

    varying vec3 v_position;
    varying vec3 v_normal;
    varying vec4 v_color;

    void main()
    {
        // Calculate normal in world coordinates
        vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

        // Calculate the location of this fragment (pixel) in world coordinates
        vec3 position = vec3(u_view*u_model * vec4(v_position, 1));

        // Calculate the vector from this pixels surface to the light source
        vec3 surfaceToLight = u_light_position - position;

        // Calculate the cosine of the angle of incidence (brightness)
        float brightness = dot(normal, surfaceToLight) /
                        (length(surfaceToLight) * length(normal));
        brightness = max(min(brightness,1.0),0.0);

        // Calculate final color of the pixel, based on:
        // 1. The angle of incidence: brightness
        // 2. The color/intensities of the light: light.intensities
        // 3. The texture and texture coord: texture(tex, fragTexCoord)

        // Specular lighting.
        vec3 surfaceToCamera = vec3(0.0, 0.0, 1.0) - position;
        vec3 K = normalize(normalize(surfaceToLight) + normalize(surfaceToCamera));
        float specular = clamp(pow(abs(dot(normal, K)), 40.), 0.0, 1.0);

        gl_FragColor = v_color * brightness * vec4(u_light_intensity, 1);
    }
    """


    class Canvas(app.Canvas):
        def __init__(self):
            app.Canvas.__init__(self, keys='interactive')
            self.size = 800, 600

            self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

            self.theta, self.phi = -80, 180
            self.translate = 3

            self.faces = gloo.IndexBuffer(faces)
            self.program.bind(gloo.VertexBuffer(data))

            self.program['u_color'] = 1, 1, 1, 1
            self.program['u_light_position'] = (1., 1., 1.)
            self.program['u_light_intensity'] = (1., 1., 1.)

            self.apply_zoom()

            gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)

            self._t0 = default_timer()
            self._timer = app.Timer('auto', connect=self.on_timer, start=True)

            self.update_matrices()

        def update_matrices(self):
            self.view = translate((0, 0, -self.translate))
            self.model = np.dot(rotate(self.theta, (1, 0, 0)),
                                rotate(self.phi, (0, 1, 0)))
            self.projection = np.eye(4, dtype=np.float32)
            self.program['u_model'] = self.model
            self.program['u_view'] = self.view
            self.program['u_normal'] = np.linalg.inv(np.dot(self.view,
                                                            self.model)).T

        def on_timer(self, event):
            elapsed = default_timer() - self._t0
            self.phi = 180 + elapsed * 50.
            self.update_matrices()
            self.update()

        def on_resize(self, event):
            self.apply_zoom()

        def on_mouse_wheel(self, event):
            self.translate += -event.delta[1]/5.
            self.translate = max(2, self.translate)
            self.update_matrices()
            self.update()

        def on_draw(self, event):
            gloo.clear()
            self.program.draw('triangles', indices=self.faces)

        def apply_zoom(self):
            gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
            self.projection = perspective(45.0, self.size[0] /
                                        float(self.size[1]), 1.0, 20.0)
            self.program['u_projection'] = self.projection

    
    c = Canvas()
    c.show()
    app.run()



def realtimesignals():
    # vispy: gallery 2
    # Copyright (c) Vispy Development Team. All Rights Reserved.
    # Distributed under the (new) BSD License. See LICENSE.txt for more info.

    """
    Multiple real-time digital signals with GLSL-based clipping.
    """

    from vispy import gloo
    from vispy import app
    import numpy as np
    import math

    # Number of cols and rows in the table.
    nrows = 16
    ncols = 20

    # Number of signals.
    m = nrows*ncols

    # Number of samples per signal.
    n = 1000

    # Various signal amplitudes.
    amplitudes = .1 + .2 * np.random.rand(m, 1).astype(np.float32)

    # Generate the signals as a (m, n) array.
    y = amplitudes * np.random.randn(m, n).astype(np.float32)

    # Color of each vertex (TODO: make it more efficient by using a GLSL-based
    # color map and the index).
    color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
                    n, axis=0).astype(np.float32)

    # Signal 2D index of each vertex (row and col) and x-index (sample index
    # within each signal).
    index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
                np.repeat(np.tile(np.arange(nrows), ncols), n),
                np.tile(np.arange(n), m)].astype(np.float32)

    VERT_SHADER = """
    #version 120

    // y coordinate of the position.
    attribute float a_position;

    // row, col, and time index.
    attribute vec3 a_index;
    varying vec3 v_index;

    // 2D scaling factor (zooming).
    uniform vec2 u_scale;

    // Size of the table.
    uniform vec2 u_size;

    // Number of samples per signal.
    uniform float u_n;

    // Color.
    attribute vec3 a_color;
    varying vec4 v_color;

    // Varying variables used for clipping in the fragment shader.
    varying vec2 v_position;
    varying vec4 v_ab;

    void main() {
        float nrows = u_size.x;
        float ncols = u_size.y;

        // Compute the x coordinate from the time index.
        float x = -1 + 2*a_index.z / (u_n-1);
        vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);

        // Find the affine transformation for the subplots.
        vec2 a = vec2(1./ncols, 1./nrows)*.9;
        vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                    -1 + 2*(a_index.y+.5) / nrows);
        // Apply the static subplot transformation + scaling.
        gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);

        v_color = vec4(a_color, 1.);
        v_index = a_index;

        // For clipping test in the fragment shader.
        v_position = gl_Position.xy;
        v_ab = vec4(a, b);
    }
    """

    FRAG_SHADER = """
    #version 120

    varying vec4 v_color;
    varying vec3 v_index;

    varying vec2 v_position;
    varying vec4 v_ab;

    void main() {
        gl_FragColor = v_color;

        // Discard the fragments between the signals (emulate glMultiDrawArrays).
        if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
            discard;

        // Clipping test.
        vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
        if ((test.x > 1) || (test.y > 1))
            discard;
    }
    """


    class Canvas(app.Canvas):
        def __init__(self):
            app.Canvas.__init__(self, title='Use your wheel to zoom!',
                                keys='interactive')
            self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
            self.program['a_position'] = y.reshape(-1, 1)
            self.program['a_color'] = color
            self.program['a_index'] = index
            self.program['u_scale'] = (1., 1.)
            self.program['u_size'] = (nrows, ncols)
            self.program['u_n'] = n

            gloo.set_viewport(0, 0, *self.physical_size)

            self._timer = app.Timer('auto', connect=self.on_timer, start=True)

            gloo.set_state(clear_color='black', blend=True,
                        blend_func=('src_alpha', 'one_minus_src_alpha'))

            self.show()

        def on_resize(self, event):
            gloo.set_viewport(0, 0, *event.physical_size)

        def on_mouse_wheel(self, event):
            dx = np.sign(event.delta[1]) * .05
            scale_x, scale_y = self.program['u_scale']
            scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),
                                        scale_y * math.exp(0.0*dx))
            self.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
            self.update()

        def on_timer(self, event):
            """Add some data at the end of each signal (real-time signals)."""
            k = 10
            y[:, :-k] = y[:, k:]
            y[:, -k:] = amplitudes * np.random.randn(m, k)

            self.program['a_position'].set_data(y.ravel().astype(np.float32))
            self.update()

        def on_draw(self, event):
            gloo.clear()
            self.program.draw('line_strip')


    c = Canvas()
    app.run()

def boids():
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    # vispy: gallery 30

    """
    Demonstration of boids simulation. Boids is an artificial life
    program, developed by Craig Reynolds in 1986, which simulates the
    flocking behaviour of birds.
    Based on code from glumpy by Nicolas Rougier.
    """

    import time

    import numpy as np
    from scipy.spatial import cKDTree

    from vispy import gloo
    from vispy import app

    VERT_SHADER = """
    #version 120
    attribute vec3 position;
    attribute vec4 color;
    attribute float size;

    varying vec4 v_color;
    void main (void) {
        gl_Position = vec4(position, 1.0);
        v_color = color;
        gl_PointSize = size;
    }
    """

    FRAG_SHADER = """
    #version 120
    varying vec4 v_color;
    void main()
    {
        float x = 2.0*gl_PointCoord.x - 1.0;
        float y = 2.0*gl_PointCoord.y - 1.0;
        float a = 1.0 - (x*x + y*y);
        gl_FragColor = vec4(v_color.rgb, a*v_color.a);
    }

    """


    class Canvas(app.Canvas):

        def __init__(self):
            app.Canvas.__init__(self, keys='interactive')

            ps = self.pixel_scale

            # Create boids
            n = 1000
            size_type = ('size', 'f4', 1*ps) if ps > 1 else ('size', 'f4')
            self.particles = np.zeros(2 + n, [('position', 'f4', 3),
                                            ('position_1', 'f4', 3),
                                            ('position_2', 'f4', 3),
                                            ('velocity', 'f4', 3),
                                            ('color', 'f4', 4),
                                            size_type])
            self.boids = self.particles[2:]
            self.target = self.particles[0]
            self.predator = self.particles[1]

            self.boids['position'] = np.random.uniform(-0.25, +0.25, (n, 3))
            self.boids['velocity'] = np.random.uniform(-0.00, +0.00, (n, 3))
            self.boids['size'] = 4*ps
            self.boids['color'] = 1, 1, 1, 1

            self.target['size'] = 16*ps
            self.target['color'][:] = 1, 1, 0, 1
            self.predator['size'] = 16*ps
            self.predator['color'][:] = 1, 0, 0, 1
            self.target['position'][:] = 0.25, 0.0, 0

            # Time
            self._t = time.time()
            self._pos = 0.0, 0.0
            self._button = None

            width, height = self.physical_size
            gloo.set_viewport(0, 0, width, height)

            # Create program
            self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

            # Create vertex buffers
            self.vbo_position = gloo.VertexBuffer(self.particles['position']
                                                .copy())
            self.vbo_color = gloo.VertexBuffer(self.particles['color'].copy())
            self.vbo_size = gloo.VertexBuffer(self.particles['size'].copy())

            # Bind vertex buffers
            self.program['color'] = self.vbo_color
            self.program['size'] = self.vbo_size
            self.program['position'] = self.vbo_position

            gloo.set_state(clear_color=(0, 0, 0, 1), blend=True,
                        blend_func=('src_alpha', 'one'))

            self._timer = app.Timer('auto', connect=self.update, start=True)

            self.show()

        def on_resize(self, event):
            width, height = event.physical_size
            gloo.set_viewport(0, 0, width, height)

        def on_mouse_press(self, event):
            self._button = event.button
            self.on_mouse_move(event)

        def on_mouse_release(self, event):
            self._button = None
            self.on_mouse_move(event)

        def on_mouse_move(self, event):
            if not self._button:
                return
            w, h = self.size
            x, y = event.pos
            sx = 2 * x / float(w) - 1.0
            sy = - (2 * y / float(h) - 1.0)

            if self._button == 1:
                self.target['position'][:] = sx, sy, 0
            elif self._button == 2:
                self.predator['position'][:] = sx, sy, 0

        def on_draw(self, event):
            gloo.clear()
            # Draw
            self.program.draw('points')
            # Next iteration
            self._t = self.iteration(time.time() - self._t)

        def iteration(self, dt):
            t = self._t

            t += 0.5 * dt
            #self.target[...] = np.array([np.sin(t),np.sin(2*t),np.cos(3*t)])*.1

            t += 0.5 * dt
            #self.predator[...] = np.array([np.sin(t),np.sin(2*t),np.cos(3*t)])*.2

            self.boids['position_2'] = self.boids['position_1']
            self.boids['position_1'] = self.boids['position']
            n = len(self.boids)
            P = self.boids['position']
            V = self.boids['velocity']

            # Cohesion: steer to move toward the average position of local
            # flockmates
            C = -(P - P.sum(axis=0) / n)

            # Alignment: steer towards the average heading of local flockmates
            A = -(V - V.sum(axis=0) / n)

            # Repulsion: steer to avoid crowding local flockmates
            D, idxs = cKDTree(P).query(P, 5)
            M = np.repeat(D < 0.05, 3, axis=1).reshape(n, 5, 3)
            Z = np.repeat(P, 5, axis=0).reshape(n, 5, 3)
            R = -((P[idxs] - Z) * M).sum(axis=1)

            # Target : Follow target
            T = self.target['position'] - P

            # Predator : Move away from predator
            dP = P - self.predator['position']
            D = np.maximum(0, 0.3 -
                        np.sqrt(dP[:, 0] ** 2 +
                                dP[:, 1] ** 2 +
                                dP[:, 2] ** 2))
            D = np.repeat(D, 3, axis=0).reshape(n, 3)
            dP *= D

            #self.boids['velocity'] += 0.0005*C + 0.01*A + 0.01*R +
            #                           0.0005*T + 0.0025*dP
            self.boids['velocity'] += 0.0005 * C + 0.01 * \
                A + 0.01 * R + 0.0005 * T + 0.025 * dP
            self.boids['position'] += self.boids['velocity']

            self.vbo_position.set_data(self.particles['position'].copy())

            return t


    c = Canvas()
    app.run()


def galaxy():
    #!/usr/bin/env python
    import numpy as np
    import sys

    from vispy.util.transforms import perspective
    from vispy.util import transforms
    from vispy import gloo
    from vispy import app
    from vispy import io

    VERT_SHADER = """
    #version 120
    uniform mat4  u_model;
    uniform mat4  u_view;
    uniform mat4  u_projection;

    //sampler that maps [0, n] -> color according to blackbody law
    uniform sampler1D u_colormap;
    //index to sample the colormap at
    attribute float a_color_index;

    //size of the star
    attribute float a_size;
    //type
    //type 0 - stars
    //type 1 - dust
    //type 2 - h2a objects
    //type 3 - h2a objects
    attribute float a_type;
    attribute vec2  a_position;
    //brightness of the star
    attribute float a_brightness;

    varying vec3 v_color;
    void main (void)
    {
        gl_Position = u_projection * u_view * u_model * vec4(a_position, 0.0,1.0);

        //find base color according to physics from our sampler
        vec3 base_color = texture1D(u_colormap, a_color_index).rgb;
        //scale it down according to brightness
        v_color = base_color * a_brightness;


        if (a_size > 2.0)
        {
            gl_PointSize = a_size;
        } else {
            gl_PointSize = 0.0;
        }

        if (a_type == 2) {
            v_color *= vec3(2, 1, 1);
        }
        else if (a_type == 3) {
            v_color = vec3(.9);
        }
    }
    """

    FRAG_SHADER = """
    #version 120
    //star texture
    uniform sampler2D u_texture;
    //predicted color from black body
    varying vec3 v_color;

    void main()
    {
        //amount of intensity from the grayscale star
        float star_tex_intensity = texture2D(u_texture, gl_PointCoord).r;
        gl_FragColor = vec4(v_color * star_tex_intensity, 0.8);
    }
    """

    galaxy = Galaxy(10000)
    galaxy.reset(13000, 4000, 0.0004, 0.90, 0.90, 0.5, 200, 300)
    # coldest and hottest temperatures of out galaxy
    t0, t1 = 200.0, 10000.0
    # total number of discrete colors between t0 and t1
    n = 1000
    dt = (t1 - t0) / n

    # maps [0, n) -> colors
    # generate a linear interpolation of temperatures
    # then map the temperatures to colors using black body
    # color predictions
    colors = np.zeros(n, dtype=(np.float32, 3))
    for i in range(n):
        temperature = t0 + i * dt
        x, y, z = spectrum_to_xyz(bb_spectrum,
                                                temperature)
        r, g, b = xyz_to_rgb(SMPTEsystem, x, y, z)
        r = min((max(r, 0), 1))
        g = min((max(g, 0), 1))
        b = min((max(b, 0), 1))
        colors[i] = norm_rgb(r, g, b)


    # load the PNG that we use to blend the star with
    # to provide a circular look to each star.
    def load_galaxy_star_image():
        fname = io.load_data_file('galaxy/star-particle.png')
        raw_image = io.read_png(fname)

        return raw_image


    class Canvas(app.Canvas):

        def __init__(self):
            # setup initial width, height
            app.Canvas.__init__(self, keys='interactive', size=(800, 600))

            # create a new shader program
            self.program = gloo.Program(VERT_SHADER, FRAG_SHADER,
                                        count=len(galaxy))

            # load the star texture
            self.texture = gloo.Texture2D(load_galaxy_star_image(),
                                        interpolation='linear')
            self.program['u_texture'] = self.texture

            # construct the model, view and projection matrices
            self.view = transforms.translate((0, 0, -5))
            self.program['u_view'] = self.view

            self.model = np.eye(4, dtype=np.float32)
            self.program['u_model'] = self.model

            self.program['u_colormap'] = colors

            w, h = self.size
            self.projection = perspective(45.0, w / float(h), 1.0, 1000.0)
            self.program['u_projection'] = self.projection

            # start the galaxy to some decent point in the future
            galaxy.update(100000)
            data = self.__create_galaxy_vertex_data()

            # setup the VBO once the galaxy vertex data has been setup
            # bind the VBO for the first time
            self.data_vbo = gloo.VertexBuffer(data)
            self.program.bind(self.data_vbo)

            # setup blending
            gloo.set_state(clear_color=(0.0, 0.0, 0.03, 1.0),
                        depth_test=False, blend=True,
                        blend_func=('src_alpha', 'one'))

            self._timer = app.Timer('auto', connect=self.update, start=True)

        def __create_galaxy_vertex_data(self):
            data = np.zeros(len(galaxy),
                            dtype=[('a_size', np.float32),
                                ('a_position', np.float32, 2),
                                ('a_color_index', np.float32),
                                ('a_brightness', np.float32),
                                ('a_type', np.float32)])

            # see shader for parameter explanations
            pw, ph = self.physical_size
            data['a_size'] = galaxy['size'] * max(pw / 800.0, ph / 800.0)
            data['a_position'] = galaxy['position'] / 13000.0

            data['a_color_index'] = (galaxy['temperature'] - t0) / (t1 - t0)
            data['a_brightness'] = galaxy['brightness']
            data['a_type'] = galaxy['type']

            return data

        def on_resize(self, event):
            # setup the new viewport
            gloo.set_viewport(0, 0, *event.physical_size)
            # recompute the projection matrix
            w, h = event.size
            self.projection = perspective(45.0, w / float(h),
                                        1.0, 1000.0)
            self.program['u_projection'] = self.projection

        def on_draw(self, event):
            # update the galaxy
            galaxy.update(50000)  # in years !

            # recreate the numpy array that will be sent as the VBO data
            data = self.__create_galaxy_vertex_data()
            # update the VBO
            self.data_vbo.set_data(data)
            # bind the VBO to the GL context
            self.program.bind(self.data_vbo)

            # clear the screen and render
            gloo.clear(color=True, depth=True)
            self.program.draw('points')


    c = Canvas()
    c.show()

    if sys.flags.interactive == 0:
        app.run()


def fireworks():
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    # vispy: gallery 20

    """
    Example demonstrating simulation of fireworks using point sprites.
    (adapted from the "OpenGL ES 2.0 Programming Guide")

    This example demonstrates a series of explosions that last one second. The
    visualization during the explosion is highly optimized using a Vertex Buffer
    Object (VBO). After each explosion, vertex data for the next explosion are
    calculated, such that each explostion is unique.
    """

    import time
    import numpy as np
    from vispy import gloo, app

    # Create a texture
    radius = 32
    im1 = np.random.normal(
        0.8, 0.3, (radius * 2 + 1, radius * 2 + 1)).astype(np.float32)

    # Mask it with a disk
    L = np.linspace(-radius, radius, 2 * radius + 1)
    (X, Y) = np.meshgrid(L, L)
    im1 *= np.array((X ** 2 + Y ** 2) <= radius * radius, dtype='float32')

    # Set number of particles, you should be able to scale this to 100000
    N = 10000

    # Create vertex data container
    data = np.zeros(N, [('a_lifetime', np.float32),
                        ('a_startPosition', np.float32, 3),
                        ('a_endPosition', np.float32, 3)])


    VERT_SHADER = """
    uniform float u_time;
    uniform vec3 u_centerPosition;
    attribute float a_lifetime;
    attribute vec3 a_startPosition;
    attribute vec3 a_endPosition;
    varying float v_lifetime;

    void main () {
        if (u_time <= a_lifetime)
        {
            gl_Position.xyz = a_startPosition + (u_time * a_endPosition);
            gl_Position.xyz += u_centerPosition;
            gl_Position.y -= 1.0 * u_time * u_time;
            gl_Position.w = 1.0;
        }
        else
            gl_Position = vec4(-1000, -1000, 0, 0);

        v_lifetime = 1.0 - (u_time / a_lifetime);
        v_lifetime = clamp(v_lifetime, 0.0, 1.0);
        gl_PointSize = (v_lifetime * v_lifetime) * 40.0;
    }
    """

    # Deliberately add precision qualifiers to test automatic GLSL code conversion
    FRAG_SHADER = """
    #version 120
    precision highp float;
    uniform sampler2D texture1;
    uniform vec4 u_color;
    varying float v_lifetime;
    uniform highp sampler2D s_texture;

    void main()
    {
        highp vec4 texColor;
        texColor = texture2D(s_texture, gl_PointCoord);
        gl_FragColor = vec4(u_color) * texColor;
        gl_FragColor.a *= v_lifetime;
    }
    """


    class Canvas(app.Canvas):

        def __init__(self):
            app.Canvas.__init__(self, keys='interactive', size=(800, 600))

            # Create program
            self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
            self._program.bind(gloo.VertexBuffer(data))
            self._program['s_texture'] = gloo.Texture2D(im1)

            # Create first explosion
            self._new_explosion()

            # Enable blending
            gloo.set_state(blend=True, clear_color='black',
                        blend_func=('src_alpha', 'one'))

            gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])

            self._timer = app.Timer('auto', connect=self.update, start=True)

            self.show()

        def on_resize(self, event):
            width, height = event.physical_size
            gloo.set_viewport(0, 0, width, height)

        def on_draw(self, event):

            # Clear
            gloo.clear()

            # Draw
            self._program['u_time'] = time.time() - self._starttime
            self._program.draw('points')

            # New explosion?
            if time.time() - self._starttime > 1.5:
                self._new_explosion()

        def _new_explosion(self):

            # New centerpos
            centerpos = np.random.uniform(-0.5, 0.5, (3,))
            self._program['u_centerPosition'] = centerpos

            # New color, scale alpha with N
            alpha = 1.0 / N ** 0.08
            color = np.random.uniform(0.1, 0.9, (3,))

            self._program['u_color'] = tuple(color) + (alpha,)

            # Create new vertex data
            data['a_lifetime'] = np.random.normal(2.0, 0.5, (N,))
            data['a_startPosition'] = np.random.normal(0.0, 0.2, (N, 3))
            data['a_endPosition'] = np.random.normal(0.0, 1.2, (N, 3))

            # Set time to zero
            self._starttime = time.time()



    c = Canvas()
    app.run()


'''
----------------------------
Additional Galaxy Elements
----------------------------
'''

# -*- coding: utf-8 -*-
# vispy: testskip

"""
Colour Rendering of Spectra

by John Walker
http://www.fourmilab.ch/

Last updated: March 9, 2003

Converted to Python by Andrew Hutchins, sometime in early
2011.

    This program is in the public domain.
    The modifications are also public domain. (AH)

For complete information about the techniques employed in
this program, see the World-Wide Web document:

    http://www.fourmilab.ch/documents/specrend/

The xyz_to_rgb() function, which was wrong in the original
version of this program, was corrected by:

    Andrew J. S. Hamilton 21 May 1999
    Andrew.Hamilton@Colorado.EDU
    http://casa.colorado.edu/~ajsh/

who also added the gamma correction facilities and
modified constrain_rgb() to work by desaturating the
colour by adding white.

A program which uses these functions to plot CIE
"tongue" diagrams called "ppmcie" is included in
the Netpbm graphics toolkit:

    http://netpbm.sourceforge.net/

(The program was called cietoppm in earlier
versions of Netpbm.)

"""
import math

# A colour system is defined by the CIE x and y coordinates of
# its three primary illuminants and the x and y coordinates of
# the white point.

GAMMA_REC709 = 0

NTSCsystem = {"name": "NTSC",
              "xRed": 0.67, "yRed": 0.33,
              "xGreen": 0.21, "yGreen": 0.71,
              "xBlue": 0.14, "yBlue": 0.08,
              "xWhite": 0.3101, "yWhite": 0.3163, "gamma": GAMMA_REC709}

EBUsystem = {"name": "SUBU (PAL/SECAM)",
             "xRed": 0.64, "yRed": 0.33,
             "xGreen": 0.29, "yGreen": 0.60,
             "xBlue": 0.15, "yBlue": 0.06,
             "xWhite": 0.3127, "yWhite": 0.3291, "gamma": GAMMA_REC709}
SMPTEsystem = {"name": "SMPTE",
               "xRed": 0.63, "yRed": 0.34,
               "xGreen": 0.31, "yGreen": 0.595,
               "xBlue": 0.155, "yBlue": 0.07,
               "xWhite": 0.3127, "yWhite": 0.3291, "gamma": GAMMA_REC709}

HDTVsystem = {"name": "HDTV",
              "xRed": 0.67, "yRed": 0.33,
              "xGreen": 0.21, "yGreen": 0.71,
              "xBlue": 0.15, "yBlue": 0.06,
              "xWhite": 0.3127, "yWhite": 0.3291, "gamma": GAMMA_REC709}

CIEsystem = {"name": "CIE",
             "xRed": 0.7355, "yRed": 0.2645,
             "xGreen": 0.2658, "yGreen": 0.7243,
             "xBlue": 0.1669, "yBlue": 0.0085,
             "xWhite": 0.3333333333, "yWhite": 0.3333333333,
             "gamma": GAMMA_REC709}

Rec709system = {"name": "CIE REC709",
                "xRed": 0.64, "yRed": 0.33,
                "xGreen": 0.30, "yGreen": 0.60,
                "xBlue": 0.15, "yBlue": 0.06,
                "xWhite": 0.3127, "yWhite": 0.3291,
                "gamma": GAMMA_REC709}


def upvp_to_xy(up, vp):
    xc = (9 * up) / ((6 * up) - (16 * vp) + 12)
    yc = (4 * vp) / ((6 * up) - (16 * vp) + 12)
    return(xc, yc)


def xy_toupvp(xc, yc):
    up = (4 * xc) / ((-2 * xc) + (12 * yc) + 3)
    vp = (9 * yc) / ((-2 * xc) + (12 * yc) + 3)
    return(up, vp)


def xyz_to_rgb(cs, xc, yc, zc):
    """
    Given an additive tricolour system CS, defined by the CIE x
    and y chromaticities of its three primaries (z is derived
    trivially as 1-(x+y)), and a desired chromaticity (XC, YC,
    ZC) in CIE space, determine the contribution of each
    primary in a linear combination which sums to the desired
    chromaticity.  If the  requested chromaticity falls outside
    the Maxwell  triangle (colour gamut) formed by the three
    primaries, one of the r, g, or b weights will be negative.

    Caller can use constrain_rgb() to desaturate an
    outside-gamut colour to the closest representation within
    the available gamut and/or norm_rgb to normalise the RGB
    components so the largest nonzero component has value 1.
    """

    xr = cs["xRed"]
    yr = cs["yRed"]
    zr = 1 - (xr + yr)
    xg = cs["xGreen"]
    yg = cs["yGreen"]
    zg = 1 - (xg + yg)
    xb = cs["xBlue"]
    yb = cs["yBlue"]
    zb = 1 - (xb + yb)
    xw = cs["xWhite"]
    yw = cs["yWhite"]
    zw = 1 - (xw + yw)

    rx = (yg * zb) - (yb * zg)
    ry = (xb * zg) - (xg * zb)
    rz = (xg * yb) - (xb * yg)
    gx = (yb * zr) - (yr * zb)
    gy = (xr * zb) - (xb * zr)
    gz = (xb * yr) - (xr * yb)
    bx = (yr * zg) - (yg * zr)
    by = (xg * zr) - (xr * zg)
    bz = (xr * yg) - (xg * yr)

    rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw
    gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw
    bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw

    rx = rx / rw
    ry = ry / rw
    rz = rz / rw
    gx = gx / gw
    gy = gy / gw
    gz = gz / gw
    bx = bx / bw
    by = by / bw
    bz = bz / bw

    r = (rx * xc) + (ry * yc) + (rz * zc)
    g = (gx * xc) + (gy * yc) + (gz * zc)
    b = (bx * xc) + (by * yc) + (bz * zc)

    return(r, g, b)


def inside_gamut(r, g, b):
    """
     Test whether a requested colour is within the gamut
     achievable with the primaries of the current colour
     system.  This amounts simply to testing whether all the
     primary weights are non-negative. */
    """
    return (r >= 0) and (g >= 0) and (b >= 0)


def constrain_rgb(r, g, b):
    """
    If the requested RGB shade contains a negative weight for
    one of the primaries, it lies outside the colour gamut
    accessible from the given triple of primaries.  Desaturate
    it by adding white, equal quantities of R, G, and B, enough
    to make RGB all positive.  The function returns 1 if the
    components were modified, zero otherwise.
    """
    # Amount of white needed is w = - min(0, *r, *g, *b)
    w = -min([0, r, g, b])  # I think?

    # Add just enough white to make r, g, b all positive.
    if w > 0:
        r += w
        g += w
        b += w
    return(r, g, b)


def gamma_correct(cs, c):
    """
    Transform linear RGB values to nonlinear RGB values. Rec.
    709 is ITU-R Recommendation BT. 709 (1990) ``Basic
    Parameter Values for the HDTV Standard for the Studio and
    for International Programme Exchange'', formerly CCIR Rec.
    709. For details see

       http://www.poynton.com/ColorFAQ.html
       http://www.poynton.com/GammaFAQ.html
    """
    gamma = cs.gamma

    if gamma == GAMMA_REC709:
        cc = 0.018
        if c < cc:
            c = ((1.099 * math.pow(cc, 0.45)) - 0.099) / cc
        else:
            c = (1.099 * math.pow(c, 0.45)) - 0.099
    else:
        c = math.pow(c, 1.0 / gamma)
    return(c)


def gamma_correct_rgb(cs, r, g, b):
    r = gamma_correct(cs, r)
    g = gamma_correct(cs, g)
    b = gamma_correct(cs, b)
    return (r, g, b)


def norm_rgb(r, g, b):
    """
    Normalise RGB components so the most intense (unless all
    are zero) has a value of 1.
    """
    greatest = max([r, g, b])

    if greatest > 0:
        r /= greatest
        g /= greatest
        b /= greatest
    return(r, g, b)


# spec_intens is a function
def spectrum_to_xyz(spec_intens, temp):
    """
    Calculate the CIE X, Y, and Z coordinates corresponding to
    a light source with spectral distribution given by  the
    function SPEC_INTENS, which is called with a series of
    wavelengths between 380 and 780 nm (the argument is
    expressed in meters), which returns emittance at  that
    wavelength in arbitrary units.  The chromaticity
    coordinates of the spectrum are returned in the x, y, and z
    arguments which respect the identity:

        x + y + z = 1.

    CIE colour matching functions xBar, yBar, and zBar for
    wavelengths from 380 through 780 nanometers, every 5
    nanometers.  For a wavelength lambda in this range::

        cie_colour_match[(lambda - 380) / 5][0] = xBar
        cie_colour_match[(lambda - 380) / 5][1] = yBar
        cie_colour_match[(lambda - 380) / 5][2] = zBar

    AH Note 2011: This next bit is kind of irrelevant on modern
    hardware. Unless you are desperate for speed.
    In which case don't use the Python version!

    To save memory, this table can be declared as floats
    rather than doubles; (IEEE) float has enough
    significant bits to represent the values. It's declared
    as a double here to avoid warnings about "conversion
    between floating-point types" from certain persnickety
    compilers.
    """

    cie_colour_match = [
        [0.0014, 0.0000, 0.0065],
        [0.0022, 0.0001, 0.0105],
        [0.0042, 0.0001, 0.0201],
        [0.0076, 0.0002, 0.0362],
        [0.0143, 0.0004, 0.0679],
        [0.0232, 0.0006, 0.1102],
        [0.0435, 0.0012, 0.2074],
        [0.0776, 0.0022, 0.3713],
        [0.1344, 0.0040, 0.6456],
        [0.2148, 0.0073, 1.0391],
        [0.2839, 0.0116, 1.3856],
        [0.3285, 0.0168, 1.6230],
        [0.3483, 0.0230, 1.7471],
        [0.3481, 0.0298, 1.7826],
        [0.3362, 0.0380, 1.7721],
        [0.3187, 0.0480, 1.7441],
        [0.2908, 0.0600, 1.6692],
        [0.2511, 0.0739, 1.5281],
        [0.1954, 0.0910, 1.2876],
        [0.1421, 0.1126, 1.0419],
        [0.0956, 0.1390, 0.8130],
        [0.0580, 0.1693, 0.6162],
        [0.0320, 0.2080, 0.4652],
        [0.0147, 0.2586, 0.3533],
        [0.0049, 0.3230, 0.2720],
        [0.0024, 0.4073, 0.2123],
        [0.0093, 0.5030, 0.1582],
        [0.0291, 0.6082, 0.1117],
        [0.0633, 0.7100, 0.0782],
        [0.1096, 0.7932, 0.0573],
        [0.1655, 0.8620, 0.0422],
        [0.2257, 0.9149, 0.0298],
        [0.2904, 0.9540, 0.0203],
        [0.3597, 0.9803, 0.0134],
        [0.4334, 0.9950, 0.0087],
        [0.5121, 1.0000, 0.0057],
        [0.5945, 0.9950, 0.0039],
        [0.6784, 0.9786, 0.0027],
        [0.7621, 0.9520, 0.0021],
        [0.8425, 0.9154, 0.0018],
        [0.9163, 0.8700, 0.0017],
        [0.9786, 0.8163, 0.0014],
        [1.0263, 0.7570, 0.0011],
        [1.0567, 0.6949, 0.0010],
        [1.0622, 0.6310, 0.0008],
        [1.0456, 0.5668, 0.0006],
        [1.0026, 0.5030, 0.0003],
        [0.9384, 0.4412, 0.0002],
        [0.8544, 0.3810, 0.0002],
        [0.7514, 0.3210, 0.0001],
        [0.6424, 0.2650, 0.0000],
        [0.5419, 0.2170, 0.0000],
        [0.4479, 0.1750, 0.0000],
        [0.3608, 0.1382, 0.0000],
        [0.2835, 0.1070, 0.0000],
        [0.2187, 0.0816, 0.0000],
        [0.1649, 0.0610, 0.0000],
        [0.1212, 0.0446, 0.0000],
        [0.0874, 0.0320, 0.0000],
        [0.0636, 0.0232, 0.0000],
        [0.0468, 0.0170, 0.0000],
        [0.0329, 0.0119, 0.0000],
        [0.0227, 0.0082, 0.0000],
        [0.0158, 0.0057, 0.0000],
        [0.0114, 0.0041, 0.0000],
        [0.0081, 0.0029, 0.0000],
        [0.0058, 0.0021, 0.0000],
        [0.0041, 0.0015, 0.0000],
        [0.0029, 0.0010, 0.0000],
        [0.0020, 0.0007, 0.0000],
        [0.0014, 0.0005, 0.0000],
        [0.0010, 0.0004, 0.0000],
        [0.0007, 0.0002, 0.0000],
        [0.0005, 0.0002, 0.0000],
        [0.0003, 0.0001, 0.0000],
        [0.0002, 0.0001, 0.0000],
        [0.0002, 0.0001, 0.0000],
        [0.0001, 0.0000, 0.0000],
        [0.0001, 0.0000, 0.0000],
        [0.0001, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]]

    X = 0
    Y = 0
    Z = 0
    # lambda = 380; lambda < 780.1; i++, lambda += 5) {
    for i, lamb in enumerate(range(380, 780, 5)):
        Me = spec_intens(lamb, temp)
        X += Me * cie_colour_match[i][0]
        Y += Me * cie_colour_match[i][1]
        Z += Me * cie_colour_match[i][2]
    XYZ = (X + Y + Z)
    x = X / XYZ
    y = Y / XYZ
    z = Z / XYZ

    return(x, y, z)


def bb_spectrum(wavelength, bbTemp=5000):
    """
    Calculate, by Planck's radiation law, the emittance of a black body
    of temperature bbTemp at the given wavelength (in metres).  */
    """
    wlm = wavelength * 1e-9  # Convert to metres
    return (3.74183e-16 *
            math.pow(wlm, -5.0)) / (math.exp(1.4388e-2 / (wlm * bbTemp)) - 1.0)

    """  Built-in test program which displays the x, y, and Z and RGB
    values for black body spectra from 1000 to 10000 degrees kelvin.
    When run, this program should produce the following output:

    Temperature       x      y      z       R     G     B
    -----------    ------ ------ ------   ----- ----- -----
       1000 K      0.6528 0.3444 0.0028   1.000 0.007 0.000 (Approximation)
       1500 K      0.5857 0.3931 0.0212   1.000 0.126 0.000 (Approximation)
       2000 K      0.5267 0.4133 0.0600   1.000 0.234 0.010
       2500 K      0.4770 0.4137 0.1093   1.000 0.349 0.067
       3000 K      0.4369 0.4041 0.1590   1.000 0.454 0.151
       3500 K      0.4053 0.3907 0.2040   1.000 0.549 0.254
       4000 K      0.3805 0.3768 0.2428   1.000 0.635 0.370
       4500 K      0.3608 0.3636 0.2756   1.000 0.710 0.493
       5000 K      0.3451 0.3516 0.3032   1.000 0.778 0.620
       5500 K      0.3325 0.3411 0.3265   1.000 0.837 0.746
       6000 K      0.3221 0.3318 0.3461   1.000 0.890 0.869
       6500 K      0.3135 0.3237 0.3628   1.000 0.937 0.988
       7000 K      0.3064 0.3166 0.3770   0.907 0.888 1.000
       7500 K      0.3004 0.3103 0.3893   0.827 0.839 1.000
       8000 K      0.2952 0.3048 0.4000   0.762 0.800 1.000
       8500 K      0.2908 0.3000 0.4093   0.711 0.766 1.000
       9000 K      0.2869 0.2956 0.4174   0.668 0.738 1.000
       9500 K      0.2836 0.2918 0.4246   0.632 0.714 1.000
      10000 K      0.2807 0.2884 0.4310   0.602 0.693 1.000
"""

if __name__ == "__main__":
    print("Temperature       x      y      z       R     G     B\n")
    print("-----------    ------ ------ ------   ----- ----- -----\n")

    for t in range(1000, 10000, 500):  # (t = 1000; t <= 10000; t+= 500) {
        x, y, z = spectrum_to_xyz(bb_spectrum, t)

        r, g, b = xyz_to_rgb(SMPTEsystem, x, y, z)

        print("  %5.0f K      %.4f %.4f %.4f   " % (t, x, y, z))

        # I omit the approximation bit here.
        r, g, b = constrain_rgb(r, g, b)
        r, g, b = norm_rgb(r, g, b)
        print("%.3f %.3f %.3f" % (r, g, b))


# -*- coding: utf-8 -*-
# vispy: testskip

# -----------------------------------------------------------------------------
#  A Galaxy Simulator based on the density wave theory
#  (c) 2012 Ingo Berg
#
#  Simulating a Galaxy with the density wave theory
#  http://beltoforion.de/galaxy/galaxy_en.html
#
#  Python version(c) 2014 Nicolas P.Rougier
# -----------------------------------------------------------------------------
import math
import numpy as np


class Galaxy(object):
    """ Galaxy simulation using the density wave theory """

    def __init__(self, n=20000):
        """ Initialize galaxy """

        # Eccentricity of the innermost ellipse
        self._inner_eccentricity = 0.8

        # Eccentricity of the outermost ellipse
        self._outer_eccentricity = 1.0

        # Velocity at the innermost core in km/s
        self._center_velocity = 30

        # Velocity at the core edge in km/s
        self._inner_velocity = 200

        # Velocity at the edge of the disk in km/s
        self._outer_velocity = 300

        # Angular offset per parsec
        self._angular_offset = 0.019

        # Inner core radius
        self._core_radius = 6000

        # Galaxy radius
        self._galaxy_radius = 15000

        # The radius after which all density waves must have circular shape
        self._distant_radius = 0

        # Distribution of stars
        self._star_distribution = 0.45

        # Angular velocity of the density waves
        self._angular_velocity = 0.000001

        # Number of stars
        self._stars_count = n

        # Number of dust particles
        self._dust_count = int(self._stars_count * 0.75)

        # Number of H-II regions
        self._h2_count = 200

        # Particles
        dtype = [('theta',       np.float32),
                 ('velocity',    np.float32),
                 ('angle',       np.float32),
                 ('m_a',         np.float32),
                 ('m_b',         np.float32),
                 ('size',        np.float32),
                 ('type',        np.float32),
                 ('temperature', np.float32),
                 ('brightness',  np.float32),
                 ('position',    np.float32, 2)]
        n = self._stars_count + self._dust_count + 2*self._h2_count
        self._particles = np.zeros(n, dtype=dtype)

        i0 = 0
        i1 = i0 + self._stars_count
        self._stars = self._particles[i0:i1]
        self._stars['size'] = 3.
        self._stars['type'] = 0

        i0 = i1
        i1 = i0 + self._dust_count
        self._dust = self._particles[i0:i1]
        self._dust['size'] = 64
        self._dust['type'] = 1

        i0 = i1
        i1 = i0 + self._h2_count
        self._h2a = self._particles[i0:i1]
        self._h2a['size'] = 0
        self._h2a['type'] = 2

        i0 = i1
        i1 = i0 + self._h2_count
        self._h2b = self._particles[i0:i1]
        self._h2b['size'] = 0
        self._h2b['type'] = 3

    def __len__(self):
        """ Number of particles """

        if self._particles is not None:
            return len(self._particles)
        return 0

    def __getitem__(self, key):
        """ x.__getitem__(y) <==> x[y] """

        if self._particles is not None:
            return self._particles[key]
        return None

    def reset(self, rad, radCore, deltaAng,
              ex1, ex2, sigma, velInner, velOuter):

        # Initialize parameters
        # ---------------------
        self._inner_eccentricity = ex1
        self._outer_eccentricity = ex2
        self._inner_velocity = velInner
        self._outer_velocity = velOuter
        self._angular_offset = deltaAng
        self._core_radius = radCore
        self._galaxy_radius = rad
        self._distant_radius = self._galaxy_radius * 2
        self.m_sigma = sigma

        # Initialize stars
        # ----------------
        stars = self._stars
        R = np.random.normal(0, sigma, len(stars)) * self._galaxy_radius
        stars['m_a'] = R
        stars['angle'] = 90 - R * self._angular_offset
        stars['theta'] = np.random.uniform(0, 360, len(stars))
        stars['temperature'] = np.random.uniform(3000, 9000, len(stars))
        stars['brightness'] = np.random.uniform(0.05, 0.25, len(stars))
        stars['velocity'] = 0.000005

        for i in range(len(stars)):
            stars['m_b'][i] = R[i] * self.eccentricity(R[i])

        # Initialize dust
        # ---------------
        dust = self._dust
        X = np.random.uniform(0, 2*self._galaxy_radius, len(dust))
        Y = np.random.uniform(-self._galaxy_radius, self._galaxy_radius,
                              len(dust))
        R = np.sqrt(X*X+Y*Y)
        dust['m_a'] = R
        dust['angle'] = R * self._angular_offset
        dust['theta'] = np.random.uniform(0, 360, len(dust))
        dust['velocity'] = 0.000005
        dust['temperature'] = 6000 + R/4
        dust['brightness'] = np.random.uniform(0.01, 0.02)
        for i in range(len(dust)):
            dust['m_b'][i] = R[i] * self.eccentricity(R[i])

        # Initialise H-II
        # ---------------
        h2a, h2b = self._h2a, self._h2b
        X = np.random.uniform(-self._galaxy_radius, self._galaxy_radius,
                              len(h2a))
        Y = np.random.uniform(-self._galaxy_radius, self._galaxy_radius,
                              len(h2a))
        R = np.sqrt(X*X+Y*Y)

        h2a['m_a'] = R
        h2b['m_a'] = R + 1000

        h2a['angle'] = R * self._angular_offset
        h2b['angle'] = h2a['angle']

        h2a['theta'] = np.random.uniform(0, 360, len(h2a))
        h2b['theta'] = h2a['theta']

        h2a['velocity'] = 0.000005
        h2b['velocity'] = 0.000005

        h2a['temperature'] = np.random.uniform(3000, 9000, len(h2a))
        h2b['temperature'] = h2a['temperature']

        h2a['brightness'] = np.random.uniform(0.005, 0.010, len(h2a))
        h2b['brightness'] = h2a['brightness']

        for i in range(len(h2a)):
            h2a['m_b'][i] = R[i] * self.eccentricity(R[i])
        h2b['m_b'] = h2a['m_b']

    def update(self, timestep=100000):
        """ Update simulation """

        self._particles['theta'] += self._particles['velocity'] * timestep

        P = self._particles
        a, b = P['m_a'], P['m_b']
        theta, beta = P['theta'], -P['angle']

        alpha = theta * math.pi / 180.0
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        P['position'][:, 0] = a*cos_alpha*cos_beta - b*sin_alpha*sin_beta
        P['position'][:, 1] = a*cos_alpha*sin_beta + b*sin_alpha*cos_beta

        D = np.sqrt(((self._h2a['position'] -
                    self._h2b['position'])**2).sum(axis=1))
        S = np.maximum(1, ((1000-D)/10) - 50)
        self._h2a['size'] = 2.0*S
        self._h2b['size'] = S/6.0

    def eccentricity(self, r):

        # Core region of the galaxy. Innermost part is round
        # eccentricity increasing linear to the border of the core.
        if r < self._core_radius:
            return 1 + (r / self._core_radius) * (self._inner_eccentricity-1)

        elif r > self._core_radius and r <= self._galaxy_radius:
            a = self._galaxy_radius - self._core_radius
            b = self._outer_eccentricity - self._inner_eccentricity
            return self._inner_eccentricity + (r - self._core_radius) / a * b

        # Eccentricity is slowly reduced to 1.
        elif r > self._galaxy_radius and r < self._distant_radius:
            a = self._distant_radius - self._galaxy_radius
            b = 1 - self._outer_eccentricity
            return self._outer_eccentricity + (r - self._galaxy_radius) / a * b

        else:
            return 1
