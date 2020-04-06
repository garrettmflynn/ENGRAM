from __future__ import division
from __future__ import print_function
import vispy
import numpy as np
import math

def select(shader="engramv1",id=None):
    selection = {
        "engramv1" : engramv1,
        "engram": engram,
        "shadertoy" : shadertoy,
        "oscilloscope" : oscilloscope,
    }
    
    # Get the function from switcher dictionary
    func = selection.get(shader, lambda: "Invalid event parser")
    # Execute the function
    if (shader != 'engramv1') and (shader != 'engram'):
        return func()
    else:
        return func(id)


# ____________________________ CUSTOM ENVIRONMENTS ____________________________
def engramv1(id):
    from vispy import app, gloo, visuals
    from vispy.util.transforms import perspective, translate, rotate

    # ____________________________ DATA ____________________________

    # Load the xyz coordinates and corresponding subject name :
    # mat = np.load(download_file('xyz_sample.npz', astype='example_data'))
    # xyz, subjects = mat['xyz'], mat['subjects']

    metadata = id.durations[0].metadata
    binary = id.durations[0].bins[0]

    positions = metadata['stream_pattern']['positions']
    assignments = metadata['stream_pattern']['hierarchy']

    SPACING = 6 # In MNI coordinates

    n_dims = np.shape(assignments)[1]
    existing_levels = []
    intersection_matrices = {}
    intersection_matrices['indices'] = np.empty([])
    intersection_matrices['streams'] = np.empty([])
    intersection_matrices['positions'] = np.empty([])
    intersection_matrices['hierarchy_lookup'] = []

    # Derive Intersection Matrix
    for k, hierarchy in enumerate(assignments):
        if '' not in hierarchy:
            intersection = []
            for level, v in enumerate(hierarchy):
                    if len(intersection_matrices['hierarchy_lookup']) <= level:
                        intersection_matrices['hierarchy_lookup'].append([])
                        intersection_matrices['hierarchy_lookup'][level].extend([v])
                    if v not in intersection_matrices['hierarchy_lookup'][level]:
                        intersection_matrices['hierarchy_lookup'][level].extend([v])
                    distinction = np.where(np.asarray(intersection_matrices['hierarchy_lookup'][level]) == v)[0][0]
                    intersection.append(distinction)
            if intersection:
                intersection = np.expand_dims(np.array(intersection), axis=0)
                pos = np.expand_dims(np.array(positions[k]), axis=0)
                stream = np.expand_dims(np.array(metadata['all_streams'][k]), axis=0)
                source_count = np.expand_dims(np.arange(np.array(sum(binary.nD_labels['1D']==metadata['all_streams'][k]))), axis=0)
                if k is 0:
                    intersection_matrices['indices'] = intersection
                    intersection_matrices['streams'] = stream
                    intersection_matrices['sources'] = source_count
                    intersection_matrices['positions'] = pos
                    
                else:
                    intersection_matrices['indices'] = np.append(intersection_matrices['indices'], intersection,axis=0)
                    intersection_matrices['streams'] = np.append(intersection_matrices['streams'], stream,axis=0)
                    intersection_matrices['sources'] = np.append(intersection_matrices['sources'],source_count+intersection_matrices['sources'][k-1][-1]+1,axis=0)
                    intersection_matrices['positions'] = np.append(intersection_matrices['positions'], pos,axis=0)

    xyz, data = position_slicer(intersection_matrices,method='full')
    X = xyz[:,0]
    Y = xyz[:,1]
    Z = xyz[:,2]
    X = 6*((np.asarray(X) - min(X))/(max(X)-min(X)) - .5)
    Y = 6*((np.asarray(Y) - min(Y))/(max(Y)-min(Y)) - .5)
    Z = 6*((np.asarray(Z) - min(Z))/(max(Z)-min(Z)) - .5)

    # Convert binary array into visualizable continuous values
    TRAIL = 50
    spikes = binary.data.T[0:100000]
    one_array = np.where(spikes == 1)
    if not not one_array:
        lb_1 = one_array[0]-TRAIL
        ub_1 = one_array[0]
        lb_2 = one_array[0]
        ub_2 = one_array[0]+TRAIL
        for ii in range(len(lb_1)):
            if ub_2[ii] > len(spikes):
                ub_2[ii] = len(spikes)
            if lb_1[ii] < 0:
                lb_1[ii] = 0
                    
            spikes[lb_1[ii]:ub_1[ii],one_array[1][ii]] += np.linspace(0,1,ub_1[ii]-lb_1[ii])
            spikes[lb_2[ii]:ub_2[ii],one_array[1][ii]] += np.linspace(1,0,ub_2[ii]-lb_2[ii])

    
    
    xyzs = np.zeros((intersection_matrices['sources'].size,4)).astype('float32')
    xyzs[:,0:3] = xyz
    data['a_rotation'] = np.repeat(
        np.random.uniform(0, .1, (intersection_matrices['sources'].size, 4)).astype(np.float32), 1, axis=0)


    vert = """
    #version 120

    // Uniforms
    // ------------------------------------
    uniform mat4 u_model;
    uniform mat4 u_view;
    uniform mat4 u_projection;
    uniform float u_size;
    uniform float u_frame;

    // Attributes
    // ------------------------------------
    attribute vec4 a_xyzs;
    attribute vec4 a_color;
    attribute vec4 a_rotation;

    // Varying
    // ------------------------------------
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

        float x_off = sin(u_frame + a_xyzs.y) * .1;
        float y_off = sin(u_frame + a_xyzs.x) * .3;

        float x1 = a_xyzs.x;
        float y1 = a_xyzs.y;
        float z1 = a_xyzs.z;

        vec2 xy = vec2(x1,y1);

        mat4 R = build_rotation(a_rotation.xyz, a_rotation.w);
        gl_Position = u_projection * u_view * u_model * vec4(xy,z1,1);


        #define SPIKE_MULT 5

        gl_PointSize = u_size*5 + (u_size*SPIKE_MULT*a_xyzs.w);

        float hue = x1;
        float sat = y1;
        float val = sin(a_xyzs.y*a_xyzs.y);

        v_color = a_color;
        v_color.r = v_color.r*a_xyzs.w;
    }
    """

    frag = """
    #version 120

    // Varying
    // ------------------------------------
    varying vec4 v_color;
    varying float v_size;

    void main()
    {
        // Point shaping function
        float d = 2*(length(gl_PointCoord.xy - vec2(0.5,0.5)));
        gl_FragColor = vec4(v_color.rgb, v_color.a*(1-d));
    }
    """


    # ----------------- Canvas class --------------------
    class Canvas(app.Canvas):

        def __init__(self):
            app.Canvas.__init__(self, keys='interactive', size=(800, 800))

            self.translate = 20 # Z Start Location
            self.program = gloo.Program(vert, frag)
            self.view = translate((0, 0, -self.translate))
            self.model = np.eye(4, dtype=np.float32)
            self.projection = np.eye(4, dtype=np.float32)

             # t1 = np.array(X.shape[0])
            # for ind, val in enumerate(X):
            #     t1[ind] = Text('Text in root scene (24 pt)', parent=c.scene, color='red')
            #     t1[ind].font_size = 24
            #     t1[ind] = pos = val, Y[ind], Z[ind]

            self.font_size = self.physical_size[1]/24
            self.text_pos = self.physical_size[0]/2, 5*self.physical_size[1]/6
            self.text = visuals.TextVisual(' ', bold=True)
            self.text.color = 'white'

            self.apply_zoom()

            self.program.bind(gloo.VertexBuffer(data))
            self.program['u_model'] = self.model
            self.program['u_view'] = self.view
            self.program['u_size'] = 50/(self.translate)

            self.theta = 0
            self.phi = 0
            self.frame = 0
            self.stop_rotation = False

            gloo.set_state('translucent', depth_test=False)

            self.program['u_frame'] = 0.0
            xyzs[:, 3] = spikes[int(self.program['u_frame'][0])]
            self.program['a_xyzs'] = xyzs

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
            self.frame += 2000/60
            if self.frame > len(spikes):
                self.frame = 0
            else:
                self.program['u_frame'] = self.frame
            
            self.text.text = 't = ' + str(self.frame//2000) + ' s'

            xyzs[:, 3] = spikes[int(self.program['u_frame'][0])]
            self.program['a_xyzs'] = xyzs
            self.update()

        def on_resize(self, event):
            self.apply_zoom()
            self.update_text()


        def on_mouse_wheel(self, event):
            self.translate += event.delta[1]
            self.translate = max(2, self.translate)
            self.view = translate((0, 0, -self.translate))
            self.program['u_view'] = self.view
            self.program['u_size'] = 50/( self.translate )
            self.update()
            self.update_text()


        def on_draw(self, event):
            gloo.clear('black')
            # gloo.clear(color='white')
            self.program.draw('points')
            self.text.draw()

        def apply_zoom(self):
            width, height = self.physical_size
            vp = (0, 0, width, height)
            gloo.set_viewport(vp)
            self.projection = perspective(45.0, width / float(height), 1.0, 1000.0)
            self.program['u_projection'] = self.projection
            self.text.transforms.configure(canvas=self, viewport=vp)

        def update_text(self):
            self.text.font_size = self.font_size
            self.text.pos = self.text_pos

            self.update()


    c = Canvas()
    app.run()
def engram(id):

    from .gui import Engram
    from visbrain.objects import RoiObj
    from .objects import SourceObj, ConnectObj
    from visbrain.io import download_file

    # Create an empty kwargs dictionnary :
    kwargs = {}

    # ____________________________ DATA ____________________________

    # Load the xyz coordinates and corresponding subject name :
    # mat = np.load(download_file('xyz_sample.npz', astype='example_data'))
    # xyz, subjects = mat['xyz'], mat['subjects']

    metadata = id.durations[0].metadata
    binary = id.durations[0].bins[0]

    positions = metadata['stream_pattern']['positions']
    assignments = metadata['stream_pattern']['hierarchy']

    SPACING = 6 # In MNI coordinates
    INITIAL_DISTINCTIONS = []

    n_dims = np.shape(assignments)[1]
    existing_levels = []
    intersection_matrices = {}
    intersection_matrices['indices'] = np.empty([])
    intersection_matrices['streams'] = np.empty([])
    intersection_matrices['positions'] = np.empty([])
    intersection_matrices['hierarchy_lookup'] = []

    # Derive Intersection Matrix
    for k, hierarchy in enumerate(assignments):
        if '' not in hierarchy:
            intersection = []
            for level, v in enumerate(hierarchy):
                    if len(intersection_matrices['hierarchy_lookup']) <= level:
                        intersection_matrices['hierarchy_lookup'].append([])
                        intersection_matrices['hierarchy_lookup'][level].extend([v])
                    if v not in intersection_matrices['hierarchy_lookup'][level]:
                        intersection_matrices['hierarchy_lookup'][level].extend([v])
                    distinction = np.where(np.asarray(intersection_matrices['hierarchy_lookup'][level]) == v)[0][0]
                    intersection.append(distinction)
            if intersection:
                intersection = np.expand_dims(np.array(intersection), axis=0)
                pos = np.expand_dims(np.array(positions[k]), axis=0)
                stream = np.expand_dims(np.array(metadata['all_streams'][k]), axis=0)
                source_count = np.expand_dims(np.arange(np.array(sum(binary.nD_labels['1D']==metadata['all_streams'][k]))), axis=0)
                if k is 0:
                    intersection_matrices['indices'] = intersection
                    intersection_matrices['streams'] = stream
                    intersection_matrices['sources'] = source_count
                    intersection_matrices['positions'] = pos
                    
                else:
                    intersection_matrices['indices'] = np.append(intersection_matrices['indices'], intersection,axis=0)
                    intersection_matrices['streams'] = np.append(intersection_matrices['streams'], stream,axis=0)
                    intersection_matrices['sources'] = np.append(intersection_matrices['sources'],source_count+intersection_matrices['sources'][k-1][-1]+1,axis=0)
                    intersection_matrices['positions'] = np.append(intersection_matrices['positions'], pos,axis=0)

    xyz = position_slicer(intersection_matrices,method=INITIAL_DISTINCTIONS,ignore_streams=True)
    
    # Convert binary array into visualizable continuous values
    print('Calculating spike durations')
    TRAIL = 100
    spikes = binary.data.T[0:10000]
    one_array = np.where(spikes == 1)
    if not not one_array:
        lb_1 = one_array[0]-TRAIL
        ub_1 = one_array[0]
        lb_2 = one_array[0]
        ub_2 = one_array[0]+TRAIL
        for ii in range(len(lb_1)):
            if ub_2[ii] > len(spikes):
                ub_2[ii] = len(spikes)
            if lb_1[ii] < 0:
                lb_1[ii] = 0
                    
            spikes[lb_1[ii]:ub_1[ii],one_array[1][ii]] += np.linspace(0,1,ub_1[ii]-lb_1[ii])
            spikes[lb_2[ii]:ub_2[ii],one_array[1][ii]] += np.linspace(1,0,ub_2[ii]-lb_2[ii])
        
    spikes = spikes.T

    N = xyz.shape[0]  # Number of electrodes

    text = ['S' + str(k) for k in range(N)]
    s_obj = SourceObj('SourceObj1', xyz, data=spikes,color='crimson', text=text,alpha=.5,
                    edge_width=2., radius_min=1., radius_max=25.)

    
    connect = np.zeros((N, N,np.shape(spikes)[1]))
    valid = np.empty((N, N,np.shape(spikes)[1]))
    edges = np.arange(N)    

    print('Calculating connectivity')
    for ind,activity in enumerate(spikes):
        if ind < len(spikes):
            edge_activity = spikes[ind+1:-1]
            weight = edge_activity + activity
            valid = ((edge_activity > 0) & (activity > 0)).astype('int')
            connect[ind,ind+1:-1] = weight * valid

    umin = 0
    umax = np.max(connect)

    c_obj = ConnectObj('ConnectObj1', xyz, connect,color_by='strength',
                    dynamic=(.1, 1.), cmap='gnuplot', vmin=umin + .2,
                    vmax=umax - .1,line_width=0.1,
                    clim=(umin, umax), antialias=True)


    r_obj = RoiObj('aal')
    idx_rh = r_obj.where_is('Hippocampus (R)')
    idx_lh = r_obj.where_is('Hippocampus (L)')
    r_obj.select_roi(select=[idx_rh, idx_lh], unique_color=False, smooth=7, translucent=True)

    vb = Engram(source_obj=s_obj,roi_obj=r_obj,connect_obj=c_obj,metadata=metadata,\
                rotation=0.1,carousel_metadata=intersection_matrices,\
                    carousel_display_method='text')
    vb.engram_control(template='B1',alpha=.02)
    vb.engram_control(visible=False)
    vb.connect_control(c_obj.name,visible=False)
    vb.sources_control(s_obj.name,visible=True)
    vb.rotate(custom=(180-45.0, 0.0))
    vb.show()


def position_slicer(intersection_matrices, method=[],ignore_streams=False):

    SPACING = 1 # In MNI coordinates
    RESCALING = 100
    X = []
    Y = []
    Z = []
    # R = []
    # G = []
    # B = []
    # W = []

    indices = np.copy(intersection_matrices['indices'])
    positions = np.copy(intersection_matrices['positions'])
    sources = np.copy(intersection_matrices['sources'])

    dims = np.arange(np.shape(indices)[1])
    if method is []:
        dim_to_remove = dims
    else:
        dim_to_remove = np.where(dims != method)[0]
    new_inds = indices
    new_inds[:,dim_to_remove] = 0
    groups, streams_in_groups,n_streams_in_group = np.unique(new_inds,axis=0,return_inverse=True,return_counts=True)
    group_pos = np.empty((np.shape(groups)[0],np.shape(positions)[1]))
    for ii,group in enumerate(groups):
        group_indices = np.where(streams_in_groups==ii)[0]
        group_pos[ii] = np.mean(positions[group_indices],axis=0)

    for group, group_properties in enumerate(groups):
        streams = np.squeeze(np.argwhere(np.all((new_inds-group_properties)==0, axis=1)))
        n_sources_in_group = sources[streams].size

        if ignore_streams:
            n_sources_in_streams = np.asarray([np.arange(n_sources_in_group)])
        else:
            n_sources_in_streams = sources[streams]

        for stream, source_inds in enumerate(n_sources_in_streams):
            source_inds -= source_inds[0]
            side = math.ceil((len(source_inds)-1)**(1./2.)) # (1./3.))
            side_1 = SPACING * ((source_inds//(side)) - ((side-1)/2))
            flatten = np.tile(0.,len(source_inds))
            side_2 = SPACING * ((source_inds%side) - ((side-1)/2)) #SPACING * ((((source/n_sources_in_group)%(side**2))//side) - (side-1)/2)
            
            if ignore_streams:
                X = np.append(X,group_pos[group][0] + side_1)
                Y = np.append(Y,group_pos[group][1] + flatten)
                Z = np.append(Z,group_pos[group][2] + side_2)
            else:
                X = np.append(X,group_pos[group][0] + flatten + (SPACING*stream))
                Y = np.append(Y,group_pos[group][1] + side_1)
                Z = np.append(Z,group_pos[group][2] + side_2)
                
            # R = np.append(R,np.tile(group/(len(groups)-1),len(source_inds)))
            # G = np.append(G,np.tile(group/(len(groups)-1),len(source_inds)))
            # B = np.append(B,np.tile(group/(len(groups)-1),len(source_inds)))
            # W = np.append(W,np.tile(1.,len(source_inds))) # Opacity 

    n_sources = sources.size

    # Recenter (to canvas) unless all distinctions have been made
    if (np.asarray([0,1,2]) != np.asarray([0,1,2])).any():
        if len(np.unique(X)) > 1:
            X = ((X - np.min(X))/(max(X) - min(X))) - .5
        else:
            X = 0
        
        X = RESCALING * X

        if len(np.unique(Y)) > 1:
            Y = ((Y - min(Y))/(max(Y) - min(Y))) - .5
        else:
            Y = 0
        
        Y = RESCALING * Y

        if len(np.unique(Z)) > 1:
            Z = ((Z - min(Z))/(max(Z) - min(Z))) - .5
        else:
            Z = 0

        Z = RESCALING * Z

    xyz = np.zeros((n_sources,3)).astype('float32')
    # data = np.zeros(n_sources, [('a_color', np.float32, 4),
    #                     ('a_rotation', np.float32, 4)])

    xyz[:, 0] = X
    xyz[:, 1] = Y
    xyz[:, 2] = Z

    # data['a_color'][:, 0] = R
    # data['a_color'][:, 1] = G
    # data['a_color'][:, 2] = B
    # data['a_color'][:, 3] = W

    return xyz # , data

# ____________________________ VISPY EXAMPLES ____________________________
# See http://vispy.org/gallery.html for similar work.



def shadertoy():

    import sys
    from datetime import datetime, time
    import numpy as np
    from vispy import gloo
    from vispy import app


    vertex = """
    #version 120

    attribute vec2 position;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """

    fragment = """
    #version 120

    uniform vec3      iResolution;           // viewport resolution (in pixels)
    uniform float     iGlobalTime;           // shader playback time (in seconds)
    uniform vec4      iMouse;                // mouse pixel coords
    uniform vec4      iDate;                 // (year, month, day, time in seconds)
    uniform float     iSampleRate;           // sound sample rate (i.e., 44100)
    uniform sampler2D iChannel0;             // input channel. XX = 2D/Cube
    uniform sampler2D iChannel1;             // input channel. XX = 2D/Cube
    uniform sampler2D iChannel2;             // input channel. XX = 2D/Cube
    uniform sampler2D iChannel3;             // input channel. XX = 2D/Cube
    uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
    uniform float     iChannelTime[4];       // channel playback time (in sec)

    %s
    """


    def get_idate():
        now = datetime.now()
        utcnow = datetime.utcnow()
        midnight_utc = datetime.combine(utcnow.date(), time(0))
        delta = utcnow - midnight_utc
        return (now.year, now.month, now.day, delta.seconds)


    def noise(resolution=64, nchannels=1):
        # Random texture.
        return np.random.randint(low=0, high=256,
                                size=(resolution, resolution, nchannels)
                                ).astype(np.uint8)


    class Canvas(app.Canvas):

        def __init__(self, shadertoy=None):
            app.Canvas.__init__(self, keys='interactive')
            if shadertoy is None:
                shadertoy = """
                void main(void)
                {
                    vec2 uv = gl_FragCoord.xy / iResolution.xy;
                    gl_FragColor = vec4(uv,0.5+0.5*sin(iGlobalTime),1.0);
                }"""
            self.program = gloo.Program(vertex, fragment % shadertoy)

            self.program["position"] = [(-1, -1), (-1, 1), (1, 1),
                                        (-1, -1), (1, 1), (1, -1)]
            self.program['iMouse'] = 0, 0, 0, 0

            self.program['iSampleRate'] = 44100.
            for i in range(4):
                self.program['iChannelTime[%d]' % i] = 0.

            self.activate_zoom()

            self._timer = app.Timer('auto', connect=self.on_timer, start=True)

            self.show()

        def set_channel_input(self, img, i=0):
            tex = gloo.Texture2D(img)
            tex.interpolation = 'linear'
            tex.wrapping = 'repeat'
            self.program['iChannel%d' % i] = tex
            self.program['iChannelResolution[%d]' % i] = img.shape

        def on_draw(self, event):
            self.program.draw()

        def on_mouse_click(self, event):
            # BUG: DOES NOT WORK YET, NO CLICK EVENT IN VISPY FOR NOW...
            imouse = event.pos + event.pos
            self.program['iMouse'] = imouse

        def on_mouse_move(self, event):
            if event.is_dragging:
                x, y = event.pos
                px, py = event.press_event.pos
                imouse = (x, self.size[1] - y, px, self.size[1] - py)
                self.program['iMouse'] = imouse

        def on_timer(self, event):
            self.program['iGlobalTime'] = event.elapsed
            self.program['iDate'] = get_idate()  # used in some shadertoy exs
            self.update()

        def on_resize(self, event):
            self.activate_zoom()

        def activate_zoom(self):
            gloo.set_viewport(0, 0, *self.physical_size)
            self.program['iResolution'] = (self.physical_size[0],
                                        self.physical_size[1], 0.)

    # -------------------------------------------------------------------------
    # COPY-PASTE SHADERTOY CODE BELOW
    # -------------------------------------------------------------------------
    SHADERTOY = """
    // From: https://www.shadertoy.com/view/MdX3Rr

    // "Vortex Street" by dr2 - 2015
    // License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

    // Motivated by implementation of van Wijk's IBFV by eiffie (lllGDl) and andregc (4llGWl) 

    const vec4 cHashA4 = vec4 (0., 1., 57., 58.);
    const vec3 cHashA3 = vec3 (1., 57., 113.);
    const float cHashM = 43758.54;

    vec4 Hashv4f (float p)
    {
    return fract (sin (p + cHashA4) * cHashM);
    }

    float Noisefv2 (vec2 p)
    {
    vec2 i = floor (p);
    vec2 f = fract (p);
    f = f * f * (3. - 2. * f);
    vec4 t = Hashv4f (dot (i, cHashA3.xy));
    return mix (mix (t.x, t.y, f.x), mix (t.z, t.w, f.x), f.y);
    }

    float Fbm2 (vec2 p)
    {
    float s = 0.;
    float a = 1.;
    for (int i = 0; i < 6; i ++) {
        s += a * Noisefv2 (p);
        a *= 0.5;
        p *= 2.;
    }
    return s;
    }

    float tCur;

    vec2 VortF (vec2 q, vec2 c)
    {
    vec2 d = q - c;
    return 0.25 * vec2 (d.y, - d.x) / (dot (d, d) + 0.05);
    }

    vec2 FlowField (vec2 q)
    {
    vec2 vr, c;
    float dir = 1.;
    c = vec2 (mod (tCur, 10.) - 20., 0.6 * dir);
    vr = vec2 (0.);
    for (int k = 0; k < 30; k ++) {
        vr += dir * VortF (4. * q, c);
        c = vec2 (c.x + 1., - c.y);
        dir = - dir;
    }
    return vr;
    }

    void main(void)
    {
    vec2 uv = gl_FragCoord.xy / iResolution.xy - 0.5;
    uv.x *= iResolution.x / iResolution.y;
    tCur = iGlobalTime;
    vec2 p = uv;
    for (int i = 0; i < 10; i ++) p -= FlowField (p) * 0.03;
    vec3 col = Fbm2 (5. * p + vec2 (-0.1 * tCur, 0.)) *
        vec3 (0.5, 0.5, 1.);
    gl_FragColor = vec4 (col, 1.);
    }

    """
    # -------------------------------------------------------------------------

    canvas = Canvas(SHADERTOY)
    # Input data.
    canvas.set_channel_input(noise(resolution=256, nchannels=1), i=0)
    canvas.show()
    canvas.app.run()

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














