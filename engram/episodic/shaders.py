import glumpy

# Note : All files currently copied from Glumpy examples (by Nicolas P. Rougier)
# See https://glumpy.readthedocs.io/en/latest/examples.html for similar work.



def select(name):
    selection = {
        "galaxy": galaxy,
        "fireworks": fireworks,
        "spherical-harmonics": sphericalharmonics,
        "frequency": frequency,
        "heatmap": heatmap,
        "matplotlib": likematplotlib,
        "oscilloscope" : oscilloscope,
        "realtimesignals" : realtimesignals,
        "fluid": fluid,
        "brain":brain
    }
    # Get the function from switcher dictionary
    func = selection.get(name, lambda: "Invalid event parser")
    # Execute the function
    return func()

def brain():
    # -----------------------------------------------------------------------------
    # Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
    # Distributed under the (new) BSD License.
    # -----------------------------------------------------------------------------
    import numpy as np
    from glumpy import app, gl, gloo, data, log
    from glumpy.transforms import Trackball, Position


    vertex = """
    uniform mat4 m_model;
    uniform mat4 m_view;
    uniform mat4 m_normal;
    attribute vec3 position;
    attribute vec3 normal;
    varying vec3 v_normal;
    varying vec3 v_position;

    void main()
    {
        gl_Position = <transform>;
        vec4 P = m_view * m_model* vec4(position, 1.0);
        v_position = P.xyz / P.w;
        v_normal = vec3(m_normal * vec4(normal,0.0));
    }
    """

    fragment = """
    varying vec3 v_normal;
    varying vec3 v_position;

    const vec3 light_position = vec3(1.0,1.0,1.0);
    const vec3 ambient_color = vec3(0.1, 0.0, 0.0);
    const vec3 diffuse_color = vec3(0.75, 0.125, 0.125);
    const vec3 specular_color = vec3(1.0, 1.0, 1.0);
    const float shininess = 128.0;
    const float gamma = 2.2;

    void main()
    {
        vec3 normal= normalize(v_normal);
        vec3 light_direction = normalize(light_position - v_position);
        float lambertian = max(dot(light_direction,normal), 0.0);
        float specular = 0.0;
        if (lambertian > 0.0)
        {
            vec3 view_direction = normalize(-v_position);
            vec3 half_direction = normalize(light_direction + view_direction);
            float specular_angle = max(dot(half_direction, normal), 0.0);
            specular = pow(specular_angle, shininess);
        }
        vec3 color_linear = ambient_color +
                            lambertian * diffuse_color +
                            specular * specular_color;
        vec3 color_gamma = pow(color_linear, vec3(1.0/gamma));
        gl_FragColor = vec4(color_gamma, 1.0);
    }
    """

    log.info("Loading brain mesh")
    vertices,indices = data.get("brain.obj")
    brain = gloo.Program(vertex, fragment)
    brain.bind(vertices)
    trackball = Trackball(Position("position"))
    brain['transform'] = trackball
    trackball.theta, trackball.phi, trackball.zoom = 80, -135, 15

    window = app.Window(width=1024, height=768)

    def update():
        model = brain['transform']['model'].reshape(4,4)
        view  = brain['transform']['view'].reshape(4,4)
        brain['m_view']  = view
        brain['m_model'] = model
        brain['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)
        
    @window.event
    def on_draw(dt):
        window.clear()
        brain.draw(gl.GL_TRIANGLES)

    @window.event
    def on_mouse_drag(x, y, dx, dy, button):
        update()
        
    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)
        update()

    window.attach(brain['transform'])
    app.run()


def fluid():
    # -*- coding: utf-8 -*-
    # Distributed under the (new) BSD License.
    # -----------------------------------------------------------------------------
    """
    Fluid with obstacles.
    https://www.shadertoy.com/view/lllGDl
    """

    from glumpy import app, gl, gloo

    vertex = '''
    attribute vec2 a_position;
    varying vec2 fragCoord;

    void main()
    {
        gl_Position = vec4(a_position, 0.0, 1.0);
        fragCoord = a_position;
    }
    '''

    fragment = '''
    varying vec2 fragCoord;
    uniform float iGlobalTime;
    uniform vec2 iMouse;

    #define STEP_COUNT 10

    //these are the field movers
    vec2 swirl(vec2 uv, vec2 center, float strength, float eyeWall) {
        vec2 d = uv - center;
        return vec2(d.y, -d.x)/(dot(d,d)/strength+eyeWall);
    }
    vec2 spray(vec2 uv, vec2 center, vec2 dir, float strength, float eyeWall){
        vec2 d = uv - center;
        return vec2(d.x, d.y)/(dot(d,d)/strength+eyeWall)*dot(d,dir);
    }
    vec2 drain(vec2 uv, vec2 center, float strength, float eyeWall){
        vec2 d = uv - center;
        return -vec2(d.x, d.y)/(dot(d,d)/strength+eyeWall);
    }
    //DE is used to define barriors
    float Tube(vec2 pa, vec2 ba){
        return length(pa-ba*clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0));
    }
    float DE(vec2 p){
        p+=vec2(0.5);
        return min(length(p),Tube(p-vec2(1.0),vec2(0.4,0.2)));
    }
    vec2 ReflectOffSurf(vec2 p, vec2 r){
        float d=max(DE(p),0.001);
        vec2 v=vec2(d,0.0);
        vec2 N=normalize(vec2(DE(p+v.xy)-DE(p-v.xy),DE(p+v.yx)-DE(p-v.yx)));
        d=clamp(sqrt(d)*1.1,0.0,1.0);
        r=mix(reflect(r,N)*clamp(0.5-0.5*dot(r,N),0.0,1.0),r*d,d);
        return r;
    }
    vec2 field(vec2 uv) {
        vec2 mouse = (iMouse.x == 0. && iMouse.y==0.) ? vec2(-0.15,-0.1) : iMouse.xy;
        mouse*=3.0;
        vec2 p=
            swirl(uv, mouse,1.5,0.25)
            +spray(uv,-mouse,vec2(-1.0,0.5),0.5,0.1)
            +drain(uv,mouse,0.5,0.75)
        ;
        p=ReflectOffSurf(uv,p);
        return p;
    }

    //just basic clouds from perlin noise
    float rand(vec2 co){return fract(sin(dot(co,vec2(12.9898,78.233)))*43758.5453);}
    float noyz(vec2 co){
        vec2 d=smoothstep(0.0,1.0,fract(co));
        co=floor(co);
        const vec2 v=vec2(1.0,0.0);
        return mix(mix(rand(co),rand(co+v.xy),d.x),
            mix(rand(co+v.yx),rand(co+v.xx),d.x),d.y);
    }
    float clouds( in vec2 q, in float tm )
    {
        float f=0.0,a=0.6;
        for(int i=0;i<5;i++){
                f+= a*noyz( q+tm );
            q = q*2.03;
            a = a*0.5;
        }
        return f;
    }

    float getPattern(vec2 uv) {
        //this can be any pattern but moving patterns work best
        float w=clouds(uv*5.0, iGlobalTime*0.5);
        return w;
    }

    vec2 calcNext(vec2 uv, float t) {
        t /= float(STEP_COUNT);
        for(int i = 0; i < STEP_COUNT; ++i) {
            uv -= field(uv)*t;
        }
        return uv;
    }

    vec3 heatmap(float h){
        return mix(vec3(0.1,0.2,0.4),vec3(2.0,1.5-h,0.5)/(1.0+h),h);
    }

    vec3 Fluid(vec2 uv, float t) {
        float t1 = t*0.5;
        float t2 = t1 + 0.5;
        vec2 uv1 = calcNext(uv, t1);
        vec2 uv2 = calcNext(uv, t2);
        float c1 = getPattern(uv1);
        float c2 = getPattern(uv2);
        float c=mix(c2,c1,t);
        float f=1.5-0.5*abs(t-0.5);
        c=pow(c,f)*f;//correcting the contrast/brightness when sliding
        float h=mix(length(uv-uv2),length(uv-uv1),t);
        return 2.0*c*heatmap(clamp(h*0.5,0.0,1.0));//blue means slow, red = fast
    }

    void main()
    {
        vec2 uv = fragCoord;
        uv*=3.0;
        float t = fract(iGlobalTime);
        vec3 c = Fluid(uv,t);//draws fluid
        float d=DE(uv);//get distance to objects
        c=mix(vec3(1.0-10.0*d*d),c,smoothstep(0.2,0.25,d));//mix in objects
        gl_FragColor = vec4(c,1.0);
    }

    '''

    program = gloo.Program(vertex, fragment, count=4)
    program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    program['iGlobalTime'] = 0.0

    window = app.Window(width=900, height=900)


    @window.event
    def on_draw(dt):
        window.clear()
        _, _, w, h = gl.glGetIntegerv(gl.GL_VIEWPORT)
        program.draw(gl.GL_TRIANGLE_STRIP)
        program['iGlobalTime'] += dt


    @window.event
    def on_mouse_press(x, y, button):
        _, _, w, h = gl.glGetIntegerv(gl.GL_VIEWPORT)
        program["iMouse"] = 2 * x / w - 1, 1 - 2 * y / h


    @window.timer(1 / 5.0)
    def timer(dt):
        window.set_title("time:{:5.1f}\tfps:{:3.1f}".format(program['iGlobalTime'][0], window.fps).encode())


    app.run()


def realtimesignals():
    # -----------------------------------------------------------------------------
    # Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
    # Distributed under the (new) BSD License.
    # -----------------------------------------------------------------------------
    # Realtime signals example
    #
    # Implementation uses a ring buffer such that only new values are uploaded in
    # GPU memory. This requires the corresponding numpy array to have a
    # Fortran-like layout.
    # -----------------------------------------------------------------------------
    import numpy as np
    from glumpy import app, gloo, gl

    vertex = """
    uniform int index, size, count;
    attribute float x_index, y_index, y_value;
    varying float do_discard;
    void main (void)
    {
        float x = 2*(mod(x_index - index, size) / (size)) - 1.0;
        if ((x >= +1.0) || (x <= -1.0)) do_discard = 1;
        else                            do_discard = 0;
        float y = (2*((y_index+.5)/(count))-1) + y_value;
        gl_Position = vec4(x, y, 0, 1);
    }
    """

    fragment = """
    varying float do_discard;
    void main(void)
    {
        if (do_discard > 0) discard;
        gl_FragColor = vec4(0,0,0,1);
    }
    """

    window = app.Window(width=1500, height=1000, color=(1,1,1,1))

    @window.event
    def on_draw(dt):
        global size, count
        window.clear()
        program.draw(gl.GL_LINES, I)
        index = int(program["index"])
        y = program["y_value"].reshape(size,count)

        yscale = 1.0/count
        y[index] = yscale * np.random.uniform(-1,+1,count)
        program["index"] = (index + 1) % size

    global size,count
    count, size = 64, 1000
    program = gloo.Program(vertex, fragment, count=size*count)
    program["size"] = size
    program["count"] = count
    program["x_index"] = np.repeat(np.arange(size),count)
    program["y_index"] = np.tile(np.arange(count),size)
    program["y_value"] = 0

    # Compute indices
    I = np.arange(count * size, dtype=np.uint32).reshape(size, -1).T
    I = np.roll(np.repeat(I, 2, axis=1), -1, axis=1)
    I = I.view(gloo.IndexBuffer)

    app.run()


def oscilloscope():
    # -----------------------------------------------------------------------------
    # Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
    # Distributed under the (new) BSD License.
    # -----------------------------------------------------------------------------
    """ This example show a very simple oscilloscope. """

    import numpy as np
    from glumpy import app, gl, glm, gloo


    vertex = """
    attribute float x, y, intensity;

    varying float v_intensity;
    void main (void)
    {
        v_intensity = intensity;
        gl_Position = vec4(x, y, 0.0, 1.0);
    }
    """

    fragment = """
    varying float v_intensity;
    void main()
    {
        gl_FragColor = vec4(0,v_intensity,0,1);
    }
    """

    window = app.Window(width=1024, height=512)

    @window.event
    def on_draw(dt):
        global index
        window.clear()
        oscilloscope.draw(gl.GL_LINE_STRIP)
        index = (index-1) % len(oscilloscope)
        oscilloscope['intensity'] -= 1.0/len(oscilloscope)
        oscilloscope['y'][index] = np.random.uniform(-0.25, +0.25)
        oscilloscope['intensity'][index] = 1.0

    global index
    index = 0
    oscilloscope = gloo.Program(vertex, fragment, count=150)
    oscilloscope['x'] = np.linspace(-1,1,len(oscilloscope))

    app.run()

def likematplotlib():
    # -----------------------------------------------------------------------------
    # Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
    # Distributed under the (new) BSD License.
    # -----------------------------------------------------------------------------
    import numpy as np
    import glumpy.api.matplotlib as gmpl

    # Create a new figure
    figure = gmpl.Figure((24,12))

    # Create a subplot on left, using trackball interface (3d)
    left = figure.add_axes( [0.010, 0.01, 0.485, 0.98],
                            xscale = gmpl.LinearScale(clamp=True),
                            yscale = gmpl.LinearScale(clamp=True),
                            zscale = gmpl.LinearScale(clamp=True),
                            interface = gmpl.Trackball(name="trackball"),
                            facecolor=(1,0,0,0.25), aspect=1 )

    # Create a subplot on right, using panzoom interface (2d)
    right = figure.add_axes( [0.505, 0.01, 0.485, 0.98],
                            xscale = gmpl.LinearScale(domain=[-2.0,+2.0], range=[0.25,1.00]),
                            yscale = gmpl.LinearScale(domain=[-2.0,+2.0], range=[0,2*np.pi]),
                            projection = gmpl.PolarProjection(),
                            interface = gmpl.Trackball(name="trackball"),
                            facecolor=(0,0,1,0.25), aspect=1 )

    # Create a new collection of points
    collection = gmpl.PointCollection("agg")

    # Add a view of the collection on the left subplot
    left.add_drawable(collection)

    # Add a view of the collection on the right subplot
    right.add_drawable(collection)

    # Add some points
    collection.append(np.random.uniform(-2,2,(10000,3)))

    # Show figure
    figure.show()


def heatmap():
    # -----------------------------------------------------------------------------
    # Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
    # Distributed under the (new) BSD License.
    # -----------------------------------------------------------------------------
    import numpy as np
    from glumpy import app, gl, gloo, library
    from glumpy.transforms import PanZoom, Position

    vertex = """
        uniform vec4 viewport;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        varying vec2 v_pixcoord;
        varying vec2 v_quadsize;
        void main()
        {
            gl_Position = <transform>;
            v_texcoord = texcoord;
            v_quadsize = viewport.zw * <transform.panzoom_scale>;
            v_pixcoord = texcoord * v_quadsize;
        }
    """

    fragment = """
    #include "markers/markers.glsl"
    #include "antialias/antialias.glsl"

    uniform sampler2D data;
    uniform vec2 data_shape;
    varying vec2 v_texcoord;
    varying vec2 v_quadsize;
    varying vec2 v_pixcoord;

    void main()
    {
        float rows = data_shape.x;
        float cols = data_shape.y;
        float v = texture2D(data, v_texcoord).r;

        vec2 size = v_quadsize / vec2(cols,rows);
        vec2 center = (floor(v_pixcoord/size) + vec2(0.5,0.5)) * size;
        float d = marker_square(v_pixcoord - center, .9*size.x);
        gl_FragColor = filled(d, 1.0, 1.0, vec4(v,v,v,1));
    }
    """

    window = app.Window(width=1024, height=1024, color=(1,1,1,1))

    @window.event
    def on_draw(dt):
        window.clear()
        program.draw(gl.GL_TRIANGLE_STRIP)

    @window.event
    def on_key_press(key, modifiers):
        if key == app.window.key.SPACE:
            transform.reset()

    @window.event
    def on_resize(width, height):
        program['viewport'] = 0, 0, width, height

    program = gloo.Program(vertex, fragment, count=4)

    n = 64
    program['position'] = [(-1,-1), (-1,1), (1,-1), (1,1)]
    program['texcoord'] = [( 0, 1), ( 0, 0), ( 1, 1), ( 1, 0)]
    program['data'] = np.random.uniform(0,1,(n,n))
    program['data_shape'] = program['data'].shape[:2]
    transform = PanZoom(Position("position"),aspect=1)

    program['transform'] = transform
    window.attach(transform)
    app.run()

def frequency():
    # -----------------------------------------------------------------------------
    # Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
    # Distributed under the (new) BSD License.
    # -----------------------------------------------------------------------------
    # High frequency (below pixel resolution) function plot
    #
    #  -> http://blog.hvidtfeldts.net/index.php/2011/07/plotting-high-frequency-functions-using-a-gpu/
    #  -> https://www.shadertoy.com/view/4sB3zz
    # -----------------------------------------------------------------------------
    import numpy as np
    from glumpy import app, gl, gloo

    vertex = """
    attribute vec2 position;
    void main (void)
    {
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """

    fragment = """
    uniform vec2 iResolution;
    uniform float iGlobalTime;

    // --- Your function here ---
    float function( float x )
    {
        float d = 3.0 - 2.0*(1.0+cos(iGlobalTime/5.0))/2.0;
        return sin(pow(x,d))*sin(x);
    }
    // --- Your function here ---


    float sample(vec2 uv)
    {
        const int samples = 128;
        const float fsamples = float(samples);
        vec2 maxdist = vec2(0.5,1.0)/40.0;
        vec2 halfmaxdist = vec2(0.5) * maxdist;

        float stepsize = maxdist.x / fsamples;
        float initial_offset_x = -0.5 * fsamples * stepsize;
        uv.x += initial_offset_x;
        float hit = 0.0;
        for( int i=0; i<samples; ++i )
        {
            float x = uv.x + stepsize * float(i);
            float y = uv.y;
            float fx = function(x);
            float dist = abs(y-fx);
            hit += step(dist, halfmaxdist.y);
        }
        const float arbitraryFactor = 4.5;
        const float arbitraryExp = 0.95;
        return arbitraryFactor * pow( hit / fsamples, arbitraryExp );
    }

    void main(void)
    {
        vec2 uv = gl_FragCoord.xy / iResolution.xy;
        float ymin = -2.0;
        float ymax = +2.0;
        float xmin = 0.0;
        float xmax = xmin + (ymax-ymin)* iResolution.x / iResolution.y;

        vec2 xy = vec2(xmin,ymin) + uv*vec2(xmax-xmin, ymax-ymin);
        gl_FragColor = vec4(0,0,0, sample(xy));
    }
    """

    window = app.Window(width=3*512, height=512, color=(1,1,1,1))
    pause = False

    @window.event
    def on_draw(dt):
        window.clear()
        program.draw(gl.GL_TRIANGLE_STRIP)
        if not pause:
            program["iGlobalTime"] += dt

    @window.event
    def on_key_press(key, modifiers):
        global pause
        if key == ord(' '):
            pause = not pause

    @window.event
    def on_resize(width, height):
        program["iResolution"] = width, height

    program = gloo.Program(vertex, fragment, count=4)
    program['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
    program["iGlobalTime"] = 0
    app.run()




def galaxy():
    import numpy as np
    from glumpy import app, gloo, gl, glm, data


    vertex = """
    #version 120
    uniform mat4  u_model;
    uniform mat4  u_view;
    uniform mat4  u_projection;
    uniform sampler1D u_colormap;

    attribute float a_size;
    attribute float a_type;
    attribute vec2  a_position;
    attribute float a_temperature;
    attribute float a_brightness;

    varying vec3 v_color;
    void main (void)
    {
        gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);
        if (a_size > 2.0)
        {
            gl_PointSize = a_size;
        } else {
            gl_PointSize = 0.0;
        }
        v_color = texture1D(u_colormap, a_temperature).rgb * a_brightness;
        if (a_type == 2)
            v_color *= vec3(2,1,1);
        else if (a_type == 3)
            v_color = vec3(.9);
    }
    """

    fragment = """
    #version 120
    uniform sampler2D u_texture;
    varying vec3 v_color;
    void main()
    {
        gl_FragColor = vec4(texture2D(u_texture, gl_PointCoord).r*v_color, 1.0);
    }
    """

    window = app.Window(width=800, height=800)

    @window.event
    def on_init():
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)

    @window.event
    def on_draw(dt):
        window.clear()
        galaxy.update(100000) # in years !
        program['a_size'] = galaxy['size'] * max(window.width/800.0, window.height/800.0)
        program['a_position'] = galaxy['position'] / 13000.0
        program.draw(gl.GL_POINTS)

    @window.event
    def on_resize(width,height):
        gl.glViewport(0, 0, width, height)
        projection = glm.perspective(45.0, width/float(height), 1.0, 1000.0)
        program['u_projection'] = projection

    galaxy = Galaxy(35000)
    galaxy.reset(13000, 4000, 0.0004, 0.90, 0.90, 0.5, 200, 300)
    t0, t1 = 1000.0, 10000.0
    n = 256
    dt =  (t1-t0)/n
    colors = np.zeros((n,3), dtype=np.float32)
    for i in range(n):
        temperature = t0 + i*dt
        x,y,z = spectrum_to_xyz(bb_spectrum, temperature)
        r,g,b = xyz_to_rgb(SMPTEsystem, x, y, z)
        r = min((max(r,0),1))
        g = min((max(g,0),1))
        b = min((max(b,0),1))
        colors[i] = norm_rgb(r, g, b)


    program = gloo.Program(vertex, fragment, count=len(galaxy))

    view = np.eye(4, dtype=np.float32)
    model = np.eye(4, dtype=np.float32)
    projection = np.eye(4, dtype=np.float32)
    glm.translate(view, 0, 0, -5)
    program['u_model'] = model
    program['u_view'] = view
    program['u_colormap'] = colors
    program['u_texture'] = data.get("particle.png")
    program['u_texture'].interpolation = gl.GL_LINEAR

    program['a_temperature'] = (galaxy['temperature'] - t0) / (t1-t0)
    program['a_brightness'] = galaxy['brightness']
    program['a_size'] = galaxy['size']
    program['a_type'] = galaxy['type']

    gl.glClearColor(0.0, 0.0, 0.03, 1.0)
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)

    app.run(framerate=60)


def fireworks():
    import numpy as np
    from glumpy import app, gl, gloo

    vertex = """
    #version 120
    uniform float time;
    uniform vec2 center;
    attribute vec2 start, end;
    attribute float lifetime;
    varying float v_lifetime;
    void main () {
        gl_Position = vec4(start + (time * end) + center, 0.0, 1.0);
        gl_Position.y -= 1.0 * time * time;
        v_lifetime = clamp(1.0 - (time / lifetime), 0.0, 1.0);
        gl_PointSize = (v_lifetime * v_lifetime) * 30.0;
    }
    """

    fragment = """
    #version 120
    const float SQRT_2 = 1.4142135623730951;
    uniform vec4 color;
    varying float v_lifetime;
    void main()
    {
        gl_FragColor = color * (SQRT_2/2.0 - length(gl_PointCoord.xy - 0.5));
        gl_FragColor.a *= v_lifetime;
    }
    """

    n = 2500
    window = app.Window(512,512)
    program = gloo.Program(vertex, fragment, count=n)

    def explosion():
        program['center'] = np.random.uniform(-0.5,+0.5)
        program['color'] = np.random.uniform(0.1,0.9,4)
        program['color'][3] = 1.0 / n ** 0.05
        program['lifetime'] = np.random.normal(4.0, 0.5, n)
        program['start'] = np.random.normal(0.0, 0.2, (n,2))
        program['end'] = np.random.normal(0.0, 1.2, (n,2))
        program['time'] = 0

    @window.event
    def on_init():
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)

    @window.event
    def on_draw(dt):
        window.clear()
        program.draw(gl.GL_POINTS)
        program['time'] += dt
        if program['time'] > 1.75:
            explosion()

    explosion()
    app.run(framerate=60)

def sphericalharmonics():
    # -----------------------------------------------------------------------------
    # Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
    # Distributed under the (new) BSD License.
    # -----------------------------------------------------------------------------
    import numpy as np
    from scipy.special import sph_harm
    from glumpy import app, gl, gloo, transforms


    def sphere(radius=1.0, slices=256, stacks=256):
        vtype = [('theta', np.float32, 1),
                ('phi', np.float32, 1)]
        slices += 1
        stacks += 1
        n = slices*stacks
        vertices = np.zeros(n, dtype=vtype)
        vertices["theta"] = np.repeat(np.linspace(0, np.pi, stacks, endpoint=True), slices)
        vertices["phi"] = np.tile(np.linspace(0, 2 * np.pi, slices, endpoint=True), stacks)
        indices = []
        for i in range(stacks-1):
            for j in range(slices-1):
                indices.append(i*(slices) + j        )
                indices.append(i*(slices) + j+1      )
                indices.append(i*(slices) + j+slices+1)
                indices.append(i*(slices) + j+slices  )
                indices.append(i*(slices) + j+slices+1)
                indices.append(i*(slices) + j        )
        indices = np.array(indices, dtype=np.uint32)
        return vertices.view(gloo.VertexBuffer), indices.view(gloo.IndexBuffer)


    vertex = """
    float harmonic(float theta, float phi, float m[8])
    {
        return pow(sin(m[0]*phi),m[1]) + pow(sin(m[4]*theta),m[5]) +
            pow(cos(m[2]*phi),m[3]) + pow(cos(m[6]*theta),m[7]);
    }

    uniform float time;
    uniform float m1[8];
    uniform float m2[8];

    attribute float phi;
    attribute float theta;
    varying float v_theta;
    varying float v_phi;
    varying vec3 v_position;

    void main()
    {
        float radius, x, y, z;

        v_phi = phi;
        v_theta = theta;

        radius = 1.0 + 0.15*(harmonic(theta,phi,m1));
        x = sin(theta) * sin(phi) * radius;
        y = sin(theta) * cos(phi) * radius;
        z = cos(theta) * radius;
        vec3 position1 = vec3(x,y,z);

        radius = 1.0 + 0.15*(harmonic(theta,phi,m2));
        x = sin(theta) * sin(phi) * radius;
        y = sin(theta) * cos(phi) * radius;
        z = cos(theta) * radius;
        vec3 position2 = vec3(x,y,z);

        float t = (1.0+cos(time))/2.0;
        vec4 position = vec4(mix(position1, position2,t), 1.0);
        v_position = position.xyz;

        gl_Position = <transform(position)>;
    }
    """

    fragment = """
    float segment(float edge0, float edge1, float x)
    {
        return step(edge0,x) * (1.0-step(edge1,x));
    }
    vec3 ice(float t)
    {
        return vec3(t, t, 1.0);
    }
    vec3 fire(float t) {
        return mix(mix(vec3(1,1,1),vec3(1,1,0),t),mix(vec3(1,1,0),vec3(1,0,0),t*t),t);
    }
    vec3 ice_and_fire(float t)
    {
        return segment(0.0,0.5,t)*ice(2.0*(t-0.0)) + segment(0.5,1.0,t)*fire(2.0*(t-0.5));
    }

    float harmonic(float theta, float phi, float m[8])
    {
        return pow(sin(m[0]*phi),m[1]) + pow(sin(m[4]*theta),m[5]) +
            pow(cos(m[2]*phi),m[3]) + pow(cos(m[6]*theta),m[7]);
    }

    uniform float time;
    uniform float m1[8];
    uniform float m2[8];

    varying vec3 v_position;
    varying vec3 v_normal;
    varying float v_phi;
    varying float v_theta;
    void main()
    {
        float t1 = (harmonic(v_theta, v_phi, m1)) / 4.0;
        float t2 = (harmonic(v_theta, v_phi, m2)) / 4.0;
        float t = (1.0+cos(time))/2.0;
        t = mix(t1,t2,t);

        vec4 bg_color = vec4(ice_and_fire(clamp(t,0,1)),1.0);
        vec4 fg_color = vec4(0,0,0,1);

        // Trace contour
        float value = length(v_position);
        float levels = 16.0;
        float antialias = 1.0;
        float linewidth = 1.0 + antialias;
        float v  = levels*value - 0.5;
        float dv = linewidth/2.0 * fwidth(v);
        float f = abs(fract(v) - 0.5);
        float d = smoothstep(-dv,+dv,f);
        t = linewidth/2.0 - antialias;

        d = abs(d)*linewidth/2.0 - t;
        if( d < 0.0 ) {
            gl_FragColor = bg_color;
        } else  {
            d /= antialias;
            gl_FragColor = mix(fg_color,bg_color,d);
        }


    }
    """

    window = app.Window(width=1024, height=1024, color=(.3,.3,.3,1))

    @window.event
    def on_draw(dt):
        global time
        time += dt

        window.clear()
        program["time"] = time
        program.draw(gl.GL_TRIANGLES, faces)

        # trackball.phi = trackball.phi + 0.13
        # trackball.theta = trackball.theta + 0.11

        if (abs(time - np.pi)) < dt:
            values = np.random.randint(0,7,8)
            keys   = ["m2[0]","m2[1]","m2[2]","m2[3]","m2[4]","m2[5]","m2[6]","m2[7]"]
            for key,value in zip(keys, values):
                program[key] = value

        elif (abs(time - 2*np.pi)) < dt:
            values = np.random.randint(0,7,8)
            keys   = ["m1[0]","m1[1]","m1[2]","m1[3]","m1[4]","m1[5]","m1[6]","m1[7]"]
            for key,value in zip(keys, values):
                program[key] = value
            time = 0

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)

    global time
    time = 0
    vertices, faces = sphere()
    program = gloo.Program(vertex, fragment)
    trackball = transforms.Trackball()
    program["transform"] = trackball()
    program.bind(vertices)

    values = np.random.randint(0,7,8)
    keys   = ["m1[0]","m1[1]","m1[2]","m1[3]","m1[4]","m1[5]","m1[6]","m1[7]"]
    for key,value in zip(keys, values):
        program[key] = value

    values = np.random.randint(0,7,8)
    keys   = ["m2[0]","m2[1]","m2[2]","m2[3]","m2[4]","m2[5]","m2[6]","m2[7]"]
    for key,value in zip(keys, values):
        program[key] = value

    trackball.zoom = 30
    window.attach(program["transform"])
    app.run()







'''
----------------------------
Additional Galaxy Elements
----------------------------
'''
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

"""
/* A colour system is defined by the CIE x and y coordinates of
   its three primary illuminants and the x and y coordinates of
   the white point. */
"""
GAMMA_REC709 = 0

NTSCsystem  =  {"name": "NTSC",
    "xRed": 0.67, "yRed": 0.33,
    "xGreen": 0.21, "yGreen": 0.71,
    "xBlue": 0.14, "yBlue": 0.08,
    "xWhite": 0.3101, "yWhite": 0.3163, "gamma": GAMMA_REC709}
EBUsystem  =  {"name": "SUBU (PAL/SECAM)",
    "xRed": 0.64, "yRed": 0.33,
    "xGreen": 0.29, "yGreen": 0.60,
    "xBlue": 0.15, "yBlue": 0.06,
    "xWhite": 0.3127, "yWhite": 0.3291, "gamma": GAMMA_REC709 }
SMPTEsystem  =  {"name": "SMPTE",
    "xRed": 0.63, "yRed": 0.34,
    "xGreen": 0.31, "yGreen": 0.595,
    "xBlue": 0.155, "yBlue": 0.07,
    "xWhite": 0.3127, "yWhite": 0.3291, "gamma": GAMMA_REC709 }
HDTVsystem  =  {"name": "HDTV",
    "xRed": 0.67, "yRed": 0.33,
    "xGreen": 0.21, "yGreen": 0.71,
    "xBlue": 0.15, "yBlue": 0.06,
    "xWhite": 0.3127, "yWhite": 0.3291, "gamma": GAMMA_REC709 }
CIEsystem  =  {"name": "CIE",
    "xRed": 0.7355, "yRed": 0.2645,
    "xGreen": 0.2658, "yGreen": 0.7243,
    "xBlue": 0.1669, "yBlue": 0.0085,
    "xWhite": 0.3333333333, "yWhite": 0.3333333333, "gamma": GAMMA_REC709 }
Rec709system  =  {"name": "CIE REC709",
    "xRed": 0.64, "yRed": 0.33,
    "xGreen": 0.30, "yGreen": 0.60,
    "xBlue": 0.15, "yBlue": 0.06,
    "xWhite": 0.3127, "yWhite": 0.3291, "gamma": GAMMA_REC709 }

def upvp_to_xy(up, vp):
    xc = (9 * up) / ((6 * up) - (16 * vp) + 12)
    yc = (4 * vp) / ((6 * up) - (16 * vp) + 12)
    return(xc, yc)

def xy_toupvp(xc, yc):
    up = (4 * xc) / ((-2 * xc) + (12 * yc) + 3);
    vp = (9 * yc) / ((-2 * xc) + (12 * yc) + 3);
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

    rx = rx / rw;  ry = ry / rw;  rz = rz / rw
    gx = gx / gw;  gy = gy / gw;  gz = gz / gw
    bx = bx / bw;  by = by / bw;  bz = bz / bw

    r = (rx * xc) + (ry * yc) + (rz * zc)
    g = (gx * xc) + (gy * yc) + (gz * zc)
    b = (bx * xc) + (by * yc) + (bz * zc)
    return(r,g,b)


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
    w = -min([0, r, g, b]) # I think?

    # Add just enough white to make r, g, b all positive.
    if w > 0:
        r += w
        g += w
        b += w
    return(r,g,b)

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
    return(r,g,b)

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

def spectrum_to_xyz(spec_intens, temp): #spec_intens is a function
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
       nanometers.  For a wavelength lambda in this range:

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
    compilers. */
    """

    cie_colour_match = [
        [0.0014,0.0000,0.0065], [0.0022,0.0001,0.0105], [0.0042,0.0001,0.0201],
        [0.0076,0.0002,0.0362], [0.0143,0.0004,0.0679], [0.0232,0.0006,0.1102],
        [0.0435,0.0012,0.2074], [0.0776,0.0022,0.3713], [0.1344,0.0040,0.6456],
        [0.2148,0.0073,1.0391], [0.2839,0.0116,1.3856], [0.3285,0.0168,1.6230],
        [0.3483,0.0230,1.7471], [0.3481,0.0298,1.7826], [0.3362,0.0380,1.7721],
        [0.3187,0.0480,1.7441], [0.2908,0.0600,1.6692], [0.2511,0.0739,1.5281],
        [0.1954,0.0910,1.2876], [0.1421,0.1126,1.0419], [0.0956,0.1390,0.8130],
        [0.0580,0.1693,0.6162], [0.0320,0.2080,0.4652], [0.0147,0.2586,0.3533],
        [0.0049,0.3230,0.2720], [0.0024,0.4073,0.2123], [0.0093,0.5030,0.1582],
        [0.0291,0.6082,0.1117], [0.0633,0.7100,0.0782], [0.1096,0.7932,0.0573],
        [0.1655,0.8620,0.0422], [0.2257,0.9149,0.0298], [0.2904,0.9540,0.0203],
        [0.3597,0.9803,0.0134], [0.4334,0.9950,0.0087], [0.5121,1.0000,0.0057],
        [0.5945,0.9950,0.0039], [0.6784,0.9786,0.0027], [0.7621,0.9520,0.0021],
        [0.8425,0.9154,0.0018], [0.9163,0.8700,0.0017], [0.9786,0.8163,0.0014],
        [1.0263,0.7570,0.0011], [1.0567,0.6949,0.0010], [1.0622,0.6310,0.0008],
        [1.0456,0.5668,0.0006], [1.0026,0.5030,0.0003], [0.9384,0.4412,0.0002],
        [0.8544,0.3810,0.0002], [0.7514,0.3210,0.0001], [0.6424,0.2650,0.0000],
        [0.5419,0.2170,0.0000], [0.4479,0.1750,0.0000], [0.3608,0.1382,0.0000],
        [0.2835,0.1070,0.0000], [0.2187,0.0816,0.0000], [0.1649,0.0610,0.0000],
        [0.1212,0.0446,0.0000], [0.0874,0.0320,0.0000], [0.0636,0.0232,0.0000],
        [0.0468,0.0170,0.0000], [0.0329,0.0119,0.0000], [0.0227,0.0082,0.0000],
        [0.0158,0.0057,0.0000], [0.0114,0.0041,0.0000], [0.0081,0.0029,0.0000],
        [0.0058,0.0021,0.0000], [0.0041,0.0015,0.0000], [0.0029,0.0010,0.0000],
        [0.0020,0.0007,0.0000], [0.0014,0.0005,0.0000], [0.0010,0.0004,0.0000],
        [0.0007,0.0002,0.0000], [0.0005,0.0002,0.0000], [0.0003,0.0001,0.0000],
        [0.0002,0.0001,0.0000], [0.0002,0.0001,0.0000], [0.0001,0.0000,0.0000],
        [0.0001,0.0000,0.0000], [0.0001,0.0000,0.0000], [0.0000,0.0000,0.0000]]

    X = 0
    Y = 0
    Z = 0
    for i, lamb in enumerate(range(380, 780, 5)): #lambda = 380; lambda < 780.1; i++, lambda += 5) {
        Me = spec_intens(lamb, temp);
        X += Me * cie_colour_match[i][0]
        Y += Me * cie_colour_match[i][1]
        Z += Me * cie_colour_match[i][2]
    XYZ = (X + Y + Z)
    x = X / XYZ;
    y = Y / XYZ;
    z = Z / XYZ;
    return(x, y, z)

def bb_spectrum(wavelength, bbTemp=5000):
    """
    Calculate, by Planck's radiation law, the emittance of a black body
    of temperature bbTemp at the given wavelength (in metres).  */
    """
    wlm = wavelength * 1e-9 # Convert to metres
    return (3.74183e-16 * math.pow(wlm, -5.0)) / (math.exp(1.4388e-2 / (wlm * bbTemp)) - 1.0)

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
    print("Temperature       x      y      z       R     G     B")
    print("-----------    ------ ------ ------   ----- ----- -----")

    for t in range(1000, 10000, 500):  # (t = 1000; t <= 10000; t+= 500) {
        x, y, z = spectrum_to_xyz(bb_spectrum, t)

        r, g, b = xyz_to_rgb(SMPTEsystem, x, y, z)

        r, g, b = constrain_rgb(r, g, b) # I omit the approximation bit here.
        r, g, b = norm_rgb(r, g, b)
        print("  %5.0f K      %.4f %.4f %.4f   %.3f %.3f %.3f" % (t, x, y, z, r, g, b))


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

    def __init__(self, n=30000):
        """ Initialize galaxy """

        # Excentricity of the innermost ellipse
        self._inner_excentricity = 0.8

        # Excentricity of the outermost ellipse
        self._outer_excentricity = 1.0

        #  Velovity at the innermost core in km/s
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
        dtype = [ ('theta',       np.float32, 1),
                  ('velocity',    np.float32, 1),
                  ('angle',       np.float32, 1),
                  ('m_a',         np.float32, 1),
                  ('m_b',         np.float32, 1),
                  ('size',        np.float32, 1),
                  ('type',        np.float32, 1),
                  ('temperature', np.float32, 1),
                  ('brightness',  np.float32, 1),
                  ('position',    np.float32, 2) ]
        n = self._stars_count + self._dust_count + 2*self._h2_count
        self._particles = np.zeros(n, dtype=dtype)

        i0 = 0
        i1 = i0  + self._stars_count
        self._stars = self._particles[i0:i1]
        self._stars['size'] = 3.
        self._stars['type'] = 0

        i0 = i1
        i1 = i0  + self._dust_count
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
        self._inner_excentricity = ex1
        self._outer_excentricity = ex2
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
        stars['m_a']        = R
        stars['angle']      = 90 - R * self._angular_offset
        stars['theta']      = np.random.uniform(0, 360, len(stars))
        stars['temperature']= np.random.uniform(3000, 9000, len(stars))
        stars['brightness'] = np.random.uniform(0.05, 0.25, len(stars))
        stars['velocity']   = 0.000005
        for i in range(len(stars)):
            stars['m_b'][i] = R[i]* self.excentricity(R[i])

        # Initialize dust
        # ---------------
        dust = self._dust
        X = np.random.uniform(0, 2*self._galaxy_radius, len(dust))
        Y = np.random.uniform(-self._galaxy_radius, self._galaxy_radius, len(dust))
        R = np.sqrt(X*X+Y*Y)
        dust['m_a']         = R
        dust['angle']       = R * self._angular_offset
        dust['theta']       = np.random.uniform(0, 360, len(dust))
        dust['velocity']    = 0.000005
        dust['temperature'] = 6000 + R/4
        dust['brightness']  = np.random.uniform(0.01,0.02)
        for i in range(len(dust)):
            dust['m_b'][i] = R[i] * self.excentricity(R[i])

        # Initialise H-II
        # ---------------
        h2a, h2b = self._h2a, self._h2b
        X = np.random.uniform(-self._galaxy_radius, self._galaxy_radius, len(h2a))
        Y = np.random.uniform(-self._galaxy_radius, self._galaxy_radius, len(h2a))
        R = np.sqrt(X*X+Y*Y)

        h2a['m_a']        = R
        h2b['m_a']        = R + 1000

        h2a['angle']      = R * self._angular_offset
        h2b['angle']      = h2a['angle']

        h2a['theta']      = np.random.uniform(0, 360, len(h2a))
        h2b['theta']      = h2a['theta']

        h2a['velocity']   = 0.000005
        h2b['velocity']   = 0.000005

        h2a['temperature'] = np.random.uniform(3000,9000,len(h2a))
        h2b['temperature'] = h2a['temperature']

        h2a['brightness']  = np.random.uniform(0.005,0.010, len(h2a))
        h2b['brightness']  = h2a['brightness']

        for i in range(len(h2a)):
            h2a['m_b'][i] = R[i] * self.excentricity(R[i])
        h2b['m_b'] = h2a['m_b']


    def update(self, timestep=100000):
        """ Update simulation """

        self._particles['theta'] += self._particles['velocity'] * timestep

        P = self._particles
        a,b = P['m_a'], P['m_b']
        theta, beta = P['theta'], -P['angle']

        alpha = theta * math.pi / 180.0
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        cos_beta  = np.cos(beta)
        sin_beta  = np.sin(beta)
        P['position'][:,0] = a*cos_alpha*cos_beta - b*sin_alpha*sin_beta
        P['position'][:,1] = a*cos_alpha*sin_beta + b*sin_alpha*cos_beta

        D = np.sqrt(((self._h2a['position'] - self._h2b['position'])**2).sum(axis=1))
        S = np.maximum(1,((1000-D)/10) - 50)
        self._h2a['size'] = 2.0*S
        self._h2b['size'] = S/6.0


    def excentricity(self, r):

        # Core region of the galaxy. Innermost part is round
        # excentricity increasing linear to the border of the core.
        if  r < self._core_radius:
            return 1 + (r / self._core_radius) * (self._inner_excentricity-1)

        elif r > self._core_radius and r <= self._galaxy_radius:
            a = self._galaxy_radius - self._core_radius
            b = self._outer_excentricity - self._inner_excentricity
            return self._inner_excentricity + (r - self._core_radius) / a * b

        # Excentricity is slowly reduced to 1.
        elif r > self._galaxy_radius and r < self._distant_radius:
            a = self._distant_radius - self._galaxy_radius
            b = 1 - self._outer_excentricity
            return self._outer_excentricity + (r - self._galaxy_radius) / a * b

        else:
            return 1
