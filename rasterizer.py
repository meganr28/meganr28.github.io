# Megan Reddy
# 3D Rasterizer
# To run, type
#   python rasterizer.py chair.txt
# All transformation matrices from OpenGL documentation (rotate, lookat matrices)

from PIL import Image
import sys
import math

# globals to store image, output filename, and function call shortcut
img = None
width = None
height = None
img_filename = None
putpixel = None

# globals to keep track of vertices and scanline points
vertex_list = []
scan_points = []

# globals to keep track of depth buffer, matrices, and current color
depth_buffer = None
vp = None
curr_mv = 0
curr_p = 0
color_list = [[255, 255, 255]]

# lighting globals
curr_normal = 0
use_flatnormals = False
scene_lights = []

"""
LINEAR ALGEBRA CLASSES AND FUNCTIONS
"""

class DepthBuffer:
    """
    Defines a depth buffer, with each pixel value initialized to 1
    """
    def __init__(self, width, height):
        self.buffer = []
        for i in range(width):
            col = []
            for i in range(height):
                col.append(1)
            self.buffer.append(col)

    def set(self, i, j, val):
        self.buffer[i][j] = val

    def get(self, i, j):
        return self.buffer[i][j]


class Matrix:
    """
    Defines a 4x4 matrix, initialized to 0
    """
    def __init__(self):
        self.matrix = []
        for i in range(4):
            self.matrix.append([0, 0, 0, 0])

    def load(self, elements):
        m = 0
        for i in range(4):
            for j in range(4):
                self.matrix[i][j] = elements[m]  # assumes that elements has correct # of elements
                m += 1

    def set(self, i, j, val):
        self.matrix[i][j] = val

    def get(self, i, j):
        return self.matrix[i][j]


class Vector:
    """
    Defines a vector of any length
    """
    def __init__(self, elements):
        self.vector = []
        for element in elements:
            self.vector.append(element)

    def add(self, other):
        new = Vector(self.vector)
        other_len = len(other.vector)  # make new vector, leave current untouched
        for i in range(other_len):
            new.vector[i] += other.vector[i]
        return new

    def sub(self, other):
        new = Vector(self.vector)
        other_len = len(other.vector)  # make new vector, leave current untouched
        for i in range(other_len):
            new.vector[i] -= other.vector[i]
        return new

    def mult(self, scalar):
        new = Vector(self.vector)
        for i in range(len(self.vector)):
            new.vector[i] *= scalar
        return new

    def div(self, scalar):
        new = Vector(self.vector)
        for i in range(len(self.vector)):
            new.vector[i] /= scalar
        return new

    def append(self, item):
        self.vector.append(item)

    def normalize(self):
        squared_sums = 0
        for item in self.vector:
            squared_sums += item ** 2
        magnitude = math.sqrt(squared_sums)
        norm_vector = Vector(self.vector)
        return norm_vector.div(magnitude)

    def set(self, i, val):
        self.vector[i] = val

    def get(self, i):
        return self.vector[i]

    def __eq__(self, other):
        return self.vector[0] == other.vector[0] \
               and self.vector[1] == other.vector[1] \
               and self.vector[2] == other.vector[2]

    def __copy__(self):
        return Vector(self.vector)


def multMbyM(a, b):
    """ multiplies two 4 x 4 matrices"""
    answer = Matrix()
    for row in range(4):
        for col in range(4):
            for i in range(4):
                val = answer.get(row, col) + (a.get(row, i) * b.get(i, col))
                answer.set(row, col, val)
    return answer


def multMbyV(m, v):
    """ multiplies a 4x4 matrix and a 4x1 vector """
    answer = v.__copy__()
    for row in range(4):
        answer.set(row, 0)
        for i in range(4):
            val = answer.get(row) + (m.get(row,i) * v.get(i))
            answer.set(row, val)
    return answer


def multVbyV(v1, v2):
    """ multiplies two vectors of the same length """
    answer = v1.__copy__()
    for i in range(len(v1.vector)):
        result = v1.get(i) * v2.get(i)
        answer.set(i, result)
    return answer


def dotProduct(v1, v2):
    """ dot product of two vectors """
    sum = 0
    for i in range(len(v1.vector)):
        sum += v1.get(i) * v2.get(i)

    return sum


def crossProduct(v1, v2):
    """ cross product of two vectors """
    rx = (v1.get(1) * v2.get(2)) - (v1.get(2) * v2.get(1))
    ry = (v1.get(2) * v2.get(0)) - (v1.get(0) * v2.get(2))
    rz = (v1.get(0) * v2.get(1)) - (v1.get(1) * v2.get(0))
    result = Vector([rx, ry, rz])

    return result


"""
MATRIX LOAD FUNCTIONS
"""


def loadmv(tokens):
    """ loads model/view matrix """
    global curr_mv

    mv = Matrix()
    elements = []
    for i in range(1, len(tokens)):
        elements.append(float(tokens[i]))
    mv.load(elements)
    curr_mv = mv


def loadp(tokens):
    """ loads projection matrix """
    global curr_p

    p = Matrix()
    elements = []
    for i in range(1, len(tokens)):
        elements.append(float(tokens[i]))
    p.load(elements)
    curr_p = p


def loadvp():
    """ initializes the viewport transformation """
    global vp

    elements = [int(width/2), 0, 0, int(width/2), 0, int(height/2), 0, int(height/2), 0, 0, 1, 0, 0, 0, 0, 1]
    viewport = Matrix()
    viewport.load(elements)
    vp = viewport


def frustum(tokens):
    """ replaces current projection matrix """
    global curr_p

    left = float(tokens[1])
    right = float(tokens[2])
    bottom = float(tokens[3])
    top = float(tokens[4])
    near = float(tokens[5])
    far = float(tokens[6])

    A = (right + left) / (right - left)
    B = (top + bottom) / (top - bottom)
    C = -(far + near) / (far - near)
    D = -(2 * far * near) / (far - near)

    # load matrix (from glFrustum documentation)
    elements = [(2 * near) / (right - left), 0, A, 0, 0, (2 * near) / (top - bottom), B, 0, 0, 0, C, D, 0, 0, -1, 0]
    f = Matrix()
    f.load(elements)
    curr_p = f


"""
MATRIX MANIPULATION FUNCTIONS
"""


def translate(tokens):
    """ adds translation to model/view transformations """
    global curr_mv

    t = Matrix()
    dx = float(tokens[1])
    dy = float(tokens[2])
    dz = float(tokens[3])
    elements = [1, 0, 0, dx, 0, 1, 0, dy, 0, 0, 1, dz, 0, 0, 0, 1]
    t.load(elements)

    if curr_mv:
        curr_mv = multMbyM(curr_mv, t)
    else:
        curr_mv = t


def rotatex(tokens):
    """ adds rotate x to model/view transformations """
    global curr_mv, mv_transforms

    rx = Matrix()
    d = float(tokens[1]) * (math.pi/180)
    elements = [1, 0, 0, 0, 0, math.cos(d), -math.sin(d), 0, 0, math.sin(d), math.cos(d), 0, 0, 0, 0, 1]
    rx.load(elements)

    if curr_mv:
        curr_mv = multMbyM(curr_mv, rx)
    else:
        curr_mv = rx


def rotatey(tokens):
    """ adds rotate y to model/view transformations """
    global curr_mv, mv_transforms

    ry = Matrix()
    d = float(tokens[1]) * (math.pi / 180)
    elements = [math.cos(d), 0, math.sin(d), 0, 0, 1, 0, 0, -math.sin(d), 0, math.cos(d), 0, 0, 0, 0, 1]
    ry.load(elements)

    if curr_mv:
        curr_mv = multMbyM(curr_mv, ry)
    else:
        curr_mv = ry


def rotatez(tokens):
    """ adds rotate z to model/view transformations """
    global curr_mv, mv_transforms

    rz = Matrix()
    d = float(tokens[1]) * (math.pi / 180)
    elements = [math.cos(d), -math.sin(d), 0, 0, math.sin(d), math.cos(d), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    rz.load(elements)

    if curr_mv:
        curr_mv = multMbyM(curr_mv, rz)
    else:
        curr_mv = rz


def rotate(tokens):
    """ adds rotate to model/view transformations """
    global curr_mv

    r = Matrix()

    # cos(theta) and sin (theta)
    d = float(tokens[1]) * (math.pi / 180)
    c = math.cos(d)
    s = math.sin(d)

    # axes
    xyz = Vector([float(tokens[2]), float(tokens[3]), float(tokens[4])])
    xyz = xyz.normalize()
    x = xyz.get(0)
    y = xyz.get(1)
    z = xyz.get(2)

    # load matrix (from glRotate documentation)
    oc = 1 - c
    elements = [(oc * (x ** 2)) + c, (oc * x * y) - s*z, (oc * x * z) + s*y, 0, (oc * x * y) + s*z, (oc * (y ** 2)) + c, (oc * y * z) - s*x, 0, (oc * x * z) - s*y, (oc * y * z) + s*x, (oc * (z ** 2)) + c, 0, 0, 0, 0, 1]
    r.load(elements)

    if curr_mv:
        curr_mv = multMbyM(curr_mv, r)
    else:
        curr_mv = r


def scale(tokens):
    """ adds translation to model/view transformations """
    global curr_mv

    s = Matrix()
    sx = float(tokens[1])
    sy = float(tokens[2])
    sz = float(tokens[3])
    elements = [sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1]
    s.load(elements)

    if curr_mv:
        curr_mv = multMbyM(curr_mv, s)
    else:
        curr_mv = s


def lookat(tokens):
    """ adds translation to model/view transformations """
    global vertex_list, curr_mv

    eye = vertex_list[int(tokens[1])-1]
    center = vertex_list[int(tokens[2])-1]
    f = center.sub(eye)
    up = Vector([float(tokens[3]), float(tokens[4]), float(tokens[5])])

    # normalize
    f_prime = f.normalize()
    up_prime = up.normalize()

    # cross products
    s = crossProduct(f_prime, up_prime)
    s = s.normalize()
    u = crossProduct(s, f_prime)

    # multiply matrices (from gluLookAt documentation)
    a = Matrix()
    a_elements = [s.get(0), s.get(1), s.get(2), 0, u.get(0), u.get(1), u.get(2), 0, -f_prime.get(0), -f_prime.get(1), -f_prime.get(2), 0, 0, 0, 0, 1]
    a.load(a_elements)
    b = Matrix()
    b_elements = [1, 0, 0, -eye.get(0), 0, 1, 0, -eye.get(1), 0, 0, 1, -eye.get(2), 0, 0, 0, 1]
    b.load(b_elements)

    la = multMbyM(a, b)
    curr_mv = la


def multmv(tokens):
    """ adds given model/view matrix to existing model/view transformations """
    global curr_mv

    mv = Matrix()
    elements = []
    for i in range(1, len(tokens)):
        elements.append(float(tokens[i]))
    mv.load(elements)

    if curr_mv:
        curr_mv = multMbyM(curr_mv, mv)
    else:
        curr_mv = mv


"""
VERTEX FUNCTIONS
"""

def xyz(tokens):
    """ adds (x,y,z,1) to the vertex list """
    global color_list

    proc_tokens = []
    for i in range(1, len(tokens)):
        proc_tokens.append(float(tokens[i]))
    proc_tokens.append(1)

    curr_color = color_list[-1]

    proc_tokens.append(curr_color[0])
    proc_tokens.append(curr_color[1])
    proc_tokens.append(curr_color[2])

    if curr_normal:  # add normal
        proc_tokens.append(curr_normal.get(0))
        proc_tokens.append(curr_normal.get(1))
        proc_tokens.append(curr_normal.get(2))

    vertex = Vector(proc_tokens)
    vertex_list.append(vertex)


def xyzw(tokens):
    """ adds (x,y,z,w) to the vertex list """
    global color_list

    proc_tokens = []
    for i in range(1, len(tokens)):
        proc_tokens.append(float(tokens[i]))

    curr_color = color_list[-1]

    proc_tokens.append(curr_color[0])
    proc_tokens.append(curr_color[1])
    proc_tokens.append(curr_color[2])

    if curr_normal:  # add normal
        proc_tokens.append(curr_normal.get(0))
        proc_tokens.append(curr_normal.get(1))
        proc_tokens.append(curr_normal.get(2))

    vertex = Vector(proc_tokens)
    vertex_list.append(vertex)


"""
COLOR FUNCTIONS
"""

def color(tokens):
    """ adds color to running list of colors """
    global color_list

    new_color = []
    for i in range(1, len(tokens)):
        token = float(tokens[i])
        new_color.append(token)

    color_list.append(new_color)


def map_color(color):
    """ maps color from floats to bytes """

    new_color = []
    for c in color:
        if c > 1:
            c = 1
        elif c < 0:
            c = 0
        add = c * 255
        new_color.append(int(add))

    final_color = (new_color[0], new_color[1], new_color[2])

    return final_color


"""
LIGHTING FUNCTIONS
"""


def normal(tokens):
    """ assigns current normal """
    global curr_normal

    new_normal = Vector([int(tokens[1]), int(tokens[2]), int(tokens[3])])
    curr_normal = new_normal


def flatnormals(tokens):
    """ use perpendicular vector of triangle instead of per-vertex normals """
    global use_flatnormals

    use_flatnormals = True


def sunlight(tokens):
    """ adds sunlight to list of lights """
    global curr_normal, color_list, scene_lights

    ld = Vector([int(tokens[1]), int(tokens[2]), int(tokens[3])])
    norm_ld = ld.normalize() # store normalized direction

    color = color_list[-1]

    light = [norm_ld, color[0], color[1], color[2]]
    scene_lights.append(light)


def calc_lighting(normal, object_color):
    """ uses Lambert's law to calculate fragment color """
    oc = Vector([object_color[0], object_color[1], object_color[2]])  # current object color
    n = Vector([normal[0], normal[1], normal[2]])  # current normal
    n = n.normalize()

    color = Vector([0, 0, 0])
    for light in scene_lights:
        ld = light[0]
        c = dotProduct(ld, n)  # calculate cosine of angle
        if c < 0:  # if negative, don't add light
            continue
        lc = Vector([light[1], light[2], light[3]])  # light color
        lo = multVbyV(lc, oc)  # multiply light and object color
        fc = lo.mult(c)  # multiply by cos(theta)
        color = color.add(fc)

    final_color = (color.get(0), color.get(1), color.get(2))
    return final_color


"""
TRIANGLE FUNCTIONS
"""

def valid(pixel):
    """ checks if pixel should be drawn """
    global width, height, depth_buffer

    draw = True
    p = pixel.vector
    px = p[0]
    py = p[1]
    pz = p[2]

    if px < 0 or px >= width:
        draw = False
    if py < 0 or py >= height:
        draw = False
    if pz < 0 or pz > 1: 
        draw = False

    if draw:
        if pz > depth_buffer.get(round(px), round(py)):
            draw = False
        else:
            depth_buffer.set(round(px), round(py), pz)

    return draw


def triangle(tokens):
    """ draws a triangle """
    global scan_points, vp, curr_mv, curr_p

    scan_points.clear()  # clear global scan_points list on each call

    # pick correct vertices from vertex_list
    vcs = [int(tokens[1]), int(tokens[2]), int(tokens[3])]
    idx = get_vertices(vcs)
    vcs = [vertex_list[idx[0]], vertex_list[idx[1]], vertex_list[idx[2]]]

    # find perpendicular vector
    p2p1 = vcs[1].sub(vcs[0])
    p3p1 = vcs[2].sub(vcs[0])
    pvec = crossProduct(p2p1, p3p1)

    # sort in ascending order by y
    vcs.sort(key=lambda vcs: vcs.vector[1])

    curr = vcs  # keep track of array to operate on at each step

    # apply model/view transformations to each vertex
    if curr_mv:
        mv_vcs = []
        for v in curr:
            mv_vertex = multMbyV(curr_mv, v)  # this is a Vector
            mv_vcs.append(mv_vertex)
        curr = mv_vcs

    # apply projection to each vertex
    if curr_p:
        p_vcs = []
        for v in curr:
            p_vertex = multMbyV(curr_p, v)
            p_vcs.append(p_vertex)
        curr = p_vcs

    # divide each x, y, and z by w
    for i in range(len(curr)):
        curr[i] = curr[i].div(curr[i].get(3))

    # apply viewport transformation
    vp_vcs = []
    for v in curr:
        vp_vertex = multMbyV(vp, v)
        vp_vcs.append(vp_vertex)

    # rasterize into fragments
    i1 = vp_vcs[0]
    i2 = vp_vcs[1]
    i3 = vp_vcs[2]

    # DDA convert edges
    if not (i1.vector[1] == i2.vector[1]):
        dda(i1, i2, pvec, False, True)
    if not (i2.vector[1] == i3.vector[1]):
        dda(i2, i3, pvec, False, True)
    dda(i1, i3, pvec, False, True)

    # DDA convert scanlines, using stored points from edge scan conversion
    if tokens[0] == "trig":
        gouraud = True
    else:
        gouraud = False
    scan_points.sort(key=lambda scan_points: scan_points.vector[1])
    for i in range(0, len(scan_points), 2):
        dda(scan_points[i], scan_points[i+1], pvec, gouraud)


def dda(ep1, ep2, pvec, interpolate=False, scan_edge=False):
    """ DDA algorithm implementation """
    global scan_points, color_list, scene_lights

    if ep1 == ep2:
        return

    # calculate deltas
    deltas = ep2.sub(ep1)
    dx = deltas.get(0)
    dy = deltas.get(1)

    step_index = 0  # step in x
    if (abs(dy) > abs(dx)) or scan_edge:  # if dy > dx, step in y
        step_index = 1
    if (ep2.get(step_index) - ep1.get(step_index)) < 0:  # switch endpoints if necessary
        temp = ep1
        ep1 = ep2
        ep2 = temp

    # once you have correct endpoints, determine the offsets
    offsets = calc_offsets(ep1, deltas, step_index)
    initial_offset = offsets[0]
    offset = offsets[1]

    # set starting pixel/colors
    curr_point = ep1
    curr_point = curr_point.add(initial_offset)  # use vector add

    while curr_point.get(step_index) < ep2.get(step_index):
        if scan_edge:  # make copy of current point to add to scanline list
            scan_point = curr_point.__copy__()
            scan_points.append(scan_point)
        else:
            if valid(curr_point):
                if interpolate:
                    rgba_value = (curr_point.get(4), curr_point.get(5), curr_point.get(6))
                else:
                    rgba_value = color_list[-1]
                # add lighting
                if len(scene_lights) > 0:
                    if use_flatnormals:
                        normal = (pvec.get(0), pvec.get(1), pvec.get(2))
                    else:
                        normal = (curr_point.get(7), curr_point.get(8), curr_point.get(9))
                    rgba_value = calc_lighting(normal, rgba_value)
                # map to bytes
                rgba_value = map_color(rgba_value)
                putpixel((round(curr_point.get(0)), round(curr_point.get(1))), rgba_value)  # set the pixel and its color
        # update current point
        curr_point = curr_point.add(offset)


def calc_offsets(ep1, deltas, step_index):
    """ calculates the initial and general offsets for the DDA algorithm """
    p1_i = ep1.get(step_index)   # starting value
    diff = math.ceil(p1_i) - p1_i   # find distance to next integer

    # make local list of deltas (vector)
    local_deltas = deltas

    # determine what to divide by
    step = deltas.get(step_index)

    offset = local_deltas.div(step)
    initial_offset = offset.mult(diff)

    offsets = [initial_offset, offset]
    return offsets


"""
MAIN PROGRAM FUNCTIONS
"""


def read_file(filename):
    """ read file and add each valid (in keywords) line to commands list """
    commands = []
    file = open(filename)
    for line in file:
        tokens = line.strip().split()
        if len(tokens) > 0:
            if tokens[0] in keywords.keys():
                commands.append(tokens)
    return commands


def create_png(tokens):
    """ reads in width and height and creates appropriate png file and initializes global variables """
    global img, width, height, img_filename, putpixel, depth_buffer
    width = int(tokens[1])
    height = int(tokens[2])
    img_filename = tokens[3]
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    putpixel = img.im.putpixel
    depth_buffer = DepthBuffer(width, height)
    loadvp()


def get_vertices(vcs):
    """ returns the vertex numbers for indexing the global vertex_list """
    final_vcs = []
    for vertex in vcs:
        curr = vertex
        if vertex > 0:
            curr = vertex - 1
        final_vcs.append(curr)
    return final_vcs


"""
MAIN PROGRAM 
"""

# valid keywords dictionary
keywords = {"png": create_png,
            "xyz": xyz,
            "xyzw": xyzw,
            "trif": triangle,
            "trig": triangle,
            "color": color,
            "loadmv": loadmv,
            "loadp": loadp,
            "frustum": frustum,
            "translate": translate,
            "scale": scale,
            "lookat": lookat,
            "multmv": multmv,
            "rotate": rotate,
            "rotatex": rotatex,
            "rotatey": rotatey,
            "rotatez": rotatez,
            "normal": normal,
            "sunlight": sunlight,
            "flatnormals": flatnormals}

# get input filename from the command line
filename = sys.argv[1]

# save valid list of commands
commands = read_file(filename)

# process each command and set the appropriate pixel
for command in commands:
    first_token = command[0]
    operation = keywords[first_token]
    operation(command)

# save the final image as a png
img.save(img_filename)