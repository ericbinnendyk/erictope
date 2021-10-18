from tkinter import *
from math import sqrt, cos, sin, pi
from itertools import product

tau = 2*pi

scale = 60
phys_offset = 250
background_gray = 217

# About the vector spaces used in Erictope:
# Abstract n-D coordinates: Universal set of coordinates used when describing features of the shape, specified in the input file. They do not depend on rotation or screen position.
# Abstract rotated n-D coordinates: Coordinates of the rotated shape, in the same abstract space as the universal coordinates.
# Concrete n-D coordinates: the first two coordinates are the literal pixel coordinates on screen, calculated from the abstract rotated coordinates, screen dimensions, and zoom level (when it is implemented). Further coordinates are an extension of this idea as if the screen was higher dimensional, used for determining the level of fading (not fully implemented)
# Concrete 2D coordinates: the first two of the concrete n-D coordinates, giving the actual position on screen.

# adds a vector to a list of basis vectors
def vadd(p, vl):
    assert all(map(lambda v: len(p) == len(v), vl))
    for v in vl:
        p = [p[i] + v[i] for i in range(len(p))]
    return p

# transform from abstract 2D coordinates in mathematical model to physical ones on screen
def coord_transform(twodcoords):
    global scale, offset
    assert len(twodcoords) == 2
    return [c*scale + phys_offset for c in twodcoords]

def scalar_mult(k, v):
    return [k*x for x in v]

def truncto2(point):
    return point[:2]

def multi_range(a, b):
    assert len(a) == len(b)
    n_nums = len(a)
    curr = a[:]
    yield tuple(curr)
    while True:
        # to increment tuple, find last value that can be incremented
        # increment and set all subsequent values to corresponding values in a
        diff = [b[i] - curr[i] for i in range(n_nums)]
        incrementable_inds = [i for i in range(n_nums) if diff[i] > 1]
        if len(incrementable_inds) == 0:
            # noting more to increment, nothing more to yield
            return
        last = incrementable_inds[-1]
        curr[last] += 1
        for i in range(last + 1, n_nums):
            curr[i] = a[i]
        yield tuple(curr)

def rotated_point_in_range(rotated_point):
    phys_point = coord_transform(truncto2(rotated_point))
    if len(rotated_point) == 3:
        return all([-100 <= x <= 600 for x in phys_point]) and -3 <= rotated_point[2] <= 3
    if len(rotated_point) == 2:
        return all([-100 <= x <= 600 for x in phys_point])
    raise RuntimeError("Dimension not implemented yet.")

def in_range(dimension, offset, basis, points, rotation_matrix):
    for point in points:
        moved_point = vadd(point, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, basis)])
        rotated_point = matrix_mult(rotation_matrix, moved_point)
        if rotated_point_in_range(rotated_point):
            return True
    return False

# gets the offsets (multiples of the basis vectors) of copies of the fundamental translational domain points that are within range of visibility in the final window, and thus should be rendered
def get_offsets_in_range(basis, points, rotation_matrix):
    def pop(queue):
        first = queue[0]
        del queue[0]
        return first

    def get_neighbors(offset):
        neighs = []
        for i in range(dimension):
            neighbor = offset[:]
            neighbor[i] -= 1
            neighs.append(neighbor)
            neighbor = offset[:]
            neighbor[i] += 1
            neighs.append(neighbor)
        return neighs

    dimension = len(basis[0])
    queue = [[0]*dimension]
    offsets = [[0]*dimension]
    while len(queue) > 0:
        x = pop(queue)
        neighbors = get_neighbors(x)
        for n in neighbors:
            if n not in offsets:
                if in_range(dimension, n, basis, points, rotation_matrix):
                    queue.append(n)
                    offsets.append(n)
                else:
                    moved_point = vadd(points[0], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(n, basis)])
                    rotated_point = matrix_mult(rotation_matrix, moved_point)
    return offsets

# edges are written in abstract non-rotated coordinate space
def draw_edges(w, basis, points, edges, offsets, rotation_matrix):
    # determines the z coordinate of the midpoint of the edge in abstract rotated coordinate space
    def edge_depth(edge, offset):
        assert(len(points[edge[0][0]]) == 3)

        # coordinates of first point with zero offset
        point1 = vadd(points[edge[0][0]], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge[0][1], basis)])
        # coordinates of first point with correct offset
        moved_point1 = vadd(point1, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, basis)])
        # coordinates of second point with zero offset
        point2 = vadd(points[edge[1][0]], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge[1][1], basis)])
        # coordinates of second point with correct offset
        moved_point2 = vadd(point2, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, basis)])
        # rotated coordinates of first point
        rotated_point1 = matrix_mult(rotation_matrix, moved_point1)
        # rotated coordinates of second point
        rotated_point2 = matrix_mult(rotation_matrix, moved_point2)
        return (rotated_point1[2] + rotated_point2[2])/2

    edgesoffsets = list(product(edges, offsets))
    # if the shape is 3d, we want to draw the far edges before the near edges (due to the fading color)
    if len(basis[0]) == 3:
        edgesoffsets.sort(key=lambda x: edge_depth(*x))
    for edge, offset in edgesoffsets:
        # coordinates of first point with zero offset
        point1 = vadd(points[edge[0][0]], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge[0][1], basis)])
        # coordinates of first point with correct offset
        moved_point1 = vadd(point1, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, basis)])
        # coordinates of second point with zero offset
        point2 = vadd(points[edge[1][0]], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge[1][1], basis)])
        # coordinates of second point with correct offset
        moved_point2 = vadd(point2, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, basis)])
        # rotated coordinates of first point
        rotated_point1 = matrix_mult(rotation_matrix, moved_point1)
        # rotated coordinates of second point
        rotated_point2 = matrix_mult(rotation_matrix, moved_point2)
        draw_line(w, rotated_point1, rotated_point2)
    '''for i in multi_range([-3] * dimension, [3] * dimension):
        for edge in edges:
            point1 = vadd(points[edge[0][0]], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge[0][1], basis)])
            moved_point1 = vadd(point1, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(i, basis)])
            point2 = vadd(points[edge[1][0]], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge[1][1], basis)])
            moved_point2 = vadd(point2, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(i, basis)])
            rotated_point1 = matrix_mult(rotation_matrix, moved_point1)
            rotated_point2 = matrix_mult(rotation_matrix, moved_point2)
            flat_point1 = truncto2(rotated_point1)
            flat_point2 = truncto2(rotated_point2)
            draw_line(w, flat_point1, flat_point2)'''

# draws points on screen given abstract coordinates in n-space
# convert from abstract coordinates to abstract rotated coordinates, then draw rotated coordinates
def draw_points(w, basis, points, offsets, rotation_matrix):
    # determines the z coordinate of the point in abstract rotated coordinate space
    def point_depth(point, offset):
        assert(len(point) == 3)

        moved_point = vadd(point, [scalar_mult(offset[k], basis[k]) for k in range(len(offset))])
        rotated_point = matrix_mult(rotation_matrix, moved_point)
        return rotated_point[2]

    pointsoffsets = list(product(points, offsets))
    # if the shape is 3d, we want to draw the far edges before the near edges (due to the fading color)
    if len(basis[0]) == 3:
        pointsoffsets.sort(key=lambda x: point_depth(*x))
    '''for i in multi_range([-3] * dimension, [3] * dimension):'''
    for point, offset in pointsoffsets:
        moved_point = vadd(point, [scalar_mult(offset[k], basis[k]) for k in range(len(offset))])
        rotated_point = matrix_mult(rotation_matrix, moved_point)
        draw_point(w, rotated_point)

# draw a point on the screen, given abstract rotated 2D coordinates
def draw_point(w, point):
    point_2d = truncto2(point)
    gray = 0
    # determine how faded ("far away") to draw the point if it has negative z coordinate in "concrete 3D space"
    if len(point) > 2:
        if point[2] > 3 or point[2] < -3:
            return # too far away from center, don't draw anything
        if point[2] < 0:
            gray = int(-point[2] / 3 * background_gray)
    phys_point = coord_transform(point_2d)
    px = phys_point[0]
    py = phys_point[1]
    w.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#%02x%02x%02x" % (gray, gray, gray), width=0)

# draw a line on the sceren, given abstract rotated 2D coordinates of endpoints
def draw_line(w, point_1, point_2):
    point_2d_1 = truncto2(point_1)
    point_2d_2 = truncto2(point_2)
    gray = 0
    # determine how faded ("far away") to draw the line if it has negative z coordinates in "concrete 3D space"
    if len(point_1) > 2:
        z_midpoint = (point_1[2] + point_2[2])/2
        if z_midpoint > 3 or z_midpoint < -3:
            return # too far away from center, don't draw anything
        if z_midpoint < 0:
            gray = int(-z_midpoint / 3 * background_gray)
    w.create_line(*coord_transform(point_2d_1), *coord_transform(point_2d_2), fill="#%02x%02x%02x" % (gray, gray, gray))

def draw_honeycomb(w, basis, points, edges, rotation):
    w.delete("all")
    offsets = get_offsets_in_range(basis, points, rotation)
    draw_edges(w, basis, points, edges, offsets, rotation)
    draw_points(w, basis, points, offsets, rotation)

def matrix_mult(mat, vec):
    return [sum([mat[i][j]*vec[j] for j in range(len(vec))]) for i in range(len(mat))]

# multiplies two matrices stored in row-major order
def two_matrix_mult(mat1, mat2):
    dim = len(mat1)
    return [[sum([mat1[a][b]*mat2[b][c] for b in range(dim)]) for c in range(dim)] for a in range(dim)]

def identity(dim):
    return [[1 if j == i else 0 for j in range(dim)] for i in range(dim)]

def rotate_up(rotation):
    delta = [[1,0,0],[0,cos(tau/24),-sin(tau/24)],[0,sin(tau/24),cos(tau/24)]]
    new_rotation = two_matrix_mult(delta, rotation)
    return new_rotation

def rotate_down(rotation):
    delta = [[1,0,0],[0,cos(tau/24),sin(tau/24)],[0,-sin(tau/24),cos(tau/24)]]
    new_rotation = two_matrix_mult(delta, rotation)
    return new_rotation

def rotate_left(rotation):
    delta = [[cos(tau/24),0,-sin(tau/24)],[0,1,0],[sin(tau/24),0,cos(tau/24)]]
    new_rotation = two_matrix_mult(delta, rotation)
    return new_rotation

def rotate_right(rotation):
    delta = [[cos(tau/24),0,sin(tau/24)],[0,1,0],[-sin(tau/24),0,cos(tau/24)]]
    new_rotation = two_matrix_mult(delta, rotation)
    return new_rotation

def rotate_ccw(rotation):
    delta = [[cos(tau/24),sin(tau/24),0],[-sin(tau/24),cos(tau/24),0],[0,0,1]]
    new_rotation = two_matrix_mult(delta, rotation)
    return new_rotation

def rotate_cw(rotation):
    delta = [[cos(tau/24),-sin(tau/24),0],[sin(tau/24),cos(tau/24),0],[0,0,1]]
    new_rotation = two_matrix_mult(delta, rotation)
    return new_rotation

def rotate_ccw_2d(rotation):
    delta = [[cos(tau/24),sin(tau/24)],[-sin(tau/24),cos(tau/24)]]
    new_rotation = two_matrix_mult(delta, rotation)
    return new_rotation

def rotate_cw_2d(rotation):
    delta = [[cos(tau/24),-sin(tau/24)],[sin(tau/24),cos(tau/24)]]
    new_rotation = two_matrix_mult(delta, rotation)
    return new_rotation

if __name__ == "__main__":
    mode = 0
    basis = [] # the basis vectors describing the symmetry of the infinite repeating pattern
    points = [] # the points in each unit of translational symmetry
    edges = [] # the edges in each unit of translational symmetry
    dimension = -1

    # open up tkinter canvas
    master = Tk()
    master.title("Erictope")
    w = Canvas(master, width=500, height=500)
    w.pack()

    # add buttons
    left_button = Button(master, text = "<<", fg = "Black", bg = "Gray")
    left_button.pack()
    right_button = Button(master, text = ">>", fg = "Black", bg = "Gray")
    right_button.pack()
    up_button = Button(master, text = "^", fg = "Black", bg = "Gray")
    up_button.pack()
    down_button = Button(master, text = "v", fg = "Black", bg = "Gray")
    down_button.pack()
    cw_button = Button(master, text = "Clockwise", fg = "Black", bg = "Gray")
    cw_button.pack()
    ccw_button = Button(master, text = "Counterclockwise", fg = "Black", bg = "Gray")
    ccw_button.pack()
    reset_button = Button(master, text = "Reset position", fg = "Black", bg = "Gray")
    reset_button.pack()

    print("Welcome to Erictope!")
    path = input("Enter file to read coordinates from: ")
    data = open(path, 'r')
    for line in data:
        line = line.strip('\n')
        if line.strip() == '' or line[0] == '#':
            continue
        elif line[0] == '@':
            if line[0:] == 'DIMENSION':
                mode = 0
            if line[1:] == 'BASIS':
                mode = 1
            if line[1:] == 'POINTS':
                mode = 2
            if line[1:] == 'EDGES':
                mode = 3
        else:
            if mode == 0:
                dimension = int(line)
                if dimension < 0:
                    raise RuntimeError("Error reading data: Dimension out of range.")
            if mode == 1:
                vstr = line.split(',')
                if len(vstr) != dimension:
                    raise RuntimeError("Error in \'{}\': number of coordinates is different from dimension.".format(line))
                v = [float(eval(s)) for s in vstr]
                basis.append(v)
                if len(basis) > dimension:
                    raise RuntimeError("Basis length is greater than dimension.")
            if mode == 2:
                pstr = line.split(',')
                if len(pstr) != dimension:
                    raise RuntimeError("Error in \'{}\': number of coordinates is different from dimension.".format(line))
                p = [float(eval(s)) for s in pstr]
                points.append(p)
            if mode == 3:
                inbrackets = False
                l = [] # format: [(p1ind, [offsetx, offsety]), (p2ind, [offsetx, offsety])]
                lstr = []
                breaks = []
                for i, c in enumerate(line):
                    if c == '[':
                        if inbrackets:
                            raise RuntimeError("Parsing error.")
                        inbrackets = True
                    if c == ']':
                        if not inbrackets:
                            raise RuntimeError("Parsing error.")
                        inbrackets = False
                    if c == ',' and not inbrackets:
                        breaks.append(i)
                breaks.append(len(line))
                start = 0
                for end in breaks:
                    lstr.append(line[start:end])
                    start = end + 1
                # now each entry of lstr is of the form a or a[b,c]
                if len(lstr) != 2:
                    raise RuntimeError("Error in \'{}\': two point identifiers are needed.".format(line))
                for entry in lstr:
                    if '[' in entry:
                        split_entry = entry.split('[')
                        assert len(split_entry) == 2
                        [pindstr, offsetstr] = split_entry
                        offsetstr = offsetstr.strip(']')
                        pind = int(pindstr)
                        split_offsetstr = offsetstr.split(',')
                        if len(split_offsetstr) != len(basis):
                            raise RuntimeError("Error in \'{}\': length of offset vector different from basis length.".format(line))
                        offset = list(map(int, split_offsetstr))
                    else:
                        pind = int(entry)
                        offset = [0] * len(basis)
                    l.append((pind, offset))
                edges.append(l)

    data.close()

    if dimension == -1:
        print("Error: No dimension specified.")
        exit()

    '''if dimension == 2:
        flat_points = points
        flat_basis = basis

    if dimension > 2:
        flat_points = truncto2(points)
        flat_basis = truncto2(basis)'''

    #flat_edges_coords = []
    '''for edge in edges:
        pind0, offset0 = edge[0]
        pind1, offset1 = edge[1]
        p0 = flat_points[pind0]
        print(offset1)
        for i in range(dimension):
            p0 = [p0[j] + offset0[i]*flat_basis[i][j] for j in range(2)]
        p1 = flat_points[pind1]
        for i in range(dimension):
            p1 = [p1[j] + offset1[i]*flat_basis[i][j] for j in range(2)]
        flat_edges_coords.append([p0, p1])'''

    # Let's start with the points, and then apply the transformation, and then draw the points
    rotation = identity(dimension)

    def left_button_fn():
        if dimension != 3: return
        global rotation
        rotation = rotate_left(rotation)
        draw_honeycomb(w, basis, points, edges, rotation)

    def right_button_fn():
        if dimension != 3: return
        global rotation
        rotation = rotate_right(rotation)
        draw_honeycomb(w, basis, points, edges, rotation)

    def up_button_fn():
        if dimension != 3: return
        global rotation
        rotation = rotate_up(rotation)
        draw_honeycomb(w, basis, points, edges, rotation)

    def down_button_fn():
        if dimension != 3: return
        global rotation
        rotation = rotate_down(rotation)
        draw_honeycomb(w, basis, points, edges, rotation)

    def cw_button_fn():
        global rotation
        if dimension == 3:
            rotation = rotate_cw(rotation)
        if dimension == 2:
            rotation = rotate_cw_2d(rotation)
        draw_honeycomb(w, basis, points, edges, rotation)

    def ccw_button_fn():
        global rotation
        if dimension == 3:
            rotation = rotate_ccw(rotation)
        if dimension == 2:
            rotation = rotate_ccw_2d(rotation)
        draw_honeycomb(w, basis, points, edges, rotation)

    def reset_position():
        global rotation
        rotation = identity(dimension)
        draw_honeycomb(w, basis, points, edges, rotation)

    draw_honeycomb(w, basis, points, edges, rotation)

    left_button.configure(command=left_button_fn)
    right_button.configure(command=right_button_fn)
    up_button.configure(command=up_button_fn)
    down_button.configure(command=down_button_fn)
    cw_button.configure(command=cw_button_fn)
    ccw_button.configure(command=ccw_button_fn)
    reset_button.configure(command=reset_position)

    mainloop()

