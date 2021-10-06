from tkinter import *
from math import sqrt, cos, sin, pi

tau = 2*pi

scale = 60
phys_offset = 250

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

def draw_edges(w, basis, points, edges, rotation_matrix):
    for i in multi_range([-3] * dimension, [3] * dimension):
        for edge in edges:
            point1 = vadd(points[edge[0][0]], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge[0][1], basis)])
            moved_point1 = vadd(point1, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(i, basis)])
            point2 = vadd(points[edge[1][0]], [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge[1][1], basis)])
            moved_point2 = vadd(point2, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(i, basis)])
            rotated_point1 = matrix_mult(rotation_matrix, moved_point1)
            rotated_point2 = matrix_mult(rotation_matrix, moved_point2)
            flat_point1 = truncto2(rotated_point1)
            flat_point2 = truncto2(rotated_point2)
            draw_line(w, flat_point1, flat_point2)

# draws points on screen given abstract coordinates in n-space
# first convert from n-space to 2-space
# then convert from virtual coordinates to physical coordinates
def draw_points(w, basis, points, rotation_matrix):
    for i in multi_range([-3] * dimension, [3] * dimension):
        for point in points:
            moved_point = vadd(point, [scalar_mult(i[k], basis[k]) for k in range(len(i))])
            rotated_point = matrix_mult(rotation_matrix, moved_point)
            flat_point = truncto2(rotated_point)
            draw_point(w, flat_point)

# draw a point on the screen, given abstract 2D coordinates
def draw_point(w, point_2d):
    assert len(point_2d) == 2
    phys_point = coord_transform(point_2d)
    px = phys_point[0]
    py = phys_point[1]
    w.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#000000")

# draw a line on the sceren, given abstract 2D coordinates
def draw_line(w, point_2d_1, point_2d_2):
    w.create_line(*coord_transform(point_2d_1), *coord_transform(point_2d_2), fil="#000000")

def matrix_mult(mat, vec):
    return [sum([mat[i][j]*vec[j] for j in range(len(vec))]) for i in range(len(mat))]

# multiplies two matrices stored in row-major order
def two_matrix_mult(mat1, mat2):
    dim = len(mat1)
    return [[sum([mat1[a][b]*mat2[b][c] for b in range(dim)]) for c in range(dim)] for a in range(dim)]

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
    if dimension == 3:
        rotation = [[1,0,0],[0,1,0],[0,0,1]]
        #rotation = [[1,0,0],[0,cos(tau/16),sin(tau/16)],[0,-sin(tau/16),cos(tau/16)]]
    if dimension == 2:
        rotation = [[1,0],[0,1]]
        #rotation = [[cos(tau/16),sin(tau/16)],[-sin(tau/16),cos(tau/16)]]
    if dimension == 1:
        rotation = [[1]]

    def left_button_fn():
        if dimension != 3: return
        global rotation
        rotation = rotate_left(rotation)
        w.delete("all")
        draw_edges(w, basis, points, edges, rotation)
        draw_points(w, basis, points, rotation)        

    def right_button_fn():
        if dimension != 3: return
        global rotation
        rotation = rotate_right(rotation)
        w.delete("all")
        draw_edges(w, basis, points, edges, rotation)
        draw_points(w, basis, points, rotation)        

    def up_button_fn():
        if dimension != 3: return
        global rotation
        rotation = rotate_up(rotation)
        w.delete("all")
        draw_edges(w, basis, points, edges, rotation)
        draw_points(w, basis, points, rotation)        

    def down_button_fn():
        if dimension != 3: return
        global rotation
        rotation = rotate_down(rotation)
        w.delete("all")
        draw_edges(w, basis, points, edges, rotation)
        draw_points(w, basis, points, rotation)        

    def cw_button_fn():
        global rotation
        if dimension == 3:
            rotation = rotate_cw(rotation)
        if dimension == 2:
            rotation = rotate_cw_2d(rotation)
        w.delete("all")
        draw_edges(w, basis, points, edges, rotation)
        draw_points(w, basis, points, rotation)        

    def ccw_button_fn():
        global rotation
        if dimension == 3:
            rotation = rotate_ccw(rotation)
        if dimension == 2:
            rotation = rotate_ccw_2d(rotation)
        w.delete("all")
        draw_edges(w, basis, points, edges, rotation)
        draw_points(w, basis, points, rotation)        

    draw_edges(w, basis, points, edges, rotation)
    draw_points(w, basis, points, rotation)

    left_button.configure(command=left_button_fn)
    right_button.configure(command=right_button_fn)
    up_button.configure(command=up_button_fn)
    down_button.configure(command=down_button_fn)
    cw_button.configure(command=cw_button_fn)
    ccw_button.configure(command=ccw_button_fn)

    mainloop()
