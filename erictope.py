# This is a test of the new Polytope class

from tkinter import *
from math import sqrt, cos, sin, pi
from itertools import product

tau = 2*pi

scale = 60
phys_offset = 250
background_gray = 217

# transform from abstract rotated 2D coordinates in mathematical model to physical ones on screen
def coord_transform(twodcoords):
    global scale, offset
    assert len(twodcoords) == 2
    return [c*scale + phys_offset for c in twodcoords]

# draw a line on the screen, given abstract rotated 2D coordinates of endpoints
def direct_draw_line(w, point_1, point_2):
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

# draw a point on the screen, given abstract rotated nD coordinates
def direct_draw_point(w, point_coords):
    point_2d = truncto2(point_coords)
    gray = 0
    # determine how faded ("far away") to draw the point if it has negative z coordinate in "concrete 3D space"
    if len(point_coords) > 2:
        if point_coords[2] > 3 or point_coords[2] < -3:
            return # too far away from center, don't draw anything
        if point_coords[2] < 0:
            gray = int(-point_coords[2] / 3 * background_gray)
    phys_point = coord_transform(point_2d)
    px = phys_point[0]
    py = phys_point[1]
    w.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#%02x%02x%02x" % (gray, gray, gray), width=0)

def matrix_mult(mat, vec):
    return [sum([mat[i][j]*vec[j] for j in range(len(vec))]) for i in range(len(mat))]

# multiplies two matrices stored in row-major order
def two_matrix_mult(mat1, mat2):
    dim = len(mat1)
    return [[sum([mat1[a][b]*mat2[b][c] for b in range(dim)]) for c in range(dim)] for a in range(dim)]

def identity(dim):
    return [[1 if j == i else 0 for j in range(dim)] for i in range(dim)]

def in_range(dimension, offset, basis, points, rotation_matrix):
    for point in points:
        moved_point = vadd(point.coords, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, basis)])
        rotated_point = matrix_mult(rotation_matrix, moved_point)
        if rotated_point_in_range(rotated_point):
            return True
    return False

def rotated_point_in_range(rotated_point):
    phys_point = coord_transform(truncto2(rotated_point))
    if len(rotated_point) == 3:
        return all([-100 <= x <= 600 for x in phys_point]) and -3 <= rotated_point[2] <= 3
    if len(rotated_point) == 2:
        return all([-100 <= x <= 600 for x in phys_point])
    raise RuntimeError("Dimension not implemented yet.")

def scalar_mult(k, v):
    return [k*x for x in v]

def truncto2(point):
    return point[:2]

# adds a vector to a list of basis vectors
def vadd(p, vl):
    assert all(map(lambda v: len(p) == len(v), vl))
    for v in vl:
        p = [p[i] + v[i] for i in range(len(p))]
    return p

class Polytope: # Is there a way to make this associated with exactly one instance of the tkinter Canvas class? If not it's okay, I can make the Canvas global, it's not like I'll be needing more than one
    #wait I guess I can just make it an instance variable nvm :unknown:
    # class variables:
    # dimension
    # basis - basis translation vectors in x, y, z directions
    # points - list of Point objects referencing coordinates of vertices (abstract, non-rotated n-D coordinates)
    # edges - pairs of indices into the vertice list that represent edges, each with an offset
    # no faces, cells, etc. yet. sorry.
    # rotation - the rotation matrix that gives the particular rotation being rendered
    def __init__(self, dimension, basis, points, edges):
        self.dimension = dimension
        self.basis = basis
        self.points = points
        # make sure each point, initially associated with None, is now associated with this polytope
        for point in self.points:
            point.polytope = self
        self.edges = edges
        # and also associate edges with polytope
        for edge in edges:
            edge.polytope = self
        self.rotation = identity(dimension)
    # based on draw_honeycomb
    def draw(self, w):
        w.delete("all")
        offsets = self._get_offsets_in_range()
        self._draw_edges(w, offsets)
        self._draw_points(w, offsets)
        if self.dimension == 2:
            # gray out four of the rotation buttons
            # to do
            pass
        # display dimension on canvas
        # to do
        pass
    def _get_offsets_in_range(self):
        def pop(queue):
            first = queue[0]
            del queue[0]
            return first

        def get_neighbors(offset):
            neighs = []
            for i in range(self.dimension):
                neighbor = offset[:]
                neighbor[i] -= 1
                neighs.append(neighbor)
                neighbor = offset[:]
                neighbor[i] += 1
                neighs.append(neighbor)
            return neighs

        dimension = len(self.basis[0])
        queue = [[0]*self.dimension]
        offsets = [[0]*self.dimension]
        while len(queue) > 0:
            x = pop(queue)
            neighbors = get_neighbors(x)
            for n in neighbors:
                if n not in offsets:
                    if in_range(self.dimension, n, self.basis, self.points, self.rotation):
                        queue.append(n)
                        offsets.append(n)
        return offsets
    def _draw_edges(self, w, offsets):
        # determines the z coordinate of the midpoint of the edge in abstract rotated coordinate space
        def edge_depth(edge, offset):
            assert(edge.dimension == 3)

            # coordinates of first point with zero offset
            point1 = vadd(edge.points[0].coords, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge.offsets[0], self.basis)])
            # coordinates of first point with correct offset
            moved_point1 = vadd(point1, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, self.basis)])
            # coordinates of second point with zero offset
            point2 = vadd(edge.points[1].coords, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(edge.offsets[1], self.basis)])
            # coordinates of second point with correct offset
            moved_point2 = vadd(point2, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, self.basis)])
            # rotated coordinates of first point
            rotated_point1 = matrix_mult(self.rotation, moved_point1)
            # rotated coordinates of second point
            rotated_point2 = matrix_mult(self.rotation, moved_point2)
            return (rotated_point1[2] + rotated_point2[2])/2

        edgesoffsets = list(product(self.edges, offsets))
        # if the shape is 3d, we want to draw the far edges before the near edges (due to the fading color)
        if len(self.basis[0]) == 3:
            edgesoffsets.sort(key=lambda x: edge_depth(*x))
        for edge, offset in edgesoffsets:
            edge.draw(offset)
    def _draw_points(self, w, offsets):
        # determines the z coordinate of the point in abstract rotated coordinate space
        def point_depth(point, offset):
            assert(point.dimension == 3)

            moved_point_coords = vadd(point.coords, [scalar_mult(offset[k], self.basis[k]) for k in range(len(offset))])
            rotated_point_coords = matrix_mult(self.rotation, moved_point_coords)
            return rotated_point_coords[2]

        pointsoffsets = list(product(self.points, offsets))
        # if the shape is 3d, we want to draw the far edges before the near edges (due to the fading color)
        if len(self.basis[0]) == 3:
            pointsoffsets.sort(key=lambda x: point_depth(*x))
        '''for i in multi_range([-3] * dimension, [3] * dimension):'''
        for point, offset in pointsoffsets:
            point.draw(w, offset)
    def rotate_left(self):
        if self.dimension != 3: return
        delta = [[cos(tau/24),0,-sin(tau/24)],[0,1,0],[sin(tau/24),0,cos(tau/24)]]
        self.rotation = two_matrix_mult(delta, self.rotation)
    def rotate_right(self):
        if self.dimension != 3: return
        delta = [[cos(tau/24),0,sin(tau/24)],[0,1,0],[-sin(tau/24),0,cos(tau/24)]]
        self.rotation = two_matrix_mult(delta, self.rotation)
    def rotate_up(self):
        if self.dimension != 3: return
        delta = [[1,0,0],[0,cos(tau/24),-sin(tau/24)],[0,sin(tau/24),cos(tau/24)]]
        self.rotation = two_matrix_mult(delta, self.rotation)
    def rotate_down(self):
        if self.dimension != 3: return
        delta = [[1,0,0],[0,cos(tau/24),sin(tau/24)],[0,-sin(tau/24),cos(tau/24)]]
        self.rotation = two_matrix_mult(delta, self.rotation)
    def rotate_cw(self):
        if self.dimension == 3:
            delta = [[cos(tau/24),-sin(tau/24),0],[sin(tau/24),cos(tau/24),0],[0,0,1]]
            self.rotation = two_matrix_mult(delta, self.rotation)
        elif self.dimension == 2:
            delta = [[cos(tau/24),-sin(tau/24)],[sin(tau/24),cos(tau/24)]]
            self.rotation = two_matrix_mult(delta, self.rotation)
    def rotate_ccw(self):
        if self.dimension == 3:
            delta = [[cos(tau/24),sin(tau/24),0],[-sin(tau/24),cos(tau/24),0],[0,0,1]]
            self.rotation = two_matrix_mult(delta, self.rotation)
        elif self.dimension == 2:
            delta = [[cos(tau/24),sin(tau/24)],[-sin(tau/24),cos(tau/24)]]
            self.rotation = two_matrix_mult(delta, self.rotation)
    # s = scaling factor
    def scale(self, s):
        tfm = identity(self.dimension)
        tfm = [scalar_mult(s, v) for v in tfm]
        self.rotation = two_matrix_mult(tfm, self.rotation)
    def reset_position(self):
        self.rotation = identity(self.dimension)

# class that represents a vertex of the polytope
# note: currently a single instance represents all vertices of the polytope equivalent under translation symmetry.
# we're going to build the points first and then the edges, so we want an interface to add edges incident to the point
class Point:
    def __init__(self, poly, coords):
        self.coords = coords # coordinates of the vertex with offset [0, 0, ...]
        self.dimension = len(coords)
        self.edges = [] # the Edge objects the vertex is associated with. each Edge has an offset relative to the point (as of this writing I don't think there's a reason the Points need to reference the Edges but I suspect it will be useful for polytope computation in the future)
        # Update: For now let's try to not specify the offsets of edges explicitly here, but rather just specify the offset of a point from each edge.
        self.polytope = poly # the Polytope object this vertex belongs to
    def draw(self, w, offset):
        moved_point_coords = vadd(self.coords, [scalar_mult(offset[k], self.polytope.basis[k]) for k in range(len(offset))])
        rotated_point_coords = matrix_mult(self.polytope.rotation, moved_point_coords)
        direct_draw_point(w, rotated_point_coords)
    def add_edge(self, edge):
        if edge not in self.edges:
            self.edges.append(edge)

class Edge:
    def __init__(self, poly, points, offsets):
        self.dimension = len(offsets[0])
        self.points = points # a list of two Point objects representing the endpoints
        self.offsets = offsets # the pair of offsets assigned to each point relative to the actual Point object
        self.polytope = poly # the Polytope object this edge belongs to
        # don't forget to add a reference to the edge from the endpoints
        for point in self.points:
            point.add_edge(self)
    def draw(self, offset):
        # coordinates of first point with zero offset
        point1 = vadd(self.points[0].coords, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(self.offsets[0], self.polytope.basis)])
        # coordinates of first point with correct offset
        moved_point1 = vadd(point1, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, self.polytope.basis)])
        # coordinates of second point with zero offset
        point2 = vadd(self.points[1].coords, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(self.offsets[1], self.polytope.basis)])
        # coordinates of second point with correct offset
        moved_point2 = vadd(point2, [scalar_mult(mult, basis_vect) for mult, basis_vect in zip(offset, self.polytope.basis)])
        # rotated coordinates of first point
        rotated_point1 = matrix_mult(self.polytope.rotation, moved_point1)
        # rotated coordinates of second point
        rotated_point2 = matrix_mult(self.polytope.rotation, moved_point2)
        direct_draw_line(w, rotated_point1, rotated_point2)

# Builds a Polytope object from a data file
def build(path):
    mode = 0
    dimension = -1
    basis = [] # the basis vectors describing the symmetry of the infinite repeating pattern
    points = [] # the points in each unit of translational symmetry
    edges = [] # the edges in each unit of translational symmetry

    # read the file
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
            if mode == 0: # parse dimension
                dimension = int(line)
                if dimension < 0:
                    raise RuntimeError("Error reading data: Dimension out of range.")
            if mode == 1: # parse basis vectors
                vstr = line.split(',')
                if len(vstr) != dimension:
                    raise RuntimeError("Error in \'{}\': number of coordinates is different from dimension.".format(line))
                v = [float(eval(s)) for s in vstr]
                basis.append(v)
                if len(basis) > dimension:
                    raise RuntimeError("Basis length is greater than dimension.")
            if mode == 2: # parse point data and create Point objects
                pstr = line.split(',')
                if len(pstr) != dimension:
                    raise RuntimeError("Error in \'{}\': number of coordinates is different from dimension.".format(line))
                p = [float(eval(s)) for s in pstr]
                points.append(Point(None, p))
            if mode == 3: # parse edge data and create Edge objects
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
                edges.append(Edge(None, [points[i] for (i, offset) in l], [offset for (i, offset) in l]))

    data.close()

    poly = Polytope(dimension=dimension, basis=basis, points=points, edges=edges)
    return poly

def assign_buttons_functions(left_button, right_button, up_button, down_button, cw_button, ccw_button, reset_button, scale_by_2_btn, scale_by_half_btn):
    def left_button_fn():
        poly.rotate_left()
        poly.draw(w)

    def right_button_fn():
        poly.rotate_right()
        poly.draw(w)

    def up_button_fn():
        poly.rotate_up()
        poly.draw(w)

    def down_button_fn():
        poly.rotate_down()
        poly.draw(w)

    def cw_button_fn():
        poly.rotate_cw()
        poly.draw(w)

    def ccw_button_fn():
        poly.rotate_ccw()
        poly.draw(w)

    def reset_position():
        poly.reset_position()
        poly.draw(w)

    def scale_by_2():
        poly.scale(2)
        poly.draw(w)

    def scale_by_half():
        poly.scale(1/2)
        poly.draw(w)

    left_button.configure(command=left_button_fn)
    right_button.configure(command=right_button_fn)
    up_button.configure(command=up_button_fn)
    down_button.configure(command=down_button_fn)
    cw_button.configure(command=cw_button_fn)
    ccw_button.configure(command=ccw_button_fn)
    reset_button.configure(command=reset_position)
    scale_by_2_btn.configure(command=scale_by_2)
    scale_by_half_btn.configure(command=scale_by_half)

if __name__ == "__main__":
    # open up tkinter canvas
    master = Tk()
    master.title("Erictope")
    w = Canvas(master, width=500, height=500)
    w.pack()

    global poly # the polytope currently being rendered

    # add buttons and indicator of dimension of space (different from rank)
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
    scale_by_2_btn = Button(master, text = "Zoom in", fg = "Black", bg = "Gray")
    scale_by_2_btn.pack()
    scale_by_half_btn = Button(master, text = "Zoom out", fg = "Black", bg = "Gray")
    scale_by_half_btn.pack()
    dim_display = Text(master, height = 1, width=10)
    dim_display.pack()

    print("Welcome to Erictope!")
    path = input("Enter file to read coordinates from: ")
    poly = build(path)

    poly.draw(w)

    assign_buttons_functions(left_button, right_button, up_button, down_button, cw_button, ccw_button, reset_button, scale_by_2_btn, scale_by_half_btn)
    dim_display.insert(INSERT, str(poly.dimension) + 'D')

    mainloop()
