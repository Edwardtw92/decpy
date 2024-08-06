from typing import Callable
from typing import Type

import numpy as np
from numpy.linalg import norm

from scipy.sparse import csr_matrix, csc_matrix, diags, save_npz, load_npz, find
from scipy.sparse.linalg import inv
from scipy.integrate import quad, dblquad
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
#import pydec
#from pydec import SimplicialMesh, write_mesh, read_mesh#, combinations, permutations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patheffects as pe
import matplotlib.cm as cm
from matplotlib.pylab import triplot, tricontourf, show
from matplotlib.patches import Polygon, RegularPolygon
from matplotlib.collections import PatchCollection
cm = 1/2.54 #centimetres in inches

# https://stackoverflow.com/questions/65426069/use-of-mathbb-in-matplotlib
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# from mpl_toolkits.axes_grid1 import make_axes_locatable

# matplotlib colorbar label
# https://stackoverflow.com/questions/33737427/top-label-for-matplotlib-colorbars

from pathlib import Path

# https://github.com/matplotlib/matplotlib/issues/11155
# https://stackoverflow.com/questions/52303660/iterating-markers-in-plots/52303895#52303895
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

# https://stackoverflow.com/questions/44025403/how-to-use-matplotlib-path-to-draw-polygon
def convexhull(p):
    p = np.array(p)
    hull = ConvexHull(p)
    return p[hull.vertices,:]

# https://stackoverflow.com/questions/44025403/how-to-use-matplotlib-path-to-draw-polygon
def ccw_sort(p):
    p = np.array(p)
    mean = np.mean(p, axis=0)
    d = p - mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s),:]

"""
Questions:
    Why do we care about the orientation of edges and the faces
    How can I implement the orientation of edges and the faces
    When to use objects
    Functions as cochain
    Vector value functions
    What is the physical meaning of the orientation of edges and triangles
    Is storing matrtixes as txt file a bad practis?
    Why is boundary operator called boundary operator
    what is the benefit of dec
    Why is the coboundary operator happen to be the exterior derivative operator
"""

"""
Ideas:
    Create a new class to import meshes and take in the name of the class in order
        to name .txt files 
    Potential converting python codes into julia
"""

"""
Todo:
    Type hints and docstring
    documentation
    Add heat map to 1-form and 2-form visualisation
"""

"""
Acknowledgement:
    Helped by my supervisor Erik Schnetter, PSI Master's student Jonathan Gouws,
    and my friend Karanveer B.
"""

def initiate_trimesh(name: str, length: float, side: int, action: str):
    return TriMesh2D(name, length, side, get_trimesh_centre(length), action)

def get_tri_centre(s) -> tuple:
    r = ((s - 1) / 3) * np.sqrt(3)
    return (-r * np.cos(np.pi/6), -r * np.sin(np.pi/6))

def get_trimesh_centre(length: float) -> tuple:
    r = (length / 3) * np.sqrt(3)
    return (-r * np.cos(np.pi/6), -r * np.sin(np.pi/6))

def normalise(vec: np.ndarray) -> float:
    # https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
    n = norm(vec)
    if n == 0:
        return vec
    return vec / n

def normalise_to_length(vec: np.ndarray, length: float) -> float:
    n = norm(vec)
    return None

def avg(x, y):
    return (x + y) / 2

def txt_to_npz(dname: str, fname: str) -> None:
    """
    If the file is in a subdircetory "mesh" then dname = "mesh"
    If the input file name is "matrix.txt" then fname = "matrix"
    """
    M_txt = np.loadtxt(f"{dname}/{fname}.txt")
    M = csr_matrix(M_txt) 
    save_npz(f"{dname}/{fname}.npz", M)

def npz_to_txt(dname: str, fname: str) -> None:
    """
    If the file is in a subdirctory "mesh" then dname = "mesh"
    If the input file name is "matrix.npz" then fname = "matrtix"
    """
    M_npz = load_npz(f"{dname}/{fname}.npz")
    M = M_npz.toarray()
    np.savetxt(f"{dname}/{fname}.txt", M, fmt='%.d')

class TriMesh2D:
    """
    Fields:
        side (int)
        coord (tuple of int)

        n_vertices (int)
        n_edges (int)
        n_triangles (int)
    """
    def __init__(self, name: str, length: float, side: int, coord: tuple, action: str) -> None:
        """
        Constructor: Create

        Effects: Mutates Self
        __init__: TriMesh2D Int Tuple -> None
        """
        self.name = name
        self.length = length
        self.side = side
        self.coord = coord
        self.edge_length = self.length/(self.side - 1)
        
        path = f"{name}/{name}"
        self.path = path

        if action == "w":
            Path(f"{name}").mkdir(parents=True, exist_ok=True)
            self.tri_mesh()
            self.gen_generic_mesh()
        elif action != "r":
            Print("Error in __init__")
            return None
        else:
            # self.vertices = np.loadtxt(f"{name}_v.txt", ndmin=2, dtype=float)
            # self.edges = np.loadtxt(f"{name}_e.txt", ndmin=2, dtype="int")
            # self.triangles = np.loadtxt(f"{name}_s.txt", ndmin=2, dtype = "int32")
            # self.e_to_s = np.loadtxt(f"{name}_e_to_s.txt", ndmin=2, dtype="int")
            # self.vertices = np.loadtxt(f"{name}/{name}_v.txt")
            self.vertices = np.loadtxt(f"{path}_v.txt")
            self.edges = np.loadtxt(f"{path}_e.txt")
            self.triangles = np.loadtxt(f"{path}_s.txt")

            x = self.vertices[:, 0]
            y = self.vertices[:, 1]
            self.triang = mtri.Triangulation(x, y, triangles)

            self.e_to_s = np.loadtxt(f"{path}_e_to_s.txt")
            self.vc = np.loadtxt(f"{path}_vc.txt")
            self.ve = np.loadtxt(f"{path}_ve.txt")
            self.vi = np.loadtxt(f"{path}_vi.txt")
            self.ee = np.loadtxt(f"{path}_ee.txt")
            self.ei = np.loadtxt(f"{path}_ei.txt")

        # if Path(f"{name}_D0.npz").is_file():
        # if Path(f"{name}/{name}._D0.npz").is_file():
        if Path(f"{path}_D0.npz").is_file():
            # self.D0 = load_npz(f"{name}_D0.npz")
            # self.D0 = load_npz(f"{name}/{name}_D0.npz")
            self.D0 = load_npz(f"{path}_D0.npz")
            self.B1 = self.D0.transpose()
            #self.D0 = np.loadtxt(f"{name}_D0.txt", ndmin=2, dtype="int")
        else:
            print(f'Please create "{path}_D0.txt"')
        
        # if Path(f"{name}_D1.npz").is_file():
        # if Path(f"{name}/{name}_D1.npz").is_file():
        if Path(f"{path}_D1.npz").is_file():
            # self.D1 = load_npz(f"{name}_D1.npz")
            # self.D1 = load_npz(f"{name}/{name}_D1.npz")
            self.D1 = load_npz(f"{path}_D1.npz")
            self.B2 = self.D1.transpose()
            #self.D1 = np.loadtxt(f"{name}_D1.txt", ndmin=2, dtype="int")
        else:
            # print(f'Please create "{name}_D1.txt"')
            print(f'Please create "{path}_D1.txt"')

    def __repr__(self):
        """
        Returns a string representation of self

        __repr__: TriMesh2D -> Str
        """
        return f"A equilateral triangular mesh of base length {self.length} and {self.side} vertices per side:\n" \
            f"Number of Vertices: {self.get_num_vertices()}\n" \
            f"Number of Edges: {self.get_num_edges()}\n" \
            f"Number of Triangles: {self.get_num_triangles()}\n" \
    
    def check_D0(self) -> bool:
        # if Path(f"{self.name}_D0.npz").is_file():
        path = self.path
        if Path(f"{path}_D0.npz").is_file():
            return True
        elif Path(f"{path}_D0.txt").is_file():
            txt_to_npz(self.name, f"{self.name}_D0")
            return True
        else:
            print(f'Please create "{path}_D0.txt" or "{path}_D0.npz"')
            return False

    def check_D1(self) -> bool:
        # if Path(f"{self.name}_D1.npz").is_file():
        path = self.path
        if Path(f"{path}_D1.npz").is_file():
            return True
        elif Path(f"{path}_D1.txt").is_file():
            txt_to_npz(self.name, f"{self.name}_D1")
            return True
        else:
            print(f'Please create "{path}_D1.txt" or "{path}_D1.npz"')
            return False

    def get_num_vertices(self) -> int:
        return ((1 + self.side) * self.side) // 2
    
    def get_num_edges(self) -> int:
        return ((1 + self.side - 1) * (self.side - 1) // 2) * 3

    def get_num_triangles(self) -> int:
        num_tri = 0
        for i in range(1, self.side):
            num_tri += 2 * i - 1
        return num_tri

    def get_resolution(self) -> float:
        resolution = self.length/(self.side - 1)
        return resolution

    def gen_unsigned_D0(self) -> None: # 0 implies from 0 to 1 and applies on 0-form
        edges = self.edges
        num_vertices = self.get_num_vertices()
        num_edges = self.get_num_edges()
 
        row = np.repeat(np.arange(0, num_edges, dtype=int), 2)
        col = edges.ravel()        
        data = np.ones(num_edges * 2, int)
        
        unsigned_D0 = csr_matrix((data, (row, col)), shape=(num_edges, num_vertices))
        self.unsigned_D0 = unsigned_D0
        save_npz(f"{self.path}_unsigned_D0.npz", unsigned_D0)
        return None

    def gen_unsigned_D1(self) -> None: # 1 implies from 1 to 2 and applies on 1-form
        edges = self.edges
        edges_to_triangles = self.e_to_s
        num_edges = self.get_num_edges()
        num_triangles = self.get_num_triangles()

        row = np.repeat(np.arange(0, num_triangles, dtype=int), 3)
        col = edges_to_triangles.ravel()
        data = np.ones(num_triangles * 3, int)

        unsigned_D1 = csr_matrix((data, (row, col)), shape=(num_triangles, num_edges))
        self.unsigned_D1 = unsigned_D1
        save_npz(f"{self.path}_unsigned_D1.npz", unsigned_D1)
        return None
    
    def get_D0(self) -> np.ndarray:
        """
        Please manually make a copy of unsigned_D0.txt and name it as D0.txt
        """
        if not(self.check_D0()):
            return None
        D0 = self.D0
        #D0 = np.loadtxt("D0.txt", dtype="int")
        #self.D0 = D0
        return D0

    def get_D1(self) -> np.ndarray:
        """
        Please manually make a copy of unsigned_D1.txt and name it as D1.txt
        """
        if not(self.check_D1()):
            return None
        D1 = self.D1
        #D0 = np.loadtxt("D1.txt", dtype="int")
        #self.D1 = D1
        return D1
        
    def zero_form(self, f: Callable | np.ndarray) -> np.ndarray:
        """
        f: R^2 -> R or an 1D np.ndarray
        """
        if type(f) == np.ndarray:
            omega = f
            return omega
        vertices = self.vertices
        omega = f(vertices[:, 0], vertices[:, 1])
        return omega
    
    def one_form_precise(self, F:np.ndarray) -> np.ndarray:
        """
        F: R^2 -> R^2 or an 1D np.ndarray
        """
        vertices = self.vertices
        edges = self.edges
        edge_length = self.length/(self.side - 1)
        if not(self.check_D0()):
            return None
        D0 = self.D0
        omega =  np.empty(self.get_num_edges(), dtype=float)

        omega_index = 0
        for i in range(0, self.get_num_edges()):
            edge_indicies = edges[i]
            if D0[i, edges[i][0]] == -1:
                initial_point = vertices[edges[i][0]]
                final_point = vertices[edges[i][1]]
            else:
                initial_point = vertices[edges[i][1]]
                final_point = vertices[edges[i][0]]

            L = norm(final_point - initial_point)
            T = (final_point - initial_point)/L
            g = lambda t: initial_point + (t * T)
            func = lambda t: np.dot(F(g(t)[0], g(t)[1]), T)
            line_integral = quad(func, 0, L)
            omega[omega_index] = line_integral[0]
            omega_index += 1
        return omega

    def one_form(self, F: Callable | np.ndarray) -> np.ndarray:
        """
        F: R^2 -> R^2 or an 1D np.ndarray
        """
        vertices = self.vertices
        edges = self.edges
        edge_length = self.length/(self.side - 1)
        if not(self.check_D0()):
            return None
        D0 = self.D0
        omega =  np.empty(self.get_num_edges(), dtype=float)

        if type(F) == np.ndarray:
            omega_index = 0
            for i in range(0, self.get_num_edges()):
                edge_indicies = edges[i]
                if D0[i, edges[i][0]] == -1:
                    initial_point = vertices[edges[i][0]]
                    final_point = vertices[edges[i][1]]
                else:
                    initial_point = vertices[edges[i][1]]
                    final_point = vertices[edges[i][0]]
                
                L = norm(final_point - initial_point)
                # T = (final_point - initial_point)/L
                T = (final_point - initial_point)
                line_integral = np.dot(F, T)
                omega[omega_index] = line_integral
                omega_index += 1
            return omega

        omega_index = 0
        for i in range(0, self.get_num_edges()):
            edge_indicies = edges[i]
            if D0[i, edges[i][0]] == -1:
                initial_point = vertices[edges[i][0]]
                final_point = vertices[edges[i][1]]
            else:
                initial_point = vertices[edges[i][1]]
                final_point = vertices[edges[i][0]]

            L = norm(final_point - initial_point)
            #T = (final_point - initial_point)/L
            T = (final_point - initial_point)
            x, y = (initial_point + final_point)/2
            line_integral = np.dot(F(x,y), T)
            omega[omega_index] = line_integral
            omega_index += 1
        return omega

    def two_form(self, f: Callable | np.ndarray):
        """
        f: R^2 -> R or an 1D np.ndarray
        """
        if not(self.check_D0()):
            return None
        D0 = self.D0
        if not(self.check_D1()):
            return None
        D1 = self.D1
        edge_length = self.length/(self.side - 1)
        vertices = self.vertices
        triangles = self.triangles
        e_to_s = self.e_to_s
        norm_direct = np.array([normalise(row) for row in (D0 @ vertices)])
        omega = np.empty(self.get_num_triangles(), dtype=float)

        def edge_sgn(v: np.ndarray) -> int:
            if v[0] > 0:
                return 1
            else:
                return -1

        if type(f) == np.ndarray:
            omega_index = 0
            for i in range(0, self.get_num_triangles()):
                a = edge_sgn(norm_direct[e_to_s[i][0]]) * D1[i, e_to_s[i][0]] * (edge_length/2)
                if a > 0:
                    sgn = 1
                elif a < 0:
                    sgn = -1
                
                area = edge_length**2 * np.sqrt(3)/4
                surface_integral = sgn * f * area
                omega[omega_index] = surface_integral
                omega_index += 1
            return omega

        omega_index = 0
        for i in range(0, self.get_num_triangles()):
            #T1 = np.array([1, 0])
            #T2 = np.array([0, 1])
            #func = np.dot(T1, (np.matmul(M, T2))) - np.dot(T2, (np.matmul(M, T1)))
            # a = edge_sgn(norm_direct[e_to_s[i][0]]) * D1[i][e_to_s[i][0]] * (edge_length/2)
            a = edge_sgn(norm_direct[e_to_s[i][0]]) * D1[i, e_to_s[i][0]] * (edge_length/2)
            if a > 0:
                sgn = 1
            elif a < 0:
                sgn = -1
            
            area = edge_length**2 * np.sqrt(3)/4
            x, y = (vertices[triangles[i][0]] + vertices[triangles[i][1]] + vertices[triangles[i][2]])/3
            surface_integral = sgn * f(x,y) * area
            #omega = np.append(omega, surface_integral)
            omega[omega_index] = surface_integral
            omega_index += 1
        return omega

    # https://youtu.be/-cUhuzwW-_A?si=YPAQI9iZDwkWYLxf
    def get_hodge0(self) -> np.ndarray:
        num_vertices = self.get_num_vertices()
        edge_length = self.edge_length
        vc = self.vc
        ve = self.ve
        
        volume_ratio = (np.sqrt(3)/2) * (edge_length**2)
        diagonals = np.empty(num_vertices)
        for vertex in np.arange(num_vertices):
            if vertex in vc:
                diagonals[vertex] = (1/6) * volume_ratio
            elif vertex in ve:
                diagonals[vertex] = (1/2) * volume_ratio
            else:
                diagonals[vertex] = volume_ratio
        hodge0 = diags(diagonals, format='csc')
        return hodge0

    # https://youtu.be/-cUhuzwW-_A?si=YPAQI9iZDwkWYLxf
    def get_hodge1(self) -> np.ndarray:
        num_edges = self.get_num_edges()
        ee = self.ee

        volume_ratio = (1/np.sqrt(3))
        diagonals = np.empty(num_edges)
        for edge in np.arange(num_edges):
            if edge in ee:
                diagonals[edge] = (1/2) * volume_ratio
            else:
                diagonals[edge] = volume_ratio
        hodge1 = diags(diagonals, format='csc')
        return hodge1

    # https://youtu.be/-cUhuzwW-_A?si=YPAQI9iZDwkWYLxf
    def get_hodge2(self) -> np.ndarray:
        num_triangles = self.get_num_triangles()
        edge_length = self.edge_length
        
        diagonals = np.empty(num_triangles)
        volume_ratio = 1/((np.sqrt(3)/4) * (edge_length**2))
        for triangle in np.arange(num_triangles):
            diagonals[triangle] = volume_ratio
        hodge2 = diags(diagonals, format='csc')
        return hodge2

    def dual_zero_form(self, omega: np.ndarray) -> np.ndarray:
        return self.hodge0 @ omega

    def dual_one_form(self, omega: np.ndarray) -> np.ndarray:
        return self.hodge1 @ omega
    
    def dual_two_form(self, omega: np.ndarray) -> np.ndarray:
        return self.hodge2 @ omega
    
    def inverse_dual_zero_form(self, omega: np.ndarray) -> np.ndarray:
        return self.inverse_hodge0 @ omega

    def inverse_dual_one_form(self, omega: np.ndarray) -> np.ndarray:
        return self.inverse_hodge1 @ omega

    def inverse_dual_two_form(self, omega: np.ndarray) -> np.ndarray:
        return self.inverse_hodge2 @ omega

    def hodge_star(self, omega: np.ndarray) -> np.ndarray:
        num_vertices = self.get_num_vertices()
        num_edges = self.get_num_edges()
        num_triangles = self.get_num_triangles()

        form_size = len(omega)
        if form_size == num_vertices:
            return self.dual_zero_form(omega)
        elif form_size == num_edges:
            return self.dual_one_form(omega)
        else:
            return self.dual_two_form(omega)
    
    def dual_hodge_star(self, omega: np.ndarray) -> np.ndarray:
        num_vertices = self.get_num_vertices()
        num_edges = self.get_num_edges()
        num_triangles = self.get_num_triangles()

        form_size = len(omega)
        if form_size == num_vertices:
            return self.inverse_dual_zero_form(omega)
        elif form_size == num_edges:
            return self.inverse_dual_one_form(omega)
        else:
            return self.inverse_dual_two_form(omega)
    
    def get_hodge0_dual(self) -> np.ndarray:
        num_vertices = self.get_num_vertices()
        edge_length = self.edge_length
        vc = self.vc
        ve = self.ve
        
        volume_ratio = 1/((np.sqrt(3)/2) * (edge_length**2))
        diagonals = np.empty(num_vertices)
        for vertex in np.arange(num_vertices):
            if vertex in vc:
                diagonals[vertex] = 6 * volume_ratio
            elif vertex in ve:
                diagonals[vertex] = 2 * volume_ratio
            else:
                diagonals[vertex] = volume_ratio
        hodge0_dual = diags(diagonals, format='csc')
        return hodge0_dual

    def get_hodge1_dual(self) -> np.ndarray:
        num_edges = self.get_num_edges()
        ee = self.ee

        volume_ratio = (np.sqrt(3))
        diagonals = np.empty(num_edges)
        for edge in np.arange(num_edges):
            if edge in ee:
                diagonals[edge] = 2 * volume_ratio
            else:
                diagonals[edge] = volume_ratio
        hodge1_dual = diags(diagonals, format='csc')
        return hodge1_dual
    
    def get_hodge2_dual(self) -> np.ndarray:
        num_triangles = self.get_num_triangles()
        edge_length = self.edge_length
        
        diagonals = np.empty(num_triangles)
        volume_ratio = ((np.sqrt(3)/4) * (edge_length**2))
        for triangle in np.arange(num_triangles):
            diagonals[triangle] = volume_ratio
        hodge2_dual = diags(diagonals, format='csc')
        return hodge2_dual

    def d0(self, omega: np.ndarray) -> np.ndarray:
        """
        Please manually make a copy of unsigned_D0.txt and name it as D0.txt or run self.gen_generic_mesh()
        """
        if not(self.check_D0()):
            return None
        D0 = self.D0
        #D0 = self.get_D0()
        D0_o_omega = (D0 @ omega)
        return D0_o_omega
    
    def d1(self, omega: np.ndarray) -> np.ndarray:
        """
        Please Manually make a copy of unsigned_D1.txt and name it as D1.txt or run self.gen_generic_mesh()
        """
        if not(self.check_D1()):
            return None
        D1 = self.D1
        D1_o_omega = (D1 @ omega)
        return D1_o_omega

    def get_circumcentres(self) -> np.ndarray:
        v = self.vertices
        circumcentres = np.empty([self.get_num_triangles(), 2], float)
        inc = self.length/(self.side - 1)

        r = inc * (np.sqrt(3)/3)

        side = self.side
        sum_side = side
        circumcentres_index = 0
        for j in range(0, self.get_num_vertices() - 2):
            if j == sum_side - 1:
                side -= 1
                sum_side += side 
            elif j == sum_side - 2:
                new_row = v[j] + np.array([r * np.cos(np.pi/6), r * np.sin(np.pi/6)])
                circumcentres[circumcentres_index] = new_row
                circumcentres_index += 1
            else:
                new_row = v[j] + np.array([r * np.cos(np.pi/6), r * np.sin(np.pi/6)])
                circumcentres[circumcentres_index] = new_row
                new_row = v[j + 1] + np.array([0, r])
                circumcentres[circumcentres_index + 1] = new_row
                circumcentres_index += 2
        return circumcentres

    def neighbour_triangles(self, vertex: int) -> np.ndarray:
        if vertex in self.vc:
            neighbour_triangles = np.empty(1, dtype=int)
        elif vertex in self.ve:
            neighbour_triangles = np.empty(3, dtype=int)
        else:
            neighbour_triangles = np.empty(6, dtype=int)

        i = 0
        for tri_index, triangle in enumerate(self.triangles):
            if vertex in triangle:
                neighbour_triangles[i] = tri_index
                i += 1
        return neighbour_triangles

    def edge_of_triangles(self, edge: int) -> np.ndarray:
        if edge in self.ee:
            tri_index = np.empty(1, dtype=int)
        else:
            tri_index = np.empty(2, dtype=int)
        
        i = 0
        for j, triangle in enumerate (self.e_to_s):
            if edge in triangle:
                tri_index[i] = j
                i += 1
        return  tri_index

    def tri_finder(self, edge_1: int, edge_2: int) -> int:
        for tri_index, triangle in enumerate(self.e_to_s):
            if (edge_1 in triangle) and (edge_2 in triangle):
                return tri_index

    def edges_finder(self, vertex: int, triangle: int) -> np.ndarray:
        final_edge = np.empty(2, dtype=int)
        final_edge_index = 0
        potential_edges = self.e_to_s[triangle]
        for edge in potential_edges:
            if vertex in self.edges[edge_index]:
                final_edge[final_edge_index] = edge_index
                final_edge_index += 1
        return final_edge

    def connecting_edges(self, vertex: int) -> np.ndarray:
        if vertex in self.vc:
            connecting_edges = np.empty(2, dtype=int)
        elif vertex in self.ve:
            connecting_edges = np.empty(4, dtype=int)
        else:
            connecting_edges = np.empty(6, dtype=int)
        
        i = 0
        for edge in self.edges:
            if vertex in edge:
                connecting_edges[i] = edge
                i += 1
        return connecting_edges

    def edges_finder(v_index: int, tri_index: int) -> np.ndarray:
        final_edges = np.empty(2, dtype=int)
        final_edges_index = 0
        potential_edges = self.e_to_s[tri_index]
        for e_index in potential_edges:
            if v_index in potential_edges[e_index]:
                final_edges[final_edges_index] = e_index
                final_edges_index += 1
        return final_edges

    def pre_tri_mesh(self) -> None:
        """
        Effects:
        Writes two file "v.txt" and "s.txt"

        Requires: sides >= 2

        Notes:
        All triangles are positively oriented
        Edges have no specified orientation but is assumed to be what is shown in the documentation
        """
        name = self.name
        path = self.path
        length = self.length
        side = self.side
        coord = self.coord
        inc = length/(side - 1)

        num_vertices = self.get_num_vertices()
        num_edges = self.get_num_edges()
        num_triangles = self.get_num_triangles()

        v = np.empty([num_vertices,2], dtype=float)
        e = np.empty([num_edges,2], dtype=int)
        e_to_s = np.empty([num_triangles,3], dtype=int)
        s = np.empty([num_triangles,3], dtype=int)

        v_index = 0
        row_index = 0
        for i in range(side, 0, -1):
            for j in range(0, i):
                v[v_index] = [coord[0] + (inc * 0.5 * (side - i)) + j * inc, coord[1] + inc * (np.sqrt(3/4) * (side - i))]
                v_index += 1
            row_index += 1
        
        self.vertices = v
        np.savetxt(f"{self.path}_v.txt", v)
 
        side = self.side
        sum_side = side
        e_index = 0
        for j in range(0, num_vertices - 2):
            if j == sum_side - 1:
                side -= 1
                sum_side += side
            else:
                e[e_index] = [j, j + 1]
                e[e_index+1] = [j + 1, j + side]
                e[e_index+2] = [j, j + side]
                e_index += 3
        self.edges = e
        # np.savetxt(f"{name}_e.txt", e, fmt='%.d')
        np.savetxt(f"{path}_e.txt", e, fmt='%.d')

        side = self.side
        sum_side = side
        index = 0
        edge_count = 0
        # for j in range(0, self.get_num_vertices() - 2):
        for j in range(0, num_vertices - 2):
            if j == sum_side - 1:
                side -= 1
                sum_side += side
            elif j == sum_side - 2:
                e_to_s[index] = [edge_count, edge_count+1, edge_count+2]
                s[index] = [j, j + 1, j + side]
                edge_count += 3
                index += 1
            else:
                e_to_s[index] = [edge_count, edge_count + 1, edge_count + 2]
                e_to_s[index + 1] = [edge_count + 1, edge_count + 2 + 3, edge_count + 3 * (side-1)]
                s[index] = [j, j + 1, j + side]
                s[index + 1] = [j + 1, j + 1 + side, j + side]
                edge_count += 3
                index += 2
        self. e_to_s = e_to_s
        self.triangles = s
        # np.savetxt(f"{name}_e_to_s.txt", e_to_s, fmt='%.d')
        np.savetxt(f"{path}_e_to_s.txt", e_to_s, fmt='%.d')
        np.savetxt(f"{path}_s.txt", s, fmt='%.d')

        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        self.triang = mtri.Triangulation(x, y, self.triangles)
        return None

    def tri_mesh(self):
        self.pre_tri_mesh()
        self.gen_unsigned_D0()
        self.gen_unsigned_D1()
        return None

    def triangulation(self):
        #self.tri_mesh()
        vertices = self.vertices
        #vertices = np.loadtxt("v.txt")
        triangles = self.triangles
        #triangles = np.loadtxt(f"{self.name}_s.txt", ndmin = 2, dtype = 'int32')
        """
        mymesh = SimplicialMesh(vertices = vertices, indices = triangles)
        write_mesh("square_8.xml", mymesh, format = 'basic')

        x = mymesh.vertices[:, 0] # x-coordinates
        y = mymesh.vertices[:, 1] # y-coordinates
        triangles = mymesh.indices # indicies of triangles
        """
        
        x = vertices[:, 0]
        y = vertices[:, 1]

        triang = mtri.Triangulation(x, y, triangles)
        self.triang = triang
        return triang

    def primal_mesh(self, ax) -> None:
        #print(self)
        #self.tri_mesh()

        triang = self.triangulation()

        #fig, ax = plt.subplots(figsize = (20*cm, 20*cm))
        ax.set_aspect('equal')
        ax.use_sticky_edges = False

        # ax.triplot(triang,'ro-',lw = 0.5, zorder=0, markersize=5)
        ax.triplot(triang,'r-',lw = 1.5, zorder=0, markersize=5)
        # ax.triplot(triang,'ro-',lw = 0.5, zorder=1, markersize=5)
        # ax.set_title(f"Equilateral triangular mesh with sides of length {self.side}")
        return None
    
    def pre_dual_mesh(self) -> None:
        B1 = self.B1
        Dn0 = B1 
        vc = np.empty(3, dtype=int) # vc as in v_corner
        vc_index = 0
        ve = np.empty((self.side - 2) * 3, dtype=int) # ve as in v_edge
        ve_index = 0
        vi = np.empty(self.get_num_vertices() - 3 - ((self.side - 2) * 3), dtype=int) # vi as in v_inside
        vi_index = 0
        for i in range(self.get_num_vertices()):
            # B1_size = B1[i].size
            B1_size = len(B1[i].data)
            if B1_size == 2:
                vc[vc_index] = i
                vc_index += 1
            elif B1_size == 4:
                ve[ve_index] = i
                ve_index += 1
            elif B1_size == 6:
                vi[vi_index] = i
                vi_index += 1
            else:
                print("error")
        self.vc = vc
        self.ve = ve
        self.vi = vi
        np.savetxt(f"{self.path}_vc.txt", vc)
        np.savetxt(f"{self.path}_ve.txt", ve)
        np.savetxt(f"{self.path}_vi.txt", vi)

        B2 = self.B2
        ee = np.empty((self.side - 1) * 3, dtype=int) # ee as in e_edge
        ee_index = 0
        ei = np.empty(self.get_num_edges() - ((self.side - 1) * 3), dtype=int) # ei as in e_inside
        ei_index = 0
        for i in range(self.get_num_edges()):
            B2_size = B2[i].size
            if B2_size == 1:
                ee[ee_index] = i
                ee_index += 1
            elif B2_size == 2:
                ei[ei_index] = i
                ei_index += 1
            else:
                print("error")
        self.ee = ee
        self.ei = ei
        np.savetxt(f"{self.path}_ee.txt", ee)
        np.savetxt(f"{self.path}_ei.txt", ei)
        return None

    def dual_mesh(self, ax, scale=0.001, linestyle='solid') -> None:
        if linestyle == 'loosely dotted':
            linestyle = (0, (5, 10))
        num_edges = self.get_num_edges()
        edge_midpoints = self.get_edge_midpoints()
        circumcentres = self.get_circumcentres()
        patches = np.empty(num_edges, dtype=list)
        D0 = self.D0
        v = self.vertices

        for i in range(num_edges):
            tri = self.edge_of_triangles(i)
            direct = (D0 @ v)[i]
            deviation = direct * scale
            if i in self.ee:
                line = Polygon([edge_midpoints[i] - deviation, circumcentres[tri[0]] - deviation, circumcentres[tri[0]] + deviation, edge_midpoints[i] + deviation])
            else:
                line = Polygon([circumcentres[tri[0]] - deviation, circumcentres[tri[1]] - deviation, circumcentres[tri[1]] + deviation, circumcentres[tri[0]] + deviation])
            patches[i] = line
        patch_collection = PatchCollection(patches, color='black', linestyle=linestyle)
        ax.add_collection(patch_collection)
    
    def dual_mesh_voronoi(self, ax) -> None:
        vor = Voronoi(self.vertices, incremental=True)
        voronoi_plot_2d(vor, ax)
        return None

    def plot_vertices_index(self, ax) -> None:
        pos = self.vertices
        markers = np.arange(self.get_num_vertices())
        for i in markers:
            ax.annotate(i, (pos[:, 0][i], pos[:, 1][i]), c='b', size=8, path_effects=[pe.withStroke(linewidth=4, foreground='w')])
        return None

    def plot_edges_index(self, ax) -> None:
        pos = self.get_edge_midpoints()
        markers = np.arange(self.get_num_edges())
        for i in markers:
            ax.annotate(i, (pos[:, 0][i], pos[:, 1][i]), c='g', size=8, path_effects=[pe.withStroke(linewidth=4, foreground='w')])
        return None

    def plot_triangles_index(self, ax) -> None:
        pos = self.get_circumcentres()
        markers = np.arange(self.get_num_triangles())
        for i in markers:
            ax.annotate(i, (pos[:, 0][i], pos[:, 1][i]), c='r', size=8, path_effects=[pe.withStroke(linewidth=4, foreground='w')])
        return None

    def init_plt_orientation(self, ax): 
        if not(self.check_D0()):
            return None
        v = self.vertices
        e = self.edges
        D0 = self.D0
        D1 = self.D1
        edge_length = (self.length)/(self.side - 1)
        e_to_s = self.e_to_s
        circumcentres = self.get_circumcentres()

        norm_direct = np.array([normalise(row) for row in (D0 @ v)])
        #old_direct = np.array([normalise(row) for row in np.matmul(D0, v)])
        direct = np.array([normalise(row) for row in (D0 @ v)])/2
        pos = np.array([avg(v[row[0]], v[row[1]]) for row in e])

        #ax.quiver(pos[:, 0], pos[:, 1], direct[:, 0], direct[:, 1], color='r', pivot='mid', scale=5, headlength=5, headaxislength=5, width=0.01)
        ax.quiver(pos[:, 0], pos[:, 1], direct[:, 0], direct[:, 1], color='r', pivot='mid')

        pos = circumcentres

        def edge_sgn(v: np.ndarray) -> int:
            if v[0] > 0:
                """
                if v[0] == 1 or v[1] < 0:
                    return 1
                else:
                    return -1
                """
                return 1
            #elif v[0] < 0:
                #return 1
            else:
                return -1

        direct = np.empty([0, 2], dtype=float)
        zero_pos_index = 0
        positive_pos_index = 0
        negative_pos_index = 0
        for i in range(0, self.get_num_triangles()):
            a = edge_sgn(norm_direct[e_to_s[i][0]]) * D1[i, e_to_s[i][0]] * (edge_length/2)
            if a > 0:
                #positive_pos = np.append(positive_pos, [pos[i]], axis=0)
                positive_pos_index += 1
            elif a < 0:
                #negative_pos = np.append(negative_pos, [pos[i]], axis=0)
                negative_pos_index += 1
            elif a == 0:
                #zero_pos = np.append(zero_pos, [pos[i]], axis = 0) 
                zero_pos_index += 1
            else:
                print("Error")
                return None


        direct = np.empty([0,2], dtype=float)
        zero_pos = np.empty([zero_pos_index,2], dtype="float")
        positive_pos = np.empty([positive_pos_index,2], dtype="float")
        negative_pos = np.empty([negative_pos_index,2], dtype="float")
        zero_pos_index = 0
        positive_pos_index = 0
        negative_pos_index = 0
        for i in range(0, self.get_num_triangles()):
            a = edge_sgn(norm_direct[e_to_s[i][0]]) * D1[i, e_to_s[i][0]] * (edge_length/2)
            if a > 0:
                #positive_pos = np.append(positive_pos, [pos[i]], axis=0)
                positive_pos[positive_pos_index] = pos[i]
                positive_pos_index += 1
            elif a < 0:
                #negative_pos = np.append(negative_pos, [pos[i]], axis=0)
                negative_pos[negative_pos_index] = pos[i]
                negative_pos_index += 1
            elif a == 0:
                #zero_pos = np.append(zero_pos, [pos[i]], axis = 0) 
                zero_pos[zero_pos_index] = pos[i]
                zero_pos_index += 1
            else:
                print("Error")
                return None
                
            #direct = np.append(direct, [[0,a]], axis=0)
        #ax.quiver(pos[:, 0], pos[:, 1], direct[:, 0], direct[:, 1], scale=3, color='r', pivot='mid', headlength=5, headaxislength=5, width=0.01)
        """
        ax.scatter(zero_pos[:, 0], zero_pos[:, 1], marker='$0$', color='c', s=200)
        ax.scatter(positive_pos[:, 0], positive_pos[:, 1], marker='o', color='r', s=200)
        ax.scatter(negative_pos[:, 0], negative_pos[:, 1], marker='X', color='b', s=200)
        """
        ax.scatter(zero_pos[:, 0], zero_pos[:, 1], marker='$0$', color='c', s=10)
        ax.scatter(positive_pos[:, 0], positive_pos[:, 1], marker='o', color='r', s=10)
        ax.scatter(negative_pos[:, 0], negative_pos[:, 1], marker='X', color='b', s=10)
        return None

    def zero_form_visualisation_heatmap(self, ax, omega: np.ndarray) -> None:
        triang = self.triangulation()
        ax.tricontourf(triang, omega, 20)
        #ax.triplot(triang, 'ro-', lw=0.5)
        ax.triplot(triang, 'ro-', lw=0.5, markersize=1)

        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        ax.margins(0.07)
        return None
    
    def  zero_form_visualisation_no_grid(self, ax, omega: np.ndarray) -> None:
        triang = self.triangulation()
        ax.tricontour(triang, omega, 20)

        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        ax.margins(0.07)
        return None
    
    def zero_form_visualisation(self, ax, omega: np.ndarray) -> None:
        vertices = self.vertices
        t = omega
        tri_plot = ax.scatter(vertices[:, 0], vertices[:, 1], c=t, cmap='viridis')
        colorbar = plt.colorbar(tri_plot)
        colorbar.ax.set(title='0-form')
        ax.set_aspect('equal')
        return None

    def get_edge_midpoints(self) -> np.ndarray:
        v = self.vertices
        e = self.edges
        midpoints = np.array([avg(v[row[0]], v[row[1]]) for row in e])
        return midpoints

    def one_form_visualisation_heatmap(self, ax, omega: np.ndarray) -> None:
        if not(self.check_D0()):
            return None
        v = self.vertices
        e = self.edges
        D0 = self.D0

        old_direct = (D0 @ v)
        #direct = normalise(old_direct * self.one_form(omega)[:, None])
        #direct = np.array([normalise(row) for row in old_direct * self.one_form(omega)[:, None]])/2
        direct = np.array([normalise(row) for row in old_direct * omega[:, None]])/2
        pos = self.get_edge_midpoints()
        #pos = np.array([avg(v[row[0]], v[row[1]]) for row in e])

        arrow_plot = ax.quiver(pos[:, 0], pos[:, 1], direct[:, 0], direct[:, 1], scale=5, pivot='mid')

        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        ax.margins(0.07)
        return None

    def one_form_visualisation(self, ax, omega: np.ndarray) -> None:
        if not(self.check_D0()):
            return None
        v = self.vertices
        e = self.edges
        D0 = self.D0

        old_direct = (D0 @ v)
        #direct = normalise(old_direct * self.one_form(omega)[:, None])
        #direct = np.array([normalise(row) for row in old_direct * self.one_form(omega)[:, None]])/2
        direct = np.array([normalise(row) for row in old_direct * omega[:, None]])/2
        pos = self.get_edge_midpoints()
        #pos = np.array([avg(v[row[0]], v[row[1]]) for row in e])

        colors = omega
        normed_old_direct = np.array([normalise(row) for row in old_direct])
        arrow_plot = ax.quiver(pos[:, 0], pos[:, 1], direct[:, 0], direct[:, 1], colors, scale=5, pivot='mid')
        colorbar =plt.colorbar(arrow_plot)
        colorbar.ax.set(title='1-form')

        ax.set_aspect('equal')
        # ax.use_sticky_edges = False
        # ax.margins(0.07)
        return None
    
    def two_form_visualisation(self, ax, omega:np.ndarray) -> None:
        if not(self.check_D1()):
            return None
        D1 = self.D1
        pos = self.get_circumcentres()

        tripcolor = ax.tripcolor(self.triang, omega, shading='flat', edgecolor='red', zorder=0)
        colorbar = plt.colorbar(tripcolor)
        colorbar.ax.set(title='2-form')
        ax.set_aspect('equal')
        return None

    def two_form_visualisation_dot(self, ax, omega:np.ndarray) -> None:
        """
        Note: only visualise the orientation not the magnitude
        """
        if not(self.check_D1()):
            return None
        D1 = self.D1
        pos = self.get_circumcentres()

        zero_pos = np.empty([0,2], dtype ="float")
        positive_pos = np.empty([0,2], dtype="float")
        negative_pos = np.empty([0,2], dtype="float")

        for i in range(0, len(omega)):
            if omega[i] == 0.0:
                zero_pos = np.append(zero_pos, [pos[i]], axis=0)
            elif omega[i] > 0:
                positive_pos = np.append(positive_pos, [pos[i]], axis=0)
            elif omega[i] < 0:
                negative_pos = np.append(negative_pos, [pos[i]], axis=0)
            else:
                print("Error")
                return None

        ax.scatter(zero_pos[:, 0], zero_pos[:, 1], marker='$0$', color='c', s=100)
        ax.scatter(positive_pos[:, 0], positive_pos[:, 1], marker='o', color='r', s=100)
        ax.scatter(negative_pos[:, 0], negative_pos[:, 1], marker='X', color='b', s=100)

        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        ax.margins(0.07)
        return None

    def two_form_visualisation_heatmap(self, ax, omega: np.ndarray) -> None:
        """
        Note: only visualise the orientation not the magnitude
        """
        if not(self.check_D1()):
            return None
        D1 = self.D1
        pos = self.get_circumcentres()

        c = omega
        m = np.empty(self.get_num_triangles(), dtype=str)
        index = 0
        for i in range(0, len(omega)):
            if omega[i] == 0.0:
                # m[index] = "^"
                m[index] = '_'
                index += 1
            elif omega[i] > 0:
                m[index] = 'o'
                index += 1
            elif omega[i] < 0:
                m[index] = 'X'
                index += 1
            else:
                print("Error")
                return None

        plot = mscatter(pos[:, 0], pos[:, 1], ax=ax, m=m, s=100, c=c, cmap='viridis')
        plt.colorbar(plot)

        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        ax.margins(0.07)
        return None

    def dual_zero_form_visualisation(self, ax, omega: np.ndarray) -> None:
        num_vertices = self.get_num_vertices()
        B1 = self.B1
        v = self.vertices
        e = self.edges
        edge_midpoints = self.get_edge_midpoints()
        circumcentres = self.get_circumcentres()
        edge_length = self.edge_length
        r = edge_length/np.sqrt(3)

        patches = np.empty(num_vertices, dtype=list)
        for i in range(num_vertices):
            if i in self.vc:
                j = find(B1[i])[1]
                neighbour_triangles = self.neighbour_triangles(i)
                polygon = Polygon(np.array([v[i], edge_midpoints[j[0]], circumcentres[neighbour_triangles[0]], edge_midpoints[j[1]]]))
            elif i in self.ve:
                j = find(B1[i])[1]
                neighbour_triangles = self.neighbour_triangles(i)
                points = [edge_midpoints[edge] for edge in j]
                points += [circumcentres[triangle] for triangle in neighbour_triangles]
                polygon = Polygon(ccw_sort(points))
            else:
                polygon = RegularPolygon(v[i], numVertices=6, radius=r)
            patches[i] = polygon
        patch_collection = PatchCollection(patches, alpha=0.4, cmap='viridis')
        # patch_collection = PatchCollection(patches, alpha=0.4, cmap='viridis', zorder=0)
        ax.add_collection(patch_collection)
        colors = omega
        patch_collection.set_array(colors)
        colorbar = plt.colorbar(patch_collection)
        # colorbar.ax.set(title='Dual 0-form')
        colorbar.ax.set(title=r"$\star$0-form")
        ax.set_aspect('equal')
        return None

    def dual_one_form_visualisation(self, ax, omega: np.ndarray, scale=0.01) -> None:
        num_edges = self.get_num_edges()
        edge_midpoints = self.get_edge_midpoints()
        circumcentres = self.get_circumcentres()
        patches = np.empty(num_edges, dtype=list)
        D0 = self.D0
        v = self.vertices

        for i in range(num_edges):
            tri = self.edge_of_triangles(i)
            direct = (D0 @ v)[i]
            deviation = direct * scale
            if i in self.ee:
                line = Polygon([edge_midpoints[i] - deviation, circumcentres[tri[0]] - deviation, circumcentres[tri[0]] + deviation, edge_midpoints[i] + deviation])
            else:
                line = Polygon([circumcentres[tri[0]] - deviation, circumcentres[tri[1]] - deviation, circumcentres[tri[1]] + deviation, circumcentres[tri[0]] + deviation])
            patches[i] = line
        patch_collection = PatchCollection(patches)
        ax.add_collection(patch_collection)
        colors = omega
        patch_collection.set_array(colors)
        colorbar = plt.colorbar(patch_collection)
        # colorbar.ax.set(title='Dual 1-form')
        colorbar.ax.set(title=r"$\star$1-form")
        ax.set_aspect('equal')
        return None

    def dual_two_form_visualisation(self, ax, omega: np.ndarray) -> None:
        pos = self.get_circumcentres()
        plot = ax.scatter(pos[:, 0], pos[:, 1], c=omega, cmap='viridis', edgecolor='white')
        colorbar = plt.colorbar(plot)
        # colorbar.ax.set(title='Dual 2-form')
        colorbar.ax.set(title=r"$\star$2-form")
        ax.set_aspect('equal')
        return None

    def init_d0_visualisation(self, ax, f: np.ndarray) -> None:
        if not(self.check_D0()):
            return None
        v = self.vertices
        e = self.edges
        D0 = self.D0

        old_direct = (D0 @ v)
        #direct = normalise(old_direct * self.d0(f)[:, None])
        direct = np.array(normalise(old_direct) for row in old_direct * self.d0(f)[:, None])/2
        pos = np.array([avg(v[row[0]], v[row[1]]) for row in e])
        #pos = v - ((1/2) * old_direct) - ((1/20) * direct)

        ax.quiver(pos[:, 0], pos[:, 1], direct[:, 0], direct[:, 1], scale=5, pivot='mid')
        ax.set_title(f"f(x,y) and df(x,y) on equilateral triangular mesh with sides of legth {self.side}")
        return None

    def quick_contour(self, ax, f: np.ndarray) -> None:
        print(self)
        #self.tri_mesh()

        x = self.vertices[:, 0] # x-coordinates
        y = self.vertices[:, 1] # y-coordinates

        triang = self.triangulation()

        z = f(x,y)

        #fig, ax = plt.subplots(figsize = (20*cm, 20*cm))
        ax.set_aspect('equal')
        ax.use_sticky_edges = False
        ax.margins(0.07)

        ax.tricontourf(triang, z, 20)
        ax.triplot(triang, 'ro-', lw = 0.5)
        ax.set_title(f"f(x,y) on equilateral triangular mesh with sides of legth {self.side}")
        return None
    
    def pre_gen_generic_mesh(self) -> None:
        name = self.name
        path = self.path
        edges = self.edges
        #edges = np.loadtxt(f"{name}_e.txt", dtype="int")
        num_vertices = self.get_num_vertices()
        num_edges = self.get_num_edges()
        row = np.repeat(np.arange(0, num_edges, dtype=int), 2)
        col = edges.ravel()        
        data = np.tile(np.array([-1, 1, 1, -1, -1, 1]), num_edges//3)

        D0 = csr_matrix((data, (row, col)), shape=(num_edges, num_vertices))
        # save_npz(f"{self.name}_D0.npz", D0) 
        self.D0 = D0
        self.B1 = D0.transpose()
        save_npz(f"{path}_B1.npz", self.B1)
        save_npz(f"{path}_D0.npz", D0)

        edges_to_triangles = self.e_to_s
        #edges_to_triangles = np.loadtxt(f"{name}_e_to_s.txt", dtype="int")
        num_triangles = self.get_num_triangles()

        row = np.repeat(np.arange(0, num_triangles, dtype=int), 3)
        col = edges_to_triangles.ravel()
        data = np.empty(num_triangles * 3, dtype=int)

        side = self.side
        sum_side = side
        data_index = 0
        for j in range(0, num_vertices - 2):
            if j == sum_side - 1:
                side -= 1
                sum_side += side
            elif j == sum_side - 2:
                data[data_index] = 1
                data[data_index + 1] = -1
                data[data_index + 2] = -1
                data_index += 3
            else:
                data[data_index] = 1
                data[data_index + 1] = -1
                data[data_index + 2] = -1
                data[data_index + 3] = 1
                data[data_index + 4] = 1
                data[data_index + 5] = -1
                data_index += 6

        D1 = csr_matrix((data, (row, col)), shape=(num_triangles, num_edges))
        self.D1 = D1
        self.B2 = D1.transpose()
        save_npz(f"{path}_B2.npz", self.B2)
        save_npz(f"{path}_D1.npz", D1)
        return None

    def gen_generic_mesh(self) -> None:
        self.pre_gen_generic_mesh()
        self.pre_dual_mesh()

        path = self.path
        self.hodge0 = self.get_hodge0()
        self.hodge1 = self.get_hodge1()
        self.hodge2 = self.get_hodge2()
        save_npz(f"{path}_hodge0.npz", self.hodge0)
        save_npz(f"{path}_hodge1.npz", self.hodge1)
        save_npz(f"{path}_hodge2.npz", self.hodge2)

        self.inverse_hodge0 = inv(self.hodge0)
        self.inverse_hodge1 = inv(self.hodge1)
        self.inverse_hodge2 = inv(self.hodge2)
        if self.side == 2:
            self.inverse_hodge2 = csc_matrix(self.inverse_hodge2)
        save_npz(f"{path}_inverse_hodge0.npz", self.inverse_hodge0)
        save_npz(f"{path}_inverse_hodge1.npz", self.inverse_hodge1)
        save_npz(f"{path}_inverse_hodge2.npz", self.inverse_hodge2)

        self.hodge0_dual = self.get_hodge0_dual()
        self.hodge1_dual = self.get_hodge1_dual()
        self.hodge2_dual = self.get_hodge2_dual()
        save_npz(f"{path}_hodge0_dual.npz", self.hodge0_dual)
        save_npz(f"{path}_hodge1_dual.npz", self.hodge1_dual)
        save_npz(f"{path}_hodge2_dual.npz", self.hodge2_dual)
        return None
