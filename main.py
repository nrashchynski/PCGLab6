import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def create_letter_r():
    vertices = np.array([
        [0, 0, 0], [0, 0, 6], [0, 1, 6], [0, 1, 0],
        [1, 0, 0], [1, 0, 6], [1, 1, 6], [1, 1, 0],

        [2, 0, 3], [2, 0, 6], [2, 1, 6], [2, 1, 3],
        [3, 0, 3], [3, 0, 6], [3, 1, 6], [3, 1, 3],

        [0, 1, 3], [3, 1, 3], [0, 1, 4], [3, 1, 4],
        [0, 0, 3], [3, 0, 3], [0, 0, 4], [3, 0, 4],

        [0, 1, 5], [3, 1, 5], [0, 1, 6], [3, 1, 6],
        [0, 0, 5], [3, 0, 5], [0, 0, 6], [3, 0, 6],
    ])

    faces = [
        [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
        [3, 0, 4, 7],  [4, 5, 6, 7], [0, 1, 2, 3],

        [8, 9, 13, 12], [9, 10, 14, 13], [10, 11, 15, 14],
        [11, 8, 12, 15], [12, 13, 14, 15], [8, 9, 10, 11],

        [16, 17, 21, 20], [18, 19, 23, 22], [16, 19, 22, 20],
        [17, 19, 23, 21], [20, 21, 23, 22], [16, 17, 19, 18],

        [24, 25, 29, 28], [26, 27, 31, 30], [24, 26, 30, 28],
        [25, 27, 31, 29], [28, 29, 31, 30], [24, 25, 27, 26],
    ]
    return vertices, faces


def scale(vertices, sx, sy, sz):
    scale_matrix = np.diag([sx, sy, sz])
    return np.dot(vertices, scale_matrix)


def translate(vertices, tx, ty, tz):
    return vertices + np.array([tx, ty, tz])


def create_general_transformation_matrix(sx, sy, sz, tx, ty, tz, rotation_angle=0):
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)

    rotation_z = np.array([
        [cos_a, sin_a, 0, 0],
        [-sin_a, cos_a, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    scaling = np.diag([sx, sy, sz, 1])

    translation = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

    transformation_matrix = translation @ rotation_z @ scaling
    return transformation_matrix


def apply_projection(vertices, matrix):
    ones = np.ones((vertices.shape[0], 1))
    vertices_homogeneous = np.hstack((vertices, ones))
    projected_vertices = vertices_homogeneous.dot(matrix.T)
    return projected_vertices[:, :3]


def create_projection_matrix(plane):
    if plane == 'Oxy':
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ])
    elif plane == 'Oxz':
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif plane == 'Oyz':
        return np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid projection plane. Choose 'Oxy', 'Oxz', or 'Oyz'")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('3D Model of the Letter "Р"')

        self.vertices, self.faces = create_letter_r()

        self.figure = plt.figure(figsize=(5, 4))
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_plot_widget = self.canvas_plot.get_tk_widget()
        self.canvas_plot_widget.pack(fill=tk.BOTH, expand=1)

        ttk.Label(root, text="Scale X").pack()
        self.scale_x = ttk.Scale(root, from_=0.5, to=3.0, orient=tk.HORIZONTAL)
        self.scale_x.set(2)
        self.scale_x.pack(fill=tk.X)

        ttk.Label(root, text="Translate X").pack()
        self.translate_x = ttk.Scale(root, from_=-10, to=10, orient=tk.HORIZONTAL)
        self.translate_x.set(0)
        self.translate_x.pack(fill=tk.X)

        ttk.Label(root, text="Translate Y").pack()
        self.translate_y = ttk.Scale(root, from_=-10, to=10, orient=tk.HORIZONTAL)
        self.translate_y.set(0)
        self.translate_y.pack(fill=tk.X)

        ttk.Label(root, text="Translate Z").pack()
        self.translate_z = ttk.Scale(root, from_=-10, to=10, orient=tk.HORIZONTAL)
        self.translate_z.set(0)
        self.translate_z.pack(fill=tk.X)

        ttk.Label(root, text="Rotate Z (degrees)").pack()
        self.rotate_z = ttk.Scale(root, from_=-180, to=180, orient=tk.HORIZONTAL)
        self.rotate_z.set(0)
        self.rotate_z.pack(fill=tk.X)

        ttk.Button(root, text="Oxy Projection", command=self.show_oxy_projection).pack(fill=tk.X)
        ttk.Button(root, text="Oxz Projection", command=self.show_oxz_projection).pack(fill=tk.X)
        ttk.Button(root, text="Oyz Projection", command=self.show_oyz_projection).pack(fill=tk.X)

        self.matrix_label = ttk.Label(root, text="Transformation Matrix:\n")
        self.matrix_label.pack()

        self.update_plot()

    def update_plot(self):
        self.ax.clear()

        sx = self.scale_x.get()
        tx = self.translate_x.get()
        ty = self.translate_y.get()
        tz = self.translate_z.get()
        rotation_angle_deg = self.rotate_z.get()
        rotation_angle_rad = np.radians(rotation_angle_deg)

        transformed_vertices = scale(self.vertices, sx, sx, sx)
        transformed_vertices = translate(transformed_vertices, tx, ty, tz)

        cos_a = np.cos(rotation_angle_rad)
        sin_a = np.sin(rotation_angle_rad)
        rotation_z = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

        transformed_vertices = transformed_vertices @ rotation_z.T

        for face in self.faces:
            face_vertices = [transformed_vertices[idx] for idx in face]
            poly = Poly3DCollection([face_vertices], alpha=0.5, edgecolor='k')
            self.ax.add_collection3d(poly)

        matrix = create_general_transformation_matrix(sx, sx, sx, tx, ty, tz, rotation_angle_rad)
        matrix_text = "Transformation Matrix:\n" + "\n".join(
            ["\t".join([f"{val:.2f}" for val in row]) for row in matrix]
        )
        self.matrix_label.config(text=matrix_text)

        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])

        self.canvas_plot.draw()
        self.root.after(100, self.update_plot)

    def show_oxy_projection(self):
        self.show_projection('Oxy')

    def show_oxz_projection(self):
        self.show_projection('Oxz')

    def show_oyz_projection(self):
        self.show_projection('Oyz')

    def show_projection(self, plane):
        proj_window = tk.Toplevel(self.root)
        proj_window.title(f"{plane} Projection")

        fig = plt.figure(figsize=(5, 4))
        ax_proj = fig.add_subplot(111)

        projection_matrix = create_projection_matrix(plane)
        projected_vertices = apply_projection(self.vertices, projection_matrix)

        if plane == 'Oxy':
            ax_proj.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='b')
            ax_proj.set_xlabel("X")
            ax_proj.set_ylabel("Y")
        elif plane == 'Oxz':
            ax_proj.scatter(projected_vertices[:, 0], projected_vertices[:, 2], c='g')
            ax_proj.set_xlabel("X")
            ax_proj.set_ylabel("Z")
        elif plane == 'Oyz':
            ax_proj.scatter(projected_vertices[:, 1], projected_vertices[:, 2], c='r')
            ax_proj.set_xlabel("Y")
            ax_proj.set_ylabel("Z")

        canvas_proj = FigureCanvasTkAgg(fig, master=proj_window)
        canvas_proj_widget = canvas_proj.get_tk_widget()
        canvas_proj_widget.pack(fill=tk.BOTH, expand=1)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
