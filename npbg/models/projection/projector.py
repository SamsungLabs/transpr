import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectVertices(nn.Module):
    def __init__(self, near_z=0.01, far_z=100):
        super().__init__()
        self.near_z = near_z
        self.far_z = far_z

        self.eps = 1e-8

    def align_vertices(self, vertices):
        vertices_screen = torch.stack([
            (-vertices[:, 1] / vertices[:, 3] + 1) * 0.5,
            (vertices[:, 0] / vertices[:, 3] + 1) * 0.5
        ], dim=1)

        mask = (vertices_screen[:, 0] > 0) & (vertices_screen[:, 0] < 1) & \
               (vertices_screen[:, 1] > 0) & (vertices_screen[:, 1] < 1) & \
               (vertices[:, 2] > self.near_z)

        return vertices_screen, mask

    def setup_model(self, vertices, view_matrix, proj_matrix):
        vertices_view = torch.mm(vertices, view_matrix)
        vertices_proj = torch.mm(vertices_view, proj_matrix)
        raw_depth = vertices_proj[:, 2]
        vertices_aligned, mask = self.align_vertices(vertices_proj)

        return vertices_aligned, raw_depth, mask

    def forward(self, vertices, view_matrix, proj_matrix):
        vertices_view, depth, exclude_mask = self.setup_model(vertices, view_matrix, proj_matrix)
        return vertices_view, depth, exclude_mask
