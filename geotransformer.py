import numpy as np
import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer


def pairwise_angle_distance_success(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """

    x_zx = torch.unsqueeze(x,1).repeat(1,x.shape[1],1,1)
    x_verse = torch.unsqueeze(x,2).repeat(1,1,x.shape[1],1)
    lianxian_vector = x_zx - x_verse

    y_expend_dim = torch.unsqueeze(y,2).repeat(1,1,y.shape[1],1)

    map = torch.zeros((y_expend_dim.shape[0],y_expend_dim.shape[1],y_expend_dim.shape[2]))



    for win_num in range(y_expend_dim.shape[0]):
        for i in range(y_expend_dim.shape[1]):
            normal_vector =  y_expend_dim[win_num,i,:,:]
            rad_angle_list = comput_paries_angle(normal_vector, lianxian_vector[win_num, i, :, :])
            map[win_num,i] = rad_angle_list

    return map.cuda()

def pairwise_angle_distance(
        x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """

    x_zx = torch.unsqueeze(x, 1).repeat(1, x.shape[1], 1, 1)
    x_verse = torch.unsqueeze(x, 2).repeat(1, 1, x.shape[1], 1)
    lianxian_vector = x_zx - x_verse

    y_expend_dim = torch.unsqueeze(y, 2).repeat(1, 1, y.shape[1], 1)

    dot_product = torch.sum(torch.mul(y_expend_dim, lianxian_vector),3)




    normal_norm = torch.norm(y_expend_dim,dim=3)
    lianxian_norm = torch.norm(lianxian_vector,dim=3)


    cos_angle = dot_product / (normal_norm*lianxian_norm+1e-6)
    cos_angle = torch.clip(cos_angle,-1,1)
    rad_angle = torch.acos(cos_angle)
    distance = torch.abs(rad_angle - rad_angle.transpose(1, 2))
    return distance

def comput_paries_angle(normals,normals2):
    # normals = normals+1e-6
    # normals2 =  normals2+1e-6
    normals_vector = normals[0]
    map = torch.zeros((normals.shape[0]))
    for j in range(normals2.shape[0]):
        tenosr = normals2[j]
        dot_product = torch.dot(normals_vector,tenosr)
        normal_norm = torch.norm(normals_vector)
        lianxian_norm = torch.norm(tenosr)
        cos_angle = dot_product / (normal_norm*lianxian_norm+1e-6)
        rad_angle = torch.acos(cos_angle)
        map[j] = rad_angle
    return map

def comput_paries_angle_v2(normals,normals2):
    normals = torch.unsqueeze(normals,0)+1e-6
    normals2 = torch.unsqueeze(normals2,0)+1e-6
    x = torch.matmul(normals, normals2.transpose(-1, -2)) / torch.matmul(torch.sqrt(torch.unsqueeze(torch.sum(torch.mul(normals, normals2), 2), 2)),
             torch.sqrt(torch.unsqueeze(torch.sum(torch.mul(normals, normals2), 2), 2)).transpose(1,2))
    x = torch.where(torch.isnan(x),torch.full_like(x,0),x)
    x = torch.clip(x,-1,1)
    seta = torch.arccos(x)
    return seta

def comput_angle(normals):
    x = torch.matmul(normals, normals.transpose(-1, -2)) / torch.matmul(torch.sqrt(torch.unsqueeze(torch.sum(torch.mul(normals, normals), 2), 2)),
             torch.sqrt(torch.unsqueeze(torch.sum(torch.mul(normals, normals), 2), 2)).transpose(1,2))
    x = torch.where(torch.isnan(x),torch.full_like(x,0),x)
    x = torch.clip(x,-1,1)
    seta = torch.arccos(x)
    return seta

class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_cat = nn.Linear(3, hidden_dim)


        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points,normals):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        angle_map = pairwise_angle_distance(points, normals)

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d
        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices,angle_map

    def forward(self, points,normals,add_num):
        d_indices, a_indices,angle_map = self.get_embedding_indices(points,normals)
        seta = comput_angle(normals)
        normals_seta_embeddings = angle_map
        d_embeddings = d_indices
        cat_normals = torch.cat([d_embeddings[..., None], seta[..., None], normals_seta_embeddings[..., None]], dim=-1)
        return cat_normals


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_normals_c,
        src_normals_c,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings_list = []
        src_embeddings_list = []
        for win_size in [2,4]:
            add_row_ref_num = int(np.ceil(ref_points.shape[1] / (win_size * win_size)) * win_size * win_size) - ref_points.shape[1]
            add_row_ref_4x4 = torch.cat([ref_points, torch.zeros((1, add_row_ref_num, 3), dtype=torch.float32).cuda()],1)
            add_row_ref_4x4_normal = torch.cat([ref_normals_c, torch.zeros((1, add_row_ref_num, 3), dtype=torch.float32).cuda()],1)

            ref_stage0 = self.embedding(
                add_row_ref_4x4.view(1, add_row_ref_4x4.shape[1] // (win_size*win_size), win_size*win_size, 3).view(add_row_ref_4x4.shape[1] //(win_size * win_size), (win_size * win_size), 3),
                add_row_ref_4x4_normal.view(1, add_row_ref_4x4_normal.shape[1] // (win_size*win_size), (win_size * win_size), 3).view(
                    add_row_ref_4x4_normal.shape[1] // (win_size*win_size), win_size*win_size, 3),add_row_ref_num)

            add_row_src_num = int(np.ceil(src_points.shape[1] / (win_size * win_size)) * win_size * win_size) - src_points.shape[1]
            add_row_src_4x4 = torch.cat([src_points, torch.zeros((1, add_row_src_num, 3), dtype=torch.float32).cuda()], 1)
            add_row_src_4x4_normal = torch.cat([src_normals_c, torch.zeros((1, add_row_src_num, 3), dtype=torch.float32).cuda()],1)
            src_stage0 = self.embedding(
                add_row_src_4x4.view(1, add_row_src_4x4.shape[1] // (win_size * win_size), (win_size * win_size), 3).view(add_row_src_4x4.shape[1] // (win_size * win_size), (win_size * win_size), 3),
                add_row_src_4x4_normal.view(1, add_row_src_4x4_normal.shape[1] // (win_size * win_size), (win_size * win_size), 3).view(
                    add_row_src_4x4_normal.shape[1] // (win_size * win_size), (win_size * win_size), 3),add_row_src_num)
            src_embeddings_list.append(src_stage0)
            ref_embeddings_list.append(ref_stage0)

        src_embeddings_list.append(self.embedding(src_points, src_normals_c,0))
        ref_embeddings_list.append(self.embedding(ref_points, ref_normals_c,0))
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings_list,
            src_embeddings_list,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats
