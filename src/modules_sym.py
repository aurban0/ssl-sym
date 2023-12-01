"""
The code for the encoder and decoder was adapted from https://github.com/Bayer-Group/giae/
"""
import torch
import torch.nn.functional as F
from torch.nn import Module
from e2cnn import gspaces
from e2cnn import nn

# Auxiliary functions

def get_non_linearity(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    relu = nn.ReLU(scalar_fields)
    norm_relu = nn.NormNonLinearity(vector_fields)
    nonlinearity = nn.MultipleModule(
        out_type,
        ['relu'] * len(scalar_fields) + ['norm'] * len(vector_fields),
        [(relu, 'relu'), (norm_relu, 'norm')]
    )
    return nonlinearity


def get_batch_norm(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    batch_norm = nn.InnerBatchNorm(scalar_fields)
    norm_batch_norm = nn.NormBatchNorm(vector_fields)
    batch_norm = nn.MultipleModule(
        out_type,
        ['bn'] * len(scalar_fields) + ['nbn'] * len(vector_fields),
        [(batch_norm, 'bn'), (norm_batch_norm, 'nbn')]
    )
    return batch_norm


# Encoder +  Decoder

class Decoder_Partial(Module):
    # MNIST Decoder (28x28 output)
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # convolution 1
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_size, hidden_size, kernel_size=1, padding=0,),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 2
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 3
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 4
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 5
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 6
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, 1, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1).unsqueeze(-1)
        # dimensions extra (for spatial)
        x = x.expand(-1, -1, 2, 2)

        x = self.block1(x)
        x = torch.nn.functional.interpolate(x, mode="bilinear", align_corners=True, scale_factor=2, recompute_scale_factor=True)
        x = self.block2(x)
        x = torch.nn.functional.interpolate(x, mode="bilinear", align_corners=True, scale_factor=2, recompute_scale_factor=True)
        x = self.block3(x)
        x = torch.nn.functional.interpolate(x, mode="bilinear", align_corners=True, scale_factor=2, recompute_scale_factor=True)
        x = self.block4(x)
        x = torch.nn.functional.interpolate(x, mode="bilinear", align_corners=True, scale_factor=1.75, recompute_scale_factor=True)
        x = self.block5(x)
        x = self.block6(x) #from 2 to 30 only 28x28 size, to fit mnist
        x = torch.sigmoid(x)
        return x

# Encoder +  Decoder

class Encoder_Partial_RGB(Module):
    def __init__(self, out_dim, hidden_dim=32, pooling="avg", in_channels=3):
        super().__init__()
        self.out_dim = out_dim
        self.pooling = pooling
        self.in_channels = in_channels
        self.r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=8)
        in_type = nn.FieldType(self.r2_act, self.in_channels*[self.r2_act.trivial_repr])
        self.input_type = in_type

        # convolution 1
        out_scalar_fields = nn.FieldType(self.r2_act, 2*hidden_dim * [
            self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block1 = nn.SequentialModule(
            # nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 2
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, 2*hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, 2*hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)

        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 4
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, 2*hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, 2*hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 6
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, 2*hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 7 --> out
        # the old output type is the input type to the next layer
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, out_dim * [
            self.r2_act.trivial_repr])  # out_dim is the number of channels in the last layer
        out_vector_field = nn.FieldType(self.r2_act, 1 * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field

        self.block_final = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor):
        x = nn.GeometricTensor(x, self.input_type)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block_final(x)
        if self.pooling == "max":
            # Max pooling
            x, _ = x.tensor.max(dim=3)
            x, _ = x.max(dim=2)
            x = x.view(x.size(0), -1)
        if self.pooling == "avg":
            x = x.tensor.mean(dim=(2, 3))

        x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]
        return x_0, x_1


# Theta Network (Invariant Network + Fully Connected)

class Theta_Eq_FC(Module):
    def __init__(self, hparams):
        super().__init__()
        # convolution 1
        self.use_one_layer = hparams.use_one_layer
        self.input_size = hparams.emb_dim_theta
        if hparams.discrete_groups:
            self.fully_connected = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.input_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.input_size // 2, self.input_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.input_size // 2, hparams.n_cyclic_groups)
                # Classify in n_cyclic_groups
            )
        else:
            if self.use_one_layer:
                self.fully_connected = torch.nn.Sequential(
                    torch.nn.Linear(self.input_size, 1),
                    torch.nn.ReLU()
                )
            else:
                self.fully_connected = torch.nn.Sequential(
                    torch.nn.Linear(self.input_size, self.input_size//2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.input_size // 2, self.input_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.input_size//2, 1),
                    torch.nn.ReLU()
                )
        self.encoder_block = Encoder_Partial_RGB(out_dim=hparams.emb_dim_theta, hidden_dim=hparams.hidden_dim_theta,
                                                 in_channels=hparams.in_channels)
        self._initialize_weights()

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.15)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = x.float()
        emb, _ = self.encoder_block(x)
        out = self.fully_connected(emb.squeeze())
        return out


class PartEqMod(Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder_Partial_RGB(out_dim=hparams.emb_dim, hidden_dim=hparams.hidden_dim,
                                           in_channels=hparams.in_channels)
        self.decoder = Decoder_Partial(input_size=hparams.emb_dim, hidden_size=hparams.hidden_dim)
        self.theta_function = Theta_Eq_FC(hparams=hparams)

    def forward(self, x, do_rot=True):
        emb, v = self.encoder(x)
        rot = self.get_rotation_matrix(v)
        recon = self.decoder(emb)
        if do_rot:
            recon = self.rot_img(recon, rot)
        return recon, rot, emb


    # Group element to rotation angle function
    def get_degrees(self, rotations):
        """
        Obtains the degrees from the matrix rotations.
        """
        # Aux functions
        def _index_from_letter(letter: str) -> int:
            if letter == "X":
                return 0
            if letter == "Y":
                return 1
            if letter == "Z":
                return 2
            raise ValueError("letter must be either X, Y or Z.")

        def _angle_from_tan(
                axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
        ) -> torch.Tensor:
            """
            Extract the first or third Euler angle from the two members of
            the matrix which are positive constant times its sine and cosine.

            Args:
                axis: Axis label "X" or "Y or "Z" for the angle we are finding.
                other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                    convention.
                data: Rotation matrices as tensor of shape (..., 3, 3).
                horizontal: Whether we are looking for the angle for the third axis,
                    which means the relevant entries are in the same row of the
                    rotation matrix. If not, they are in the same column.
                tait_bryan: Whether the first and third axes in the convention differ.

            Returns:
                Euler Angles in radians for each matrix in data as a tensor
                of shape (...).
            """

            i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
            if horizontal:
                i2, i1 = i1, i2
            even = (axis + other_axis) in ["XY", "YZ", "ZX"]
            if horizontal == even:
                return torch.atan2(data[..., i1], data[..., i2])
            if tait_bryan:
                return torch.atan2(-data[..., i2], data[..., i1])
            return torch.atan2(data[..., i2], -data[..., i1])

        def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
            """
            Convert rotations given as rotation matrices to Euler angles in radians.

            Args:
                matrix: Rotation matrices as tensor of shape (..., 3, 3).
                convention: Convention string of three uppercase letters.

            Returns:
                Euler angles in radians as tensor of shape (..., 3).
            """
            if len(convention) != 3:
                raise ValueError("Convention must have 3 letters.")
            if convention[1] in (convention[0], convention[2]):
                raise ValueError(f"Invalid convention {convention}.")
            for letter in convention:
                if letter not in ("X", "Y", "Z"):
                    raise ValueError(f"Invalid letter {letter} in convention string.")
            if matrix.size(-1) != 3 or matrix.size(-2) != 3:
                raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
            i0 = _index_from_letter(convention[0])
            i2 = _index_from_letter(convention[2])
            tait_bryan = i0 != i2
            if tait_bryan:
                central_angle = torch.asin(
                    matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
                )
            else:
                central_angle = torch.acos(matrix[..., i0, i0])

            o = (
                _angle_from_tan(
                    convention[0], convention[1], matrix[..., i2], False, tait_bryan
                ),
                central_angle,
                _angle_from_tan(
                    convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
                ),
            )
            return torch.stack(o, -1)

        # Full Logic
        ones = torch.tensor([0, 0, 1]).repeat((rotations.size()[0], 1)).cuda()
        ones = torch.reshape(ones, (rotations.shape[0], 1, 3))
        new_rot = torch.cat((rotations, ones), dim=1)  # Complete matrices
        angles = torch.rad2deg(matrix_to_euler_angles(new_rot, convention="XYZ"))[:, 2]
        return angles

    # Auxiliary functions

    def get_rotation_matrix(self, v, eps=10e-5):
        try:
            v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
            rot = torch.stack((
                torch.stack((v[:, 0], v[:, 1]), dim=-1),
                torch.stack((-v[:, 1], v[:, 0]), dim=-1),
                torch.zeros(v.size(0), 2).type_as(v)
            ), dim=-1)
            return rot
        except:
            return v

    def rot_img(self, x, rot, rot_inverse=False):
        try:
            rot = rot.clone()
            if rot_inverse:
                rot[:, 0, 1] = rot[:, 0, 1] * -1
                rot[:, 1, 0] = rot[:, 1,
                               0] * -1 # Inverse of a rotation is just the negative of the sin(theta) components
                grid = F.affine_grid(rot, x.size(), align_corners=False).type_as(x)
                x = F.grid_sample(x, grid, align_corners=False)
                return x
            else:
                grid = F.affine_grid(rot, x.size(), align_corners=False).type_as(x)
                x = F.grid_sample(x, grid, align_corners=False)
                return x
        except:  # Return x when there is no group function estimator
            return x