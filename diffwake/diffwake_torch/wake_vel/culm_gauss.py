import torch
from torch import nn


def safe_sqrt(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor: 
    return torch.sqrt(torch.clamp(x, min=eps))

def wake_expansion(delta_x, ct, ti, D, a_s, b_s, c_s1, c_s2):
    beta = 0.5 * (1.0 + safe_sqrt(1.0 - ct)) / safe_sqrt(1.0 - ct + 1e-8)
    k = a_s * ti + b_s
    eps = (c_s1 * ct + c_s2) * safe_sqrt(beta)
    x_tilde = torch.abs(delta_x) / D
    sigma_y = k * x_tilde + eps
    return sigma_y


class CumulativeGaussCurlVelocityDeficit(nn.Module):
    def __init__(self,
                 a_s=0.179367259,
                 b_s=0.0118889215,
                 c_s1=0.0563691592,
                 c_s2=0.13290157,
                 a_f=3.11,
                 b_f=-0.68,
                 c_f=2.41,
                 alpha_mod=1.0):
        super().__init__()
        self.register_buffer("a_s", torch.tensor(a_s))
        self.register_buffer("b_s", torch.tensor(b_s))
        self.register_buffer("c_s1",torch.tensor(c_s1))
        self.register_buffer("c_s2",torch.tensor(c_s2))
        self.register_buffer("a_f", torch.tensor(a_f))
        self.register_buffer("b_f", torch.tensor(b_f))
        self.register_buffer("c_f", torch.tensor(c_f))
        self.register_buffer("alpha_mod", torch.tensor(alpha_mod))
    

    def vec_sum_lbda(self, ii, Ctmp, u_initial, x, x_c, y_c, z_c,
                    ct, ti, D, sigma_n_sq, y_i_loc, z_i_loc, defl):
        """
        Vektoriserad λ-sum som exakt motsvarar NumPy-loopen,
        men utan Python-for – bara slicing + broadcast.
        """

        if ii == 0:                         
            return torch.zeros_like(u_initial)

        ct_m = ct[:, :ii]                   # (B, ii, 1, 1)
        ti_m = ti[:, :ii]
        x_m  = x_c[:, :ii]
        y_m  = y_c[:, :ii]
        z_m  = z_c[:, :ii]

        Ctmp_m = Ctmp[:ii].permute(1, 0, 2, 3, 4)

        dx_m = x.unsqueeze(1) - x_m.unsqueeze(2)         # (B, ii, T, Ny, Nz)

        sigma_i = wake_expansion(
            dx_m,
            ct_m.unsqueeze(2), ti_m.unsqueeze(2), D,
            self.a_s, self.b_s, self.c_s1, self.c_s2
        )
        sigma_i_sq = sigma_i ** 2
        S_i = sigma_n_sq.unsqueeze(1) + sigma_i_sq       # (B, ii, T, Ny, Nz)

        defl_m = defl.unsqueeze(1).expand(-1, ii, -1, -1, -1)
        Y_i = ((y_i_loc.unsqueeze(1) - y_m.unsqueeze(2) - defl_m) ** 2) / (2 * S_i)
        Z_i = ((z_i_loc.unsqueeze(1) - z_m.unsqueeze(2)) ** 2) / (2 * S_i)

        lbda = sigma_i_sq / S_i * torch.exp(-Y_i) * torch.exp(-Z_i)

        # -------- 3.  NumPy-exakt kombination  -------------------------
        #            OBS: division med u_initial, ingen invers-mult.
        term = lbda * (Ctmp_m / u_initial.unsqueeze(1))  # (B, ii, T, Ny, Nz)
        return term.sum(dim=1)                           # → (B, T, Ny, Nz)
    

    def forward(self, ii, 
                x_i,            #(n,1,1,1)
                y_i,            #(n,1,1,1)
                z_i,            #(n,1,1,1)
                u_i,            #(n,1,1,1)
                deflection_field,#(n,t,3,3)
                yaw_i,          #(n,1,1,1)
                ti,             #(n,t,1,1)
                ct,             #(n,t,1,1)
                D,              #scalar
                turb_u_wake,    #(n,t,d,d)
                Ctmp,           #(t,n,t,d,d)
                x,              #(n,t,d,d)
                y,              #(n,t,d,d)
                z,              #(n,t,d,d)
                u_initial):     #(n,t,d,d)

        mean_cubed = u_i.pow(3).mean(dim=(2, 3), keepdim=True)
        turb_avg_vels = torch.sign(mean_cubed) * torch.abs(mean_cubed).pow(1/3)
        delta_x = x - x_i

        sigma_n = wake_expansion(delta_x, ct[:, ii:ii+1], ti[:, ii:ii+1], D,
                                 self.a_s, self.b_s, self.c_s1, self.c_s2)

        y_i_loc = y_i.mean(dim=(2, 3), keepdim=True)
        z_i_loc = z_i.mean(dim=(2, 3), keepdim=True)

        x_coord = x.mean(dim=(2, 3), keepdim=True)
        y_coord = y.mean(dim=(2, 3), keepdim=True)
        z_coord = z.mean(dim=(2, 3), keepdim=True)

        sigma_n_sq = sigma_n.square()
        
        sum_lbda = self.vec_sum_lbda(ii, Ctmp, u_initial, x,
                    x_coord, y_coord, z_coord, ct, ti, D,
                    sigma_n_sq, y_i_loc, z_i_loc, deflection_field)

            
        x_tilde = torch.abs(delta_x) / D

        inside = (y - y_i_loc - deflection_field)**2 + (z - z_i_loc)**2
        r_tilde = safe_sqrt(inside)     
        r_tilde = r_tilde/D

        n = self.a_f * torch.exp(self.b_f * x_tilde) + self.c_f
        a1 = 2 ** (2 / n - 1)
        a2 = 2 ** (4 / n - 2)

        gamma_val = torch.exp(torch.lgamma(2 / n))
        #(gamma_val)

        tmp = a2 - ((n * ct[:, ii:ii+1]) * torch.cos(yaw_i) /
                    (16.0 * gamma_val * torch.sign(sigma_n) *
                     (torch.abs(sigma_n) ** (4 / n)) * (1 - sum_lbda) ** 2))

        C_new = (a1 - safe_sqrt(tmp)) * (1 - sum_lbda)


        Ctmp_out = Ctmp.clone()
        Ctmp_out[ii] = C_new

        yR = y - y_i_loc
        xR = yR * torch.tan(yaw_i) + x_i

        velDef = C_new * torch.exp(-(torch.abs(r_tilde) ** n) / (2 * sigma_n_sq))
        velDef[~( (x - xR) >= 0.1)] = 0.0
        turb_u_wake.add_(velDef * turb_avg_vels)
        return turb_u_wake, Ctmp_out