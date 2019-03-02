import torch
import torch.nn as nn
import torch.nn.functional as F
import dill 

# override torch's module
class Module(nn.Module):
    def to(self, device):
        # for poor people like us.. who run on single gpu/cpu doing this is fine
        self.device = device
        return super(Module, self).to(device)


class VAE(Module):
    def __init__(self, z_size, loss_type="mse" ):
        # loss_type = mse/ce
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, z_size*2)
        )
       
    
        if (loss_type=="mse"):
            self.decoder = nn.Sequential(
                nn.Linear(z_size, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 28*28),
    #             nn.Sigmoid()
            )
            self.loss_fn = nn.MSELoss()
        elif loss_type=="ce":
            self.decoder = nn.Sequential(
                nn.Linear(z_size, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 28*28),
                nn.Sigmoid()
            )
            self.loss_fn = nn.BCELoss()
        else:
            raise Exception("invalid loss_type should be mse/ce")
        self.loss_type = loss_type
        self.z_size = z_size
        
    def to(self, device):
        self.device = device
        return super(VAE, self).to(device)
           
    def encode (self, x):
        enc = self.encoder(x)    
        mu = enc[:, :self.z_size]
        log_sigma = enc[:, self.z_size:]
        return mu, log_sigma
    
    def decode (self, z):
         return self.decoder(z)
        
    def forward(self, x):
        mu, log_sigma = self.encode(x)
        
        noise = torch.randn_like(log_sigma)
        z_hat = mu + torch.exp(log_sigma/2)*noise
        
        dec = self.decode(z_hat)
        return dec, mu, log_sigma
    
    def loss (self, x, recon, mu, log_sigma):
        recon_loss = self.loss_fn(recon, x)*(28*28) 
        # In case of MSE loss variance of sqrt(0.5) in pixel space? (Think: Shouldn't we be aiming for lesser variance than this?)
        kl_div = - torch.mean(0.5 * torch.sum(torch.exp(log_sigma) + mu**2 - 1. - log_sigma, dim=1))
        loss = recon_loss - kl_div
        return loss, recon_loss, kl_div
    
class PlanarFlow(Module):
    def __init__(self, D:int):
        super().__init__()
        self.D = D
        self.u = nn.Parameter(torch.rand(D))
        self.w = nn.Parameter(torch.rand(D))
        self.b = nn.Parameter(torch.rand((1,)))
        
        self.reset_parameters()

    def reset_parameters(self):

        self.u.data.uniform_(-0.01, 0.01)
        self.w.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)

        
    def forward(self, z):
#         w = self.w.detach()
        w = self.w
        wtu = torch.sum(w*self.u)
        # In this case we are reparamtrizing only when the invertibility conditions are violated 
        # pymc3 reparametrizes everywhere 
        # See https://github.com/pymc-devs/pymc3/blob/1cdd1631bea48fef8d140e37c3588a8208498ba0/pymc3/variational/flows.py#L374
        
        if (wtu<=-1):
            u_new = self.u + (-1 + F.softplus(wtu) - wtu)*w/(torch.sum(w*w)+1e-8)
        else:
            u_new = self.u
            
        wtz_p_b =torch.tanh(torch.sum(self.w*z, dim=1) + self.b)
        
        out = z + u_new.unsqueeze(0)*(wtz_p_b).unsqueeze(1)
        det_j = 1 + (1-wtz_p_b**2)*torch.sum(u_new*self.w)
        
        return out, det_j
    
class PlanarVAE(VAE):
    def __init__(self, flow_length, **kwargs):
        super().__init__(**kwargs)
        self.flow_length = flow_length
        self.flows = nn.ModuleList([PlanarFlow(self.z_size) for _ in range(self.flow_length)])
        
    def forward(self, x):
        mu, log_sigma = self.encode(x)
        
        noise = torch.randn_like(log_sigma)
        z_hat = mu + torch.exp(log_sigma/2)*noise
        
        sum_det = torch.tensor(0.0, device=self.device)
        
        for flow in self.flows:
            z_hat, log_det = flow(z_hat)
            sum_det += torch.mean(log_det)
            
        dec = super(PlanarVAE, self).decode(z_hat)
        return dec, mu, log_sigma, sum_det
    
    def decode(self, z):
        z_hat = z
        for flow in self.flows:
            z_hat, log_det = flow(z_hat)
        dec = super(PlanarVAE, self).decode(z_hat)
        return dec
        
    def loss(self, x, recon, mu, log_sigma, sum_det):
        _, recon_loss, kl_div = super(PlanarVAE, self).loss(x, recon, mu, log_sigma)
        kl_div =kl_div - sum_det
        loss = recon_loss - kl_div
        return loss, recon_loss, kl_div
    
# class IAFFlow(Module):
#     def __init__(self, D:int):
#         super().__init__()
#         self.D = D
#         self.u = nn.Parameter(torch.rand(D))
#         self.w = nn.Parameter(torch.rand(D))
#         self.b = nn.Parameter(torch.rand((1,)))
        
#         self.reset_parameters()

#     def reset_parameters(self):

#         self.u.data.uniform_(-0.01, 0.01)
#         self.w.data.uniform_(-0.01, 0.01)
#         self.b.data.uniform_(-0.01, 0.01)

        
#     def forward(self, z):
# #         w = self.w.detach()
#         w = self.w
#         wtu = torch.sum(w*self.u)
#         u_new = self.u + (-1 + F.softplus(wtu) - wtu)*w/(torch.sum(w*w)+1e-8)
        
#         wtz_p_b =torch.tanh(torch.sum(self.w*z, dim=1) + self.b)
        
#         out = z + u_new.unsqueeze(0)*(wtz_p_b).unsqueeze(1)
#         det_j = 1 + (1-wtz_p_b**2)*torch.sum(u_new*self.w)
        
#         return out, det_j
    
# class IAFVAE(VAE):
#     def __init__(self, flow_length, **kwargs):
#         super().__init__(**kwargs)
#         self.flow_length = flow_length
#         self.flows = nn.ModuleList([IAFFlow(self.z_size) for _ in range(self.flow_length)])
        
#     def forward(self, x):
#         mu, log_sigma = self.encode(x)
        
#         noise = torch.randn_like(log_sigma)
#         z_hat = mu + torch.exp(log_sigma/2)*noise
        
#         sum_det = torch.tensor(0.0, device=self.device)
        
#         for flow in self.flows:
#             z_hat, log_det = flow(z_hat)
#             sum_det += torch.mean(log_det)
            
#         dec = super(PlanarVAE, self).decode(z_hat)
#         return dec, mu, log_sigma, sum_det
    
#     def decode(self, z):
#         z_hat = z
#         for flow in self.flows:
#             z_hat, log_det = flow(z_hat)
#         dec = super(PlanarVAE, self).decode(z_hat)
#         return dec
        
#     def loss(self, x, recon, mu, log_sigma, sum_det):
#         _, recon_loss, kl_div = super(PlanarVAE, self).loss(x, recon, mu, log_sigma)
#         kl_div =kl_div - sum_det
#         loss = recon_loss - kl_div
#         return loss, recon_loss, kl_div
    