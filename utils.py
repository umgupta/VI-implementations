import torch 
import dill 

def save(fname, model, extras=[]):

    # save model, optimizer, epoch
    data = {}

    for extra in extras:
        data[extra] = getattr(model, extra)
        
    data["model"] = model.state_dict()
    torch.save(data, fname, pickle_module=dill)

def load(self, fname, model):

    data = torch.load(fname, pickle_module=dill)
    
    for key in data:
        if (key=="model"):
            model_dict = model.state_dict()
            model_dict.update(data["model"])
            model.load_state_dict(model_dict)
        else:
            setattr(model, key, data[key])

                
def reconstruct(model, x, num=10):
    N = x.shape[0]
    with torch.no_grad():
        model.eval()
        mu, log_sigma = model.encode(x) 
        mu.unsqueeze_(1)
        log_sigma.unsqueeze_(1)
        
        noise = torch.randn((N, num, model.z_size), device=model.device)
        
        z = mu + torch.exp(log_sigma)*noise
        z = z.view(-1, model.z_size)
        recon = model.decode(z).unsqueeze_(1)
        
        return recon.view((N, num)+ recon.shape[2:])
    
def generate(model, z):
    with torch.no_grad():
        model.eval()
        z = torch.tensor(z, device=model.device).float()
        return model.decode(z)