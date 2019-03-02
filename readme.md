We set out to compare planar flow and normal vae. Is the assumption that p(z|x) or q(z|x) really bad?

First comparision
------

Planar flow with MSE loss, unconstrained output, flow length of 10 Vs VAE :

Optimizer : ADAM
Learning Rate: 1e-3

|             | T-Planar  | T-VAE  |    Tr-Planar|   Tr-VAE  |
|:-----------:|:---------:|:------:|:-----------:|:---------:|
|log p(x)     | 30.2500   | 29.7822|    29.7901  |   29.1716 |
|kl-div       | -7.5695   | -7.7904|    -7.5966  |  -7.8220  |
|recon loss   | 22.6805   | 21.9918|    22.1936  |  21.3496  |

Planar flow with CE loss, sigmoid output, flow length of 10 Vs VAE :

|             | T-Planar  | T-VAE    |    Tr-Planar|   Tr-VAE  |
|:-----------:|:---------:| :------: |:-----------:|:---------:|
|log p(x)     | 109.4825  | 109.7383 |  104.1598   |  104.4876 |   
|kl-div       | -14.2876  | -14.2025 |  -14.3105   |  -14.2629 |
|recon loss   | 95.1949   |  95.5359 |    89.8493  |  90.2247  |


For the above params of flow are initialized as uniform(-.01, .01), if I use other initialization like just rand(), results are somewhat worse. 

-------------------

This are independent reproduction, and does not use same arch/hyper-param as paper

-------------------------------

Since paper uses RMSProp, lets try that too. :::: Same conclusion.