import numpy as np
import torch

if __name__ == '__main':
    h = 0.06762433378062413
    K_np = [[1.00000000e-02, 0.00000000e+00], [1.00000000e-02, 9.14610104e-07], [1.00000139e-02, 2.05787273e-06],
            [1.00002639e-02, 1.46338522e-05], [1.00003620e-02, 1.80664943e-05], [1.00005154e-02, 2.28654776e-05],
            [1.00005154e-02, 2.28655472e-05]]
    K_tensor = torch.tensor(K_np, dtype=torch.float64)
    E_np = [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40]
    E_tensor = torch.tensor(E_np, dtype=torch.float64)
    r1 = np.dot(K_np.T, E_np)
    r2 = torch.matmul(K_tensor.T, E_tensor)
    print(r1,r2)
