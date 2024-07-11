import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


def rand_cos_sim(v, costheta):
    # Ensure input tensor is a float tensor for division and sqrt operations
    v = v.float()
    
    # Form the unit vector parallel to v:
    u = v / torch.norm(v)

    # Pick a random vector:
    len_v = len(v)
    r = MultivariateNormal(torch.zeros(len_v), torch.eye(len_v)).sample()

    # Form a vector perpendicular to v:
    uperp = r - r.dot(u) * u

    # Make it a unit vector:
    uperp = uperp / torch.norm(uperp)

    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = costheta * u + torch.sqrt(1 - costheta ** 2) * uperp

    return w

target = torch.tensor(-0.9)
vec = torch.rand(20)
new_vec = rand_cos_sim(vec, target)
print(vec)
print(new_vec)
print(F.cosine_similarity(vec, new_vec, dim=0))