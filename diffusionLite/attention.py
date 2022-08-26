from inspect import isfunction
import torch

def default(val, d):
    if val is not None: return val
    if isfunction(d): return d()
    return d

def CrossAttention_forward_lite(self, x, context=None, mask=None):
    if mask is not None:
        raise RuntimeError("CrossAttentionLite is not currently compatible with masks. This is a fixeable problem, please contact the maintainers.")

    context = default(context, x)
    q = self.to_q(x)
    k = self.to_k(context)
    v = self.to_v(context)

    q = self.reshape_heads_to_batch_dim(q)
    v = self.reshape_heads_to_batch_dim(v)
    k = self.reshape_heads_to_batch_dim(k)

    def inplace_att(q,k,v):
        """compute sthe attention in place"""
        # TODO this is lef as an inspiration to improve the existing code
        #      one needs to check if it stays valid in fp16
        # einsum in place
        #sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        k = torch.permute(k, (0, 2, 1))
        sim = torch.zeros( (q.shape[0], q.shape[1], k.shape[2]) , device=q.device) # TODO cannot allocate that on gpu
        sim = torch.bmm(q, k, out=sim)
        # * scale
        sim.mul_(self.scale)
        # softmax in place
        # https://stackoverflow.com/a/53741386/6422174
        torch.exp(sim, out=sim)
        summed = torch.sum(sim, dim=-1, keepdim=True)
        sim /= summed
        out = torch.einsum("b i j, b j d -> b i d", sim, v)
        return out

    def basic_att(q,k,v):
        """basic way to compute the attention"""
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale # TODO we could reuse a buffer for that to reduce allocations
        attn = sim.softmax(dim=-1) # TODO doing this in place could half the memory use
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        return out

    def sliced_att(q,k,v):
        """compytes the attention by slices along the j axis"""
        (b,i,d) = q.shape
        (b,nb_j,d) = k.shape
        out = torch.zeros( (b,i,d) , device=q.device)
        nb_slices = 5 # TODO minimum that run on my system with no OOM, make this configureable
        for slice in range(nb_slices):
            j_start = (slice * nb_j) // nb_slices
            j_end = ((slice+1) * nb_j) // nb_slices
            out += basic_att(q,k[:,j_start:j_end,:],v[:,j_start:j_end,:])
        return out

    out = sliced_att(q,k,v)
    out = self.reshape_batch_dim_to_heads(out)
    return self.to_out(out)
