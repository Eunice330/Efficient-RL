import torch
from . import util

# @profile
def pnqp(H, q, lower, upper, x_init=None, n_iter=20):
    GAMMA = 0.1
    n_batch, n, _ = H.size()
    pnqp_I = 1e-11*torch.eye(n).type_as(H).expand_as(H)


    def obj(x):
        return 0.5*util.bquad(x, H) + util.bdot(q, x)

    if x_init is None:
        if n == 1:
            x_init = -(1./H.squeeze(2))*q
        else:
            H_lu = H.btrifact()
            x_init = -q.btrisolve(*H_lu) # Clamped in the x assignment.
    else:
        x_init = x_init.clone() # Don't over-write the original x_init.

    x = util.eclamp(x_init, lower, upper)

    # Active examples in the batch.
    J = torch.ones(n_batch).type_as(x).byte()

    for i in range(n_iter):
        g = util.bmv(H, x) + q
        Ic = ((x == lower) & (g > 0)) | ((x == upper) & (g < 0))
        If = 1-Ic

        if If.is_cuda:
            Hff_I = util.bger(If.float(), If.float()).type_as(If)
            not_Hff_I = 1-Hff_I
            Hfc_I = util.bger(If.float(), Ic.float()).type_as(If)
        else:
            Hff_I = util.bger(If, If)
            not_Hff_I = 1-Hff_I
            Hfc_I = util.bger(If, Ic)

        g_ = g.clone()
        g_[Ic] = 0.
        H_ = H.clone()
        #print('not Hff I', not_Hff_I, not_Hff_I.dtype)
        not_Hff_I = not_Hff_I.long()
        not_Hff_I = not_Hff_I.type(torch.uint8)
        H_[not_Hff_I] = 0.0
        H_ += pnqp_I

        if n == 1:
            dx = -(1./H_.squeeze(2))*g_
        else:
            H_lu_ = H_.btrifact()
            dx = -g_.btrisolve(*H_lu_)

        J = torch.norm(dx, 2, 1) >= 1e-4
        m = J.sum().item() # Number of active examples in the batch.
        if m == 0:
            return x, H_ if n == 1 else H_lu_, If, i

        alpha = torch.ones(n_batch).type_as(x)
        decay = 0.1
        max_armijo = GAMMA
        count = 0
        while max_armijo <= GAMMA and count < 10:
            # Crude way of making sure too much time isn't being spent
            # doing the line search.
            # assert count < 10

            maybe_x = util.eclamp(x+torch.diag(alpha).mm(dx), lower, upper)
            armijos = (GAMMA+1e-6)*torch.ones(n_batch).type_as(x)

            #x = x.float()
            #maybe_x = maybe_x.float()
            #g = g.float()
            
            #print('x input armi type', x.dtype)
            #print('maybe x type', maybe_x.dtype)
            #print('obj x type', obj(x).dtype)
            #print('obj maybe', obj(maybe_x).dtype)
            #print('g type', g.dtype)
            #print('maybe_x input armi type', maybe_x.dtype)
            #print('util.bdot(g, x-maybe_x) type', util.bdot(g, x-maybe_x).dtype) 
            #armijos[J] = (obj(x).float()-obj(maybe_x).float())[J]/util.bdot(g, x-maybe_x)[J] #
            #armijos[J] = (obj(x.double()).double()-obj(maybe_x.double()).double())[J]/util.bdot(g.double(), x.double()-maybe_x.double()).double()[J]
            armijos[J] = (obj(x.float()).float()-obj(maybe_x.float()).float())[J]/util.bdot(g.float(), x.float()-maybe_x.float()).float()[J]
            I = armijos <= GAMMA
            alpha[I] *= decay
            max_armijo = torch.max(armijos)
            count += 1

        x = maybe_x

    # TODO: Maybe change this to a warning.
    print("[WARNING] pnqp warning: Did not converge")
    return x, H_ if n == 1 else H_lu_, If, i
