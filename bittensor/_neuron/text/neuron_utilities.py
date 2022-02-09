import bittensor
import torch
import torch.nn.functional as F


def joining_context(return_ops, topk_weights, responses):
    # ---- Join based on weights ----
    joining_uids= torch.where( return_ops == bittensor.proto.ReturnCode.Success )[0]
    print(joining_uids)
    joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 ) 
    output = torch.zeros( (responses[0].shape[0], responses[0].shape[1], bittensor.__network_dim__))
    for index, joining_weight in enumerate( joining_weights ):
        output += responses[joining_uids[index]]* joining_weight
    
    return output, joining_uids

def jacobian(y, x, create_graph=False,hessian =False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)): 
        if hessian ==True and flat_y[i].item() == 0:
            grad_x = torch.zeros_like(x)
            jac.append(grad_x.reshape(x.shape)) 
            pass
        else:
            grad_y[i] = 1.
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
            jac.append(grad_x.reshape(x.shape))                                                           
            grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)     

def fisher_score_approixmation(loss, peer_weights, ):
    peer_weights_d1 = jacobian(loss, peer_weights, create_graph=True)
    if peer_weights_d1 == None: return torch.ones_like( peer_weights ) # None if no grad w.r.t the chain weights.
    peer_weights_d2 = jacobian(peer_weights_d1, peer_weights, hessian=True)
    second_order = (peer_weights_d2.detach() * (torch.outer(-peer_weights.detach(),-peer_weights.detach()))/2 ).sum(dim=1)
    first_order = (peer_weights_d1.detach()* -peer_weights.detach())
    validator_scores =  second_order + first_order
    return validator_scores

def partial_contexts(return_ops, topk_uids, topk_weights, responses):
    partial_context = {}
    with torch.no_grad():
        for i, uid in enumerate(topk_uids):
            partial_return_ops = return_ops.clone()
            partial_return_ops[i] = bittensor.proto.ReturnCode.NoReturn
            partial_context[uid], _ = joining_context(partial_return_ops, topk_weights, responses)
    return partial_context
