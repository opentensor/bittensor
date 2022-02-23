from numpy import zeros_like
import bittensor
import torch
import torch.nn as nn
import torch.nn.functional as F

def update_metagraph_peerweight(metagraph, nucleus, device):
    r"""
    Check for the change in hotkey before and after metagraph sync, 
    update the peer_weight of nucleus accordingingly.
        Args:
            metagraph (:obj:`bittensor.metagraph`, `required`):
                The metagraph to sync.
            nucleus (:obj:`bittensor.neuron.text.nucleus`, `required`):
                The nn.Module class that needs the peerweight to be updated.
            device (:type:`torch.device`)
                The device where peer_weight should be stored. 
    """ 
    old_hotkeys = metagraph.hotkeys
    metagraph.sync()
    new_hotkeys = metagraph.hotkeys
    peer_weight_mean = torch.mean(nucleus.peer_weights)
    chain_growth = max(metagraph.n.item() - nucleus.peer_weights.shape[0], 0)
    nucleus.peer_weights = nn.Parameter(torch.cat([nucleus.peer_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True).to(device)]))
    
    for i, (old_hotkey, new_hotkey) in enumerate(zip(old_hotkeys, new_hotkeys)):
        if old_hotkey != new_hotkey:
            with torch.no_grad():
                nucleus.peer_weights[i] = peer_weight_mean
    
def joining_context(return_ops, topk_weights, responses):
    """
    Joins response embbedings depending on the return codes 
        Args:
            return_ops  (:type:`pytorch.LongTensor`, `required`):
                The return codes of dendrite call return ops.
            topk_weights  (:type:`pytorch.FloatTensor`, `required`):
                The topk weights selected for joining
            responses  (:type:`pytorch.FloatTensor`, `required`):
                The embeddings that sent by the peers

        Returns:
            output (:type:`pytorch.FloatTensor``, `required):
                The joinned output embedding using the weights
            joining_uids  (:type:`pytorch.LongTensor`, `required`):
                The uids used to create output
    
    """
    joining_uids= torch.where( return_ops == bittensor.proto.ReturnCode.Success )[0]
    joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 ) 
    output = torch.zeros( (responses[0].shape[0], responses[0].shape[1], bittensor.__network_dim__))
    for index, joining_weight in enumerate( joining_weights ):
        output += responses[joining_uids[index]]* joining_weight
    return output, joining_uids

def jacobian(y, x, create_graph=False,hessian =False): 

    """
    Calulates the Jacobian from the inputs; adapted from : https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
        Args:
            y  (:type:`pytorch.FloatTensor`, `required`):
                The loss function
            x  (:type:`pytorch.FloatTensor`, `required`):
                The parameters to differentiate loss by
            create_graph  (:type:`bool`, `optional`):
                If we should pass parameter to grad function
            hessian (:type:`bool`, `optional`):
                turn on if the calculation is for a hessian instead of jacobian

        Returns:
            jacobian (:type:`pytorch.FloatTensor``, `required):
                The jacobian matrix which contains the partial differentials 
    
    """
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
            try:
                grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
            except Exception as e:
                return torch.zeros(y.shape + x.shape)
            jac.append(grad_x.reshape(x.shape))                                                           
            grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)     

def fisher_score_approximation(loss, peer_weights, ):
    """
    Uses the jacobian function to approximate the saliency scores, currently not used

        Args:
            loss  (:type:`pytorch.Loss`, `required`):
                The remote target loss 
            peer_weights  (:type:`pytorch.FloatTensor`, `required`):
                The peer weights which was used to calculate the loss

        Returns:
            validator_scores (:type:`pytorch.FloatTensor``, `required):
                A saliency score that approximates the fisher information of each peer
    
    
    """
    peer_weights_d1 = jacobian(loss, peer_weights, create_graph=True)
    if peer_weights_d1 == None: return torch.ones_like( peer_weights ) # None if no grad w.r.t the chain weights.
    peer_weights_d2 = jacobian(peer_weights_d1, peer_weights, hessian=True)
    second_order = (peer_weights_d2.detach() * (torch.outer(-peer_weights.detach(),-peer_weights.detach()))/2 ).sum(dim=1)
    first_order = (peer_weights_d1.detach()* -peer_weights.detach())
    validator_scores =  second_order + first_order
    return validator_scores

def partial_contexts(return_ops, topk_uids, topk_weights, responses):
    """
    Creates the partial contexts which are used to calculate the shapley scores 

        Args:
            return_ops  (:type:`pytorch.LongTensor`, `required`):
                The return codes of dendrite call return ops.
            topk_weights  (:type:`pytorch.FloatTensor`, `required`):
                The topk weights selected for joining
            responses  (:type:`pytorch.FloatTensor`, `required`):
                The embeddings that sent by the peers

        Returns:
            partial_context (:type:`Dictionary``, `required):
                A dict containing all of joinned contexts with a single peer masked out 
    
    """
    partial_context = {}
    with torch.no_grad():
        for i, uid in enumerate(topk_uids):
            partial_return_ops = return_ops.clone()
            if partial_return_ops[i] != bittensor.proto.ReturnCode.Success:
                pass
            else:
                partial_return_ops[i] = bittensor.proto.ReturnCode.NoReturn
            partial_context[uid.item()], _ = joining_context(partial_return_ops, topk_weights, responses)
    return partial_context
