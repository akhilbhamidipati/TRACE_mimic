def detect_incr_loss(losses, n):
    # Returns True iff the last n values of losses are in non decreasing order.
    if len(losses) < n:
        return False
    else:
        non_dec = True
        last_n_losses = losses[-n:]
        for i in range(len(last_n_losses)-1):
            if last_n_losses[i+1] < last_n_losses[i]:
                non_dec = False
                break
        
        return non_dec