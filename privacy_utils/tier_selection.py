

def create_tiers(client_lst, total_clients, num_buckets):
    """
    Partitions clients in client_lst by their privacy budgets into 5 buckets. Currently eps_lst does not
    correlate to any specific client. eps_lst is in order and is assigned to client at same index.
    
    Parameters:
        - client_lst: list of client ids 
        - total_clients: number of total clients in client_lst

    Return:
        - tier_lst: list of 5 lists containing client_ids
    """
    prev = 0
    index = total_clients/num_buckets
    tier_lst = [[],[],[],[],[]]
    for count in range(5):
        tier_lst[count] = client_lst[prev:index]
        prev = index
        index = index+total_clients/num_buckets
    # if total_clients % 5 is not zero
    if prev != total_clients:
        tier_lst[-1] += client_lst[index:]
    return tier_lst

# select bucket randomly       # higher prob of selecting higher budget bucket
# pass bucket and tier_lst to Server
# Server creates sample_client lst by selecting clients from given bucket in tier_lst

# note: what if k is greater than number of clients in selected bucket?