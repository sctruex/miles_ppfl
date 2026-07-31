import matplotlib.pyplot as plt

def plot_l2_norms():

    fin = open('l2_norm_vals.txt','r')
    val = 0
    norms,num_norms,curr_norm, num_in_curr = [0 for _ in range(80)],0,0,0
    run = 1
    for line in fin:
        if line[0] != "*" and line[0] != "-": # still collecting values for training round
            val = float(line[:-1])
            curr_norm += val
            num_in_curr += 1
        elif line[:-1] == "*"*200: # training round ended, average data and add to norms
            print(f"attempting to add norm number {num_norms} after last val {val}")
            norms[num_norms] = (curr_norm-val) / (num_in_curr-1)
            num_norms += 1
            curr_norm, num_in_curr = 0,0
        elif line[0] == "-": # experimental run has ended, plot l2_norm trajectory
            plt.plot(range(num_norms), norms[:num_norms])
            plt.savefig(f"l2_norm_trajectory_run{run}.png")
            plt.close()
            run += 1
            num_norms = 0

plot_l2_norms()