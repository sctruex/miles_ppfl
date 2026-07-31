from numpy.random import Generator, default_rng
from privacy_utils.search import find_mult_search, find_rounds_search
# from search import find_mult_search, find_rounds_search
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import matplotlib.pyplot as plt
from typing import Union


def get_noise(approach, priv_param, accountant_params, np_rng: Generator):
    if approach == 'dropout':
        # let the noise multiplier as is
        accountant_params['noise_multiplier'] = priv_param
        return priv_param, generate_eps_lst(find_eps(**accountant_params), np_rng, **accountant_params)
    eps_lst = generate_eps_lst(priv_param, np_rng, **accountant_params)
    return find_mult(min(eps_lst), **accountant_params), eps_lst


def generate_eps_lst(
        eps_loc: Union[int, float], np_rng: Generator,
        total_clients: int, min_eps: Union[int, float],
        eps_distrib_scale: Union[int, float], sort_eps_lst: bool, **_):
    eps_lst = np_rng.normal(loc=eps_loc, scale=max(
        2, eps_distrib_scale*eps_loc), size=total_clients)
    # if private - switch in accountant params
    if sort_eps_lst:
        eps_lst.sort()
    eps_lst[eps_lst < min_eps] = min_eps
    return eps_lst


def find_eps(rounds, clients_per_round, total_clients, local_epochs, noise_multiplier, target_delta,
             batch_size, len_data, **_):
    expected_epochs = local_epochs*rounds*clients_per_round/total_clients
    # event = make_event_from_param(rounds, q, noise_multiplier)
    # accountant = dp_accounting.rdp.RdpAccountant()
    # return accountant.compose(event).get_epsilon(target_delta)
    return compute_dp_sgd_privacy_lib._compute_dp_sgd_example_privacy(
        expected_epochs,
        noise_multiplier,
        target_delta,
        False,
        poisson_subsampling_probability=batch_size / (len_data/total_clients),
        accountant_type=compute_dp_sgd_privacy_lib.AccountantType.RDP,
    )


def find_mult(eps, rounds, total_clients, clients_per_round, target_delta, MIN_MULT, MAX_MULT, **_):
    # noise_multiplier = 1e-5         # some minimum
    # step = .1           # search step
    # # TO-DO: ????
    q = clients_per_round/total_clients
    # event = make_event_from_param(rounds, q, noise_multiplier)
    # accountant = dp_accounting.rdp.RdpAccountant()
    # curr_spend =  accountant.compose(event).get_epsilon(target_delta)
    # # ALTER TO BE BINARY SEARCH
    # while curr_spend > epsilon:
    #     print(f"currently spending {curr_spend} instead of {epsilon} with noise multiplier {noise_multiplier}")
    #     noise_multiplier += step
    #     event = make_event_from_param(rounds, q, noise_multiplier)
    #     accountant = dp_accounting.rdp.RdpAccountant()
    #     curr_spend =  accountant.compose(event).get_epsilon(target_delta)
    # return noise_multiplier
    return find_mult_search(eps, rounds, q, target_delta, MIN_MULT, MAX_MULT)


# def make_event_from_param(candidate_rounds, q, noise_multiplier):
#     gaussian_event = dp_accounting.GaussianDpEvent(noise_multiplier)
#     sampled_event = dp_accounting.PoissonSampledDpEvent(q, gaussian_event)
#     composed_event = dp_accounting.SelfComposedDpEvent(
#         sampled_event, candidate_rounds)
#     return composed_event


def find_avail_rounds(target_delta, rounds, local_epochs, noise_multiplier, batch_size, eps, q, client_data_len, **_):
    # for r in range(1,rounds+1):
    #     event = make_event_from_param(r, q, noise_multiplier)
    #     accountant = dp_accounting.rdp.RdpAccountant()
    #     eps_spend=accountant.compose(event).get_epsilon(target_delta)
    #     print(f"participating in {r} rounds results in spend of {eps_spend} (target is {eps})")
    #     if eps_spend > eps:
    #         return r-1
    # return rounds
    # print(f"Beginning available client rounds search:")
    return find_rounds_search(target_delta=target_delta,
                              rounds=rounds,
                              local_epochs=local_epochs,
                              noise_multiplier=noise_multiplier,
                              eps=eps,
                              q=q,
                              client_data_len=client_data_len,
                              batch_size=batch_size)


def test_eps_gen(np_rng: Generator):

    runs = 30
    eps_norm_scales = [i for i in range(6)]  # [.5, 2, 5]
    # eps_loc = 10. # 5. 1.
    loc = [10.]
    y_lim = [.5]
    len_data = 60000  # say mnist data
    num_clients = 30
    rounds = 100
    selections = ['tiered', 'uniform']
    distributions = ['normal', 'uniform']
    buckets = [.1, .15, .2, .25, .3]
    num_buckets = 5
    bucket_size = num_clients//len(buckets)
    min_eps = 0.01

    for selection in selections:
        for distribution in distributions:
            for val in range(len(loc)):
                eps_loc = loc[val]
                print(eps_loc)
                y_limit = y_lim[val]
                print(y_limit)
                final_scale_average = []
                labels = []
                for eps_norm_scale in eps_norm_scales:
                    average_rejection = [[] for i in range(rounds)]
                    if distribution == "normal":
                        labels.append(
                            f"loc={int(eps_loc)}, scale={int(eps_norm_scale)}")
                    elif distribution == "uniform":
                        labels.append(
                            f"low=-{int(4*eps_norm_scale +eps_loc)},high={int(4*eps_norm_scale+eps_loc)}")
                    else:
                        labels.append("?")
                    # labels.append(f"scale={eps_loc}")
                    for i in range(runs):
                        if distribution == "normal":
                            eps_lst = np_rng.normal(
                                loc=eps_loc, scale=eps_norm_scale, size=num_clients)
                        elif distribution == "uniform":
                            eps_lst = np_rng.uniform(
                                low=-4*eps_norm_scale + eps_loc, high=4*eps_norm_scale+eps_loc, size=num_clients)
                        else:
                            eps_lst = [eps_loc]*num_clients
                        # eps_lst = exponential(scale=eps_loc,size=30)

                        eps_lst[eps_lst < min_eps] = min_eps
                        eps_lst.sort()

                        avail_rounds = []

                        if selection == 'uniform':
                            q = [(num_clients*.2)/num_clients] * num_clients
                        if selection == 'tiered':
                            q = []
                            for i in range(num_buckets-1):
                                q += bucket_size * \
                                    [buckets[i] *
                                        min(1., (num_clients*.2)/bucket_size)]
                            last_bucket_size = num_clients-len(q)
                            q += last_bucket_size * \
                                [buckets[-1] *
                                    min(1., (num_clients*.2)/last_bucket_size)]
                        for i in range(len(eps_lst)):
                            avail_rounds.append(find_avail_rounds(target_delta=1e-5, rounds=rounds, noise_multiplier=1., eps=eps_lst[i], q=q[i],
                                                                  client_data_len=(len_data//num_clients), batch_size=128))
                        print(f"available rounds: {avail_rounds}")
                        rejection_metrics = []
                        for r in range(rounds):
                            sample_clients = [i for i in range(
                                len(eps_lst)) if np_rng.random() < q[i]]
                            num_reject = 0
                            for client in sample_clients:
                                if avail_rounds[client] <= r:
                                    num_reject += 1
                            if len(sample_clients) > 0:
                                rejection_metrics.append(
                                    num_reject/len(sample_clients))
                        print("rejection_metrics", rejection_metrics)
                        for r in range(len(rejection_metrics)):
                            average_rejection[r].append(rejection_metrics[r])
                    final_avg = []
                    for n in range(len(average_rejection)):
                        sum = 0
                        for m in average_rejection[n]:
                            sum += m
                        final_avg.append(sum/runs)
                    final_scale_average.append(final_avg)

                print(final_scale_average)
                bottom, top = plt.ylim()
                plt.ylim(top=y_limit)  # set y_limit
                for i in range(len(final_scale_average)):
                    plt.plot(
                        range(len(final_scale_average[i])), final_scale_average[i], label=labels[i])
                plt.xlabel('Rounds')
                plt.ylabel("Average Rejections(%)")
                plt.legend()
                plt.savefig(
                    f"eps_distrib_scale_{distribution}_{selection}_{eps_loc}.png")
                plt.close()


if __name__ == "__main__":
    test_eps_gen(default_rng())
