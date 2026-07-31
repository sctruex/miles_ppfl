import dp_accounting
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

# function also in privacy.py but couldn't get importing for some reason
# fix and remove later


def make_event_from_param(candidate_rounds, q, noise_multiplier):
    gaussian_event = dp_accounting.GaussianDpEvent(noise_multiplier)
    sampled_event = dp_accounting.PoissonSampledDpEvent(q, gaussian_event)
    composed_event = dp_accounting.SelfComposedDpEvent(
        sampled_event, candidate_rounds)
    return composed_event


def find_rounds_search(target_delta, rounds, local_epochs, noise_multiplier, eps, q, client_data_len, batch_size):
    min_r, max_r = 1, rounds
    while min_r <= max_r:
        mid = ((max_r-min_r)//2) + min_r

        # event = make_event_from_param(mid, q, noise_multiplier)
        # accountant = dp_accounting.rdp.RdpAccountant()
        # eps_spend=accountant.compose(event).get_epsilon(target_delta)
        expected_epochs = local_epochs * mid * q
        eps_spend = compute_dp_sgd_privacy_lib._compute_dp_sgd_example_privacy(
            expected_epochs,
            noise_multiplier,
            target_delta,
            False,
            poisson_subsampling_probability=batch_size / client_data_len,
            accountant_type=compute_dp_sgd_privacy_lib.AccountantType.RDP,
        )

        if eps_spend > eps:
            max_r = mid-1
        else:
            min_r = mid+1
    return min(min_r, max_r)


def find_mult_search(eps, rounds, q, target_delta, min_mult, max_mult):
    min_mult, max_mult = min_mult, max_mult
    while min_mult < max_mult:
        mid = ((max_mult-min_mult)//2) + min_mult
        event = make_event_from_param(rounds, q, mid)
        accountant = dp_accounting.rdp.RdpAccountant()
        eps_spend = accountant.compose(event).get_epsilon(target_delta)
        # print(f"using multiplier {mid} results in spend of {eps_spend} (target is {eps})")
        if eps_spend > eps:
            min_mult = mid+1
        else:
            max_mult = mid-1
    print(f"Using multiplier {max(min_mult,max_mult)}, (target is {eps})")
    return max(min_mult, max_mult)
