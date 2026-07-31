import matplotlib.pyplot as plt
from typing import Union


class StepClipDecay:
    def __init__(self,
                 initial_clip: Union[float, int],
                 decay_rounds: int,
                 decay_rate: Union[float, int],
                 min_clip: Union[float, int],
                 first_round_clip: Union[float, int] = 0,
                 **_):
        """A clip scheduler that uses an exponential decay schedule.

        The schedule coontains a 1-arg callable function that produces a decayed clip
        value when passed the current training round. This can be useful for changing
        the l2_norm_clip in a differentially private optimizer across different rounds of training.
        It is computed as:

        ```python
        if round == 1:
            return min_clip
        else:
            return max (
                initial_clip * decay_rate ^ (round // decay_rounds), 
                min_clip
            )
        ```

        Args:
            initial_clip:       A Python float. The initial clip value.
            decay_rounds:       A Python float. Must be non-negative. 
                                See the decay computation above.
            decay_rate:         A Python float. The decay rate.
                                See the decay computation above.
            min_clip:           A Python float. The minimum clip value to be used.
            first_round_clip:   A Python float. The clip value to be used in the first round.
                                Value of 0 indicates use of computation above.
        """
        self._initial_clip = initial_clip
        self._decay_rate = decay_rate
        self._decay_rounds = max(decay_rounds, 1)
        self._first_round_clip = first_round_clip
        self._min_clip = min_clip

    def get_round_clip(self, curr_round):
        """
        Compute the decayed clip value for a given round.
        """
        if curr_round == 0 and self._first_round_clip > 0:
            return self._first_round_clip
        return max(self._initial_clip * (self._decay_rate ** ((curr_round+1) // self._decay_rounds)),  self._min_clip)

    def __str__(self):
        return f"\n\tinitial_clip:{self._initial_clip},\n\tdecay_rate: {self._decay_rate},\n\tdecay_rounds: {self._decay_rounds},\n\tfirst_round_clip: {self._first_round_clip}"


class ExponentialClipDecay:
    def __init__(self,
                 initial_clip: Union[float, int],
                 power: Union[float, int],
                 min_clip: Union[float, int],
                 first_round_clip: Union[float, int] = 0,
                 **_):
        """A clip scheduler that uses an exponential decay schedule.

        The schedule coontains a 1-arg callable function that produces a decayed clip
        value when passed the current training round. This can be useful for changing
        the l2_norm_clip in a differentially private optimizer across different rounds of training.
        It is computed as:

        ```python
        if round == 1:
            return min_clip
        else:
            return max (
                initial_clip / round^power, 
                min_clip
            )
        ```

        Args:
            initial_clip:       A Python float. The initial clip value.
            power:              A Python float. Must be non-negative. 
                                See the decay computation above.
            min_clip:           A Python float. The minimum clip value to be used.
            first_round_clip:   A Python float. The clip value to be used in the first round.
                                Value of 0 indicates use of computation above.
        """
        self._initial_clip = initial_clip
        self._first_round_clip = first_round_clip
        self._min_clip = min_clip
        self._power = max(0, power)

    def get_round_clip(self, curr_round):
        """
        Compute the decayed clip value for a given round.
        """
        if curr_round == 0 and self._first_round_clip > 0:
            return self._first_round_clip
        # return self._initial_clip * self._decay_rate ** (curr_round // self._decay_rounds)
        # print(f"\tgoing to return a clip of {max(self._initial_clip / ((curr_round+1)**self._power), self._min_clip)}")
        return max(self._initial_clip / ((curr_round+1)**self._power), self._min_clip)

    def __str__(self):
        return f"\n\tinitial_clip: {self._initial_clip},\n\tpower: {self._power},\n\tmin_clip: {self._min_clip},\n\tfirst_round_clip: {self._first_round_clip}"


def test_clip_vals():
    num_rounds = 50
    x_vals = [x for x in range(num_rounds)]
    clip_scheduler = StepClipDecay(
        initial_clip=1,
        decay_rounds=5,
        decay_rate=.6,
        min_clip=0.001
    )
    y_vals = [clip_scheduler.get_round_clip(x) for x in x_vals]

    plt.plot(x_vals, y_vals)
    plt.savefig("test_clip_params.png")
    print(y_vals)
    print(clip_scheduler)


if __name__ == "__main__":
    test_clip_vals()
