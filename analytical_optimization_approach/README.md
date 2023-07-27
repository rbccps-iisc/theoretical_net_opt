## Done:

1.) GEKKO() code example to optimize the Bit Rate achieved for a single snapshot of time (aposteriori, i.e., after
    observing the signal strengths).
2.) GEKKO() code example to optimize the Bit Rate achieved for a Finite Horizon (aposteriori, i.e., after
    observing the signal strengths over the entire horizon of time). This is not realistic but can be used
    to generate 'near optimal' policies by replacing randomness with expected values.


## To Do:

1.) (Incomplete) Add environment and perform Value/Policy Iteration methods for Finite and Infinite Horizon (deterministic vs data-driven)
    - Transition probabilities are given, and there is no randomness in signal strength.
    - Transition probabilities learned, and there is no randomness in signal strength.
    - Transition probabilities are given, and there is randomness in signal strength (distribution given).
    - Transition probabilities learned, and there is randomness in signal strength (distribution given).
    - Transition probabilities learned, and there is randomness in signal strength (distribution learned).

2.) Use GEKKO() to do the above analytically using a suitable optimization objective (Closed Loop vs. Open Loop)

