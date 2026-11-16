# Deep Q-learning: individual ALU intranet project

This implements project 2417 in `alu-machine_learning`, independently of the
Formative 3 group repository. It was prepared with AI assistance.

`train.py` uses Keras, Gym's `BreakoutDeterministic-v4`, keras-rl's DQNAgent,
SequentialMemory, and an annealed EpsGreedyQPolicy. Four grayscale 84×84 frames
feed a convolutional policy network. Replay stores byte frames; minibatches
are normalized before inference. A separate target network is updated every
10,000 steps. The final full Keras policy model is saved as `policy.h5`.

`play.py` loads that model and evaluates a DQNAgent with GreedyQPolicy.
It displays the game by default and reports unclipped game rewards.
The initial start steps use FIRE to launch the ball; subsequent policy actions
are greedy. A maximum episode length prevents an idle policy running forever.

## Assignment runtime

The intranet specifies Ubuntu 16.04, Python 3.5, NumPy 1.15, Gym 0.17.2,
Keras 2.2.5, keras-rl 0.4.2, Pillow, and h5py. Keras also requires a supported
backend. Use the course's configured backend and Atari environment.

```sh
python3 train.py
python3 play.py
```

Training defaults to 1,750,000 transitions and can take substantial compute.
The 100,000-frame replay limit bounds frame storage to about 706 MB before
Python overhead; increase it if sufficient memory is available.

```sh
python3 train.py --steps 64 --warmup 16 --memory-limit 1000 \
  --output smoke-policy.h5
python3 play.py --policy smoke-policy.h5 --episodes 1 --headless --max-steps 64
python3 -m unittest discover -s tests -v
pycodestyle .
```

The short run verifies execution and serialization. It does not establish a
trained policy that scores above 10. That assessment criterion must be checked
using a completed training run and actual evaluation scores; never infer it
from successful imports, a saved model, or a passing smoke test.

## Reference

Architecture and processing reference:
https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py

The policy is saved as a full model rather than weights alone, so playback
loads its architecture and weights from the same file.

## Validation on 16 November 2026

The real integration tests passed using Python 3.9.6 on macOS ARM, Keras
2.2.5, keras-rl 0.4.2, Gym 0.17.2, atari-py 0.2.6, NumPy 1.23.5,
Theano 1.0.5, and h5py 3.14.0. They exercised real Breakout transitions and
gradient updates, then compared predictions after saving and loading HDF5.
The 64-step greedy evaluation scored 0; this is a runtime smoke test,
not a passing performance result. Pycodestyle 2.4 passed.

The exact Ubuntu/Python 3.5 runtime has not been executed here. The source
parses with Python 3.5 syntax. `requirements.txt` describes the intended
legacy TensorFlow runtime; the local test uses Theano because TensorFlow
1.x has no native macOS ARM wheel.
