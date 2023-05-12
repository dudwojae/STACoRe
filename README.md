# STACoRe: Spatio-Temporal and Action-Based Contrastive Representations for Reinforcement Learning in Atari
This repository provides the code to implement the STACoRe.

# File Description
    .
    ├── agents
    │   └── stacore_agent.py      # The agent script to select actions and optimize polices
    ├── environment                     
    │   ├── env.py                # Atari environment
    ├── networks                     
    │   ├── stacore_network.py    # Deep neural networks code needed to train STACoRe
    ├── tasks                     
    │   ├── stacore.py            # Code to train or test STACoRe
    │   ├── stacore_test.py       # Code used when testing in stacore.py
    ├── utils                    
    │   ├── args.py               # Arguments needed to run the code
    │   ├── automatic.py          # Upper confidence bound (UCB) algorithm code for automatic data augmentation
    │   ├── layers.py             # Deep neural networks initialization
    │   ├── loss.py               # STACoRe loss
    │   ├── memory.py             # Prioritized experience replay
    │   ├── mypath.py             # The path to saver or load the file
    └── run_stacore.py            # The main run code
    
# Installation
The python version we used is 3.6.13.
~~~
pip install -r requirements.txt
~~~

# Train STACoRe
~~~
python run_stacore.py
~~~
