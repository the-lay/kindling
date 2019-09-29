### kindling
_n. material that can be readily ignited 🎇

#### Work is heavily in progress, nothing to see yet, move along ✋👮👉 

#### Idea
The idea is to have a framework on top of PyTorch that can make research experimentation easier:
- Reduce generic boilerplate and allow you to focus on the most important
- Make experiment code more descriptive and short, ideally so small that all can be in one file
- Track projects and experiments
    - Save projects' and associated experiments' properties
    - Save all hyperparameters
    - Save git revision and/or .patch of code that was used for the experiment 
    - Save all actions that were used in the experiment process
- Handle training/validation process
    - Fully flexible (user-defined) training and validation
    - Pretty and useful feedback
    - Support for metrics (associated with model) and callbacks (associated with training/validation process)
    - Automatically sync metrics with tensorboard for realtime visualization
- Increase productivity with network prototyping
    - Provide easily customizable building blocks for different models
    (**Important**: the project is being developed primarily for my own use (I focus on CNNs for 3D volumetric data),
    so not all features have same priority :) When the project will be stable enough to be used, PRs will be super welcome.)
- Moonshot idea: generate reports of experiments automatically?
- Moonshot idea: have GUI, maybe as a jupyter notebook addon?

#### Overview of classes
- Projects have Experiments
- Each Experiment should track everything that you did to your Model (a log: each event start and finish)
- Model is a combination of pytorch network, hyperparameters and functions defining training, validation and testing
- You can train/validate with Trainer
- Trainer connects Dataset, Model, pytorch optimizer, Callbacks (like scheduler, checkpointer etc.) and Metrics

Example: check out `target_example_usage.py`