# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DreamDojo is a generalist robot world model built on top of NVIDIA's Cosmos-Predict2 foundation model. It learns action-conditioned video generation from large-scale human egocentric video data and supports post-training to specific robot embodiments (GR-1, Unitree G1, AgiBot, YAM). The codebase also includes a distillation pipeline to produce a fast causal student model for real-time teleoperation at 10 FPS.

## Code Edit Conventions

When making code edits, mark the start of each change with `### CLAUDE ###` plus a short explanation of the change, and the end with `### END CLAUDE ###`. Additionally, instead of overwriting or replacing existing code, please comment out the existing code and add your change after it.  

Example:
```python
### CLAUDE ### Fix off-by-one error in frame indexing
frames = frames[1:]
### END CLAUDE ###
```