"""A minimal tool-using agent and the sandbox it runs in.

The agent exists so the framework can be exercised on multi-step,
tool-calling trajectories with real ground truth (did the tests pass? were
the test files touched?). It is deliberately small: a ReAct loop over five
file and test tools inside a per-task temporary directory.
"""
