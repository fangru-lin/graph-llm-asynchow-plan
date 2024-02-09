Use files in this folder to generate naturalistic dataset.

First, run ```preprocess.py``` to get preprocessed file with highly-rated scripts excluding defined keywords.

Then, run ```annotate_time_dep.py``` to generate time estimations per step and step dependencies for wikihow dataset.

Last, use ```generate_graph_prompt.py``` to generate directed acyclic graphs for tasks in the annotated wikihow dataset from above steps and proscript dataset.

If you want to validate step dependency annotation quality, run ```proscript_validation.py```.