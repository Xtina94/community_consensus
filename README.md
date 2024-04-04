To run the code:
1. Clone the repo in your local storage
2. Create a folder called `Outputs` within your github repo
3. Run `python main.py <DEALBREAKER>`

Where `DEALBREAKER` is a number among [0, 1, 2]. As per code docstrings, the three values correspond to:

> `0 -- All conditions are respected and the algorithm should lead to community consensus`

> `1 -- if the minimum degree is not respected but the graph is excess robust`

> `2 -- if the graph is not (0, f + 1)- excess robust but the minimum degree is respected`

At every new run, the content of `Outputs` gets erased, so make sure to extract the needed results if you want to store them permanently.
