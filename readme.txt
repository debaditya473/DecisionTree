Outputs:
    Tree Structure, and Predicted output at each node
    Accuracy on Test Set
    Predicted Values and Input Values

Requirements:
    pandas (for data handling)
    numpy (for numerical calculations)
    Counter from collections is also imported but it is an inbuilt function

Methodoly:

    The tree is based on Node objects, where every node is a decision point.
    Information Gain is calculated using the entropy of these nodes.
    Once the feature with the maximum information gain is found, we create child nodes.
    These Child Nodes correspond to all the possiblities of the feature,
    for example number of people being 2, 4, 6 will create 3 child Nodes.

How to use:
    to create a tree, a root node must be made like this:
        root = Node(Data, 0, target_name=target_name)
    here, Data must be a pandas Dataframe. target_name is the target column to be predicted.
    To train and print the tree, the following lines need to be used.
        root.train_tree()
        root.print_tree()

    For prediction, the following code must be used:
        predicted = root.predict(Data)
    predicted will be a list of predictions

It is recommended to redirect the output of the program to a file 
for better visibility as the tree is very large in size.