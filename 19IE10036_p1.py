# Roll Number 19IE10036
# Name Debaditya Mukhopadhyay
# Assignment number 1

import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(
        self,
        Data: pd.DataFrame,
        depth=None,
        rule = None,
        node_type = None,
        target_name = None
    ) -> None:
        
        self.Data = Data
        self.Y = Data[target_name]
        Y = self.Y.values.tolist()

        self.X = Data.drop(target_name, axis=1)
        self.depth = depth if depth else 0

        self.features = list(self.X.columns)
        
        # total count and count by classification
        self.length = len(Y)
        self.counts = Counter(Y)
        self.entropy = self.get_entropy( Data[target_name])

        # Sorting the counts and saving the final prediction of the node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Getting the last item and aving to object attribute.
        # This node will predict the class with the most frequent class
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        self.yhat = yhat 

        # child nodes
        self.children = []

        # Default values for splits
        self.best_feature = None 

        # Rule for spliting 
        self.rule = rule if rule else ""
        self.node_type = node_type if node_type else 'root'

        self.target_name = target_name

        pass

    def get_entropy(self, target_col: pd.Series):
        p_i = target_col.value_counts()/target_col.shape[0]
        entropy = np.sum(-p_i*(np.log2(p_i)))
        return entropy

    def information_gain(self, split_attribute_name):
        Counter(self.Data[split_attribute_name].values.tolist())

        new_entropy = 0
        counts1 = []
        entropies1 = []

        entropies1 = []
        for j in self.Data[split_attribute_name].unique():
            target_col = []
            #print(j)
            for k in self.Data[split_attribute_name].index:
                #print(features[split_attribue_name].iloc[k])
                if self.Data[split_attribute_name].iloc[k] == j:
                    #print(features[split_attribue_name].iloc[k])
                    target_col.append(self.Data[self.target_name].iloc[k])
            
            target_col_DF = pd.DataFrame(target_col, columns = ['class'])
            #print(j)
            counts1.append(len(target_col))
            entropies1.append(self.get_entropy(target_col_DF['class']))
        
        #Calculate the weighted entropy  
        counts = np.array(counts1)
        entropies = np.array(entropies1)
        weighted_entropy = np.sum(np.multiply(counts, entropies))/np.sum(counts)
        # print(entropies)

        #Calculate the information gain  
        info_gain = self.entropy - weighted_entropy
        return info_gain

    def best_split(self) -> str:
        
        best_gain = 0
        best_feature = None

        for i in self.features:
            gain = self.information_gain(i)
            if gain > best_gain:
                best_gain = gain
                best_feature = i

        return best_feature
    
    def train_tree(self):
        """
        Recursive method to create the decision tree
        """
        df = self.Data.copy()

        # first we get the best feature to split on
        best_feature = self.best_split()

        if best_feature is not None:
            self.best_feature = best_feature

            # we create child nodes

            for attribute in self.Data[best_feature].unique():
                child_df = df[df[best_feature] == attribute].copy()
                child_df.reset_index(drop=True, inplace=True)
                
                child = Node(
                    Data=child_df,
                    depth=self.depth + 1,
                    node_type = "child",
                    rule= f"if {best_feature} = {attribute}",
                    target_name=target_name
                    )
                
                self.children.append(child)
            
            for child in self.children:
                child.train_tree()
        else:
            pass
    
    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 

        if self.best_feature is not None:
            indentation = "|" + "     |" * (self.depth-1)
            spaces = indentation + "------"
            
            if self.node_type == 'root':
                print("Root")
            else:
                print(f"{spaces} {self.rule} | Predicted class: {self.yhat}")
            # print(f"{' ' * const}   | Entropy of the node: {round(self.entropy, 2)}")
            # print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")



    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info()

        for child in self.children:
            child.print_tree()
    
    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        while(len(cur_node.children) != 0):
            attribute = values.get(cur_node.best_feature)
            for child in cur_node.children:
                if child.rule == f"if {cur_node.best_feature} = {attribute}":
                    cur_node = child
                    break
        
        return cur_node.yhat


if __name__ == "__main__":
    
    target_name = 'class'
    
    # training data
    Data = pd.read_csv("Data/project1 1.data", sep= ',', header=None, names=['price', 'maint', 'doors', 'persons', 'lug_boot', 'safety', target_name])

    root = Node(Data, 0, target_name=target_name)

    # training
    root.train_tree()
    root.print_tree()

    # test data
    Data = pd.read_csv("Data/project1_test.data", sep= ',', header=None, names=['price', 'maint', 'doors', 'persons', 'lug_boot', 'safety', target_name])
    X = Data.drop(target_name, axis=1)
    Y = Data[target_name]
    Y = Y.values.tolist()
    
    # predicting on test data 
    predicted = root.predict(Data)

    # Accuracy

    print("\n ACCURACY \n")
    for i in range(len(Y)):
        acc = 0
        count = 0
        if predicted[i] == Y[i]:
            acc += 1
            count += 1
    print(f"Accuracy = {acc/count*100}%")


    # Printing out predictions
    print("\n PREDICTIONS \n")

    print("Predicted    Actual")
    for i in range(len(Y)):
        print(f'{predicted[i]}  {Y[i]}')
    
    
    
