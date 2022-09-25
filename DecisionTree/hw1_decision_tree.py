"""
@author Sabina Miani
@date 23 September 2022
@assignment CS5350 HW1 Decision trees 
"""

import sys
import copy 
import numpy as np
import pandas as pd

# decision tree node, catagorical data values
class TreeNode:
    def __init__(self) :
        self.feature = None
        self.children = None
        self.depth = -1
        self.isLeaf = False
        self.label = None

    # setters  
    def set_feature(self, feature):
        self.feature = feature

    def set_children(self, children):
        self.children = children
        
    def set_depth(self, depth):
        self.depth = depth

    def set_isLeaf(self, isLeaf):
        self.isLeaf = isLeaf

    def set_label(self, label):
        self.label = label

    # getters 
    def get_depth(self):
        return self.depth

    def get_isLeaf(self):
        return self.isLeaf

        
class ID3:
    ## constructor
    # feature selection: 0 information gain; 1 majority error; 2 gini index
    # max depth maximum depth of decision tree
    def __init__(self, feature_selection = 0, max_depth = 10):
        self.feature_selection = feature_selection
        self.max_depth = max_depth
        
    def set_feature_selection(self, feature_selection):
        self.feature_selection = feature_selection

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth

        
    # param
    # node - {'dataset': dataset, 'features': features, 'dtNode': dtRoot}
    #
    # splits the given node using the max depth and feature selection
    #  function (information gain, majority error, or gini index)  
    def split_dataset(self, node):
        # if node isLeaf, don't split, return {}
        if node['dtNode'].get_isLeaf():
            return {}

        # check if all labels in current dataset are of the same label value
        node_dataset = np.array(node['dataset'])[:,-1]
        common_labels = np.unique(node_dataset)
        most_common = common_labels[0]

        # if maxdepth == depth OR all labels are equivalent (only one unique label), 
        #  don't split. pick label, update current node, return {}
        if self.max_depth == node['dtNode'].get_depth() or common_labels.size == 1:
            node['dtNode'].set_isLeaf(True)
            node['dtNode'].set_label(most_common)
            return {}
        
        # else continue splitting

        # find max gain
        gains, max_gain_feature = self.calc_gain(node['dataset'], node['features'])

        # init new node(s) -- new datasets, feature array
        dataset_split = {}
        
        # split using max gain feature            
        for data_point in node['dataset']:
            # for each data pt, insert it into new array based on feature value
            # remove that feature value from data pt
            feature_index = node['features']['column'].index(max_gain_feature)
            feature_value = data_point[feature_index]
            
            data_pt_split = data_point.copy()
            data_pt_split.remove(feature_value)

            # create new array inside dataset dictionary
            if feature_value not in dataset_split:
                dataset_split[feature_value] = []
                
            dataset_split[feature_value].append(data_pt_split)
            
        # create children nodes
        children = []
        children_nodes = []
        features = copy.deepcopy(node['features'])
        features['column'].remove(max_gain_feature)
        
        for feature_value in dataset_split:
            child = TreeNode()
            # update depth on new nodes
            child.set_depth(node['dtNode'].get_depth() + 1)

            children.append(child)

            # create return structures
            children_nodes.append({'dataset': dataset_split.get(feature_value),
                                   'features': features,
                                   'dtNode': child})

        # update current node
        node['dtNode'].set_feature(max_gain_feature)
        node['dtNode'].set_children(children)

        return children_nodes

        
    # calculates the Gain over a feature
    # returns
    #     gains - array of feature gains
    #     feature - feature of max gain  
    def calc_gain(self, dataset, features):
        feature_props = []
        dataset = np.array(dataset).T
        # generate proportions for each feature 
        for feature_data in dataset:
            ser = pd.Series(feature_data)
            feature_props.append(ser.value_counts(normalize=True))

        feature_gains = []
        ent_S = self.calc_entropy(feature_props[-1])
        for feature in feature_props[:-1]:
            gain = ent_S - np.sum(feature * self.calc_entropy(feature))
            feature_gains.append(gain)

        # index of max gain
        index_gain = np.argmax(np.array(feature_gains))

        return feature_gains, features['column'][index_gain]
        
        
    ## generate decision tree
    # dataset   2d array matrix, each row is an instance
    # features  feature column names and values ('column': [values]}
    def generate_decision_tree(self, dataset, features) :
        Q = []
        dtRoot = TreeNode()
        dtRoot.set_depth(0)
        
        # processing node root
        root = {'dataset': dataset, 'features': features, 'dtNode': dtRoot}
        Q.append(root)
        while len(Q) > 0:
            cur = Q.pop(0)
            nodes = self.split_dataset(cur)
            for node in nodes:
                Q.append(node)

        # return head tree node 
        return dtRoot

    
    # run Entropy calculation based on feature_selection value
    # 0 - Info Gain, 1 - Majority Error, 2 - Gini Index
    # default Info Gain
    def calc_entropy(self, props):
        if self.feature_selection == 1:
            return self.majority_error(props)
        if self.feature_selection == 2:
            return self.gini_index(props)

        # if not ME or GI, run IG
        else:
            return self.information_gain(props) 
        
    def information_gain(self, props):
        return -1 * np.sum(props * np.log(props))

    def majority_error(self, props):
        return 1 - np.max(props)
    
    def gini_index(self, props):
        return 1 - np.sum(np.square(props))

    
# Parses csv file 
def csv_to_data(CSVfile):
    # turn into 2d array of data points  
    data = []
    with open(CSVfile , 'r') as f :
        for line in f :
            # terms = one row of data = one data point
            terms = line.strip().split(',')
            data.append(terms)

    return data


# Parses data.desc file into an object containing a list of label values,
#  a dictionary of attributes and corresponding values, and a list of
#  columns labels. 
def load_data_desc(datafile):
    label_values = {} # {'label': [values]}
    attributes = {} # {'attribute': [values], ...}
    columns = {} # {'column': [values]}
    
    with open(datafile , 'r') as f :
        # remove blank lines and save to data
        data = [line for line in f.readlines() if line.strip()]
        label_values['label'] = data[1].strip().split(',')

        # with known number of attributes - 6
        for att_list in data[3:8]:
            attribute = att_list.strip().split(':')
            attributes[attribute[0]] = attribute[1].strip().split(',|.')

        columns['column'] = data[10].strip().split(',')

    return label_values, attributes, columns


# returns the decision tree over the car data 
def dt_car_data(self, max_depth = 2):
    id3 = ID3()
    id3.set_max_depth(max_depth)
    
    data_file = 'car/data-desc.txt'
    csv_file = 'car/train.csv'
    label_values, attributes, columns = load_data_desc(data_file)
    dataset = csv_to_data(csv_file)

    columns['column'].remove('label')
    dtRoot = id3.generate_decision_tree(dataset, columns)

    return dtRoot


## actually running the functions
print(dt_car_data(sys.argv[1]))
print('finished')
