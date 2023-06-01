import pandas as pd
import pickle
from scipy.stats import chi2_contingency

class DecisionTree:
    def __init__(self, max_depth=None, alpha=0.05):
        self.model_name = 'DecisionTree'
        self.tree = {}
        self.route = []
        self.max_depth = max_depth
        self.alpha = alpha

    def chisquare(self, x, column1, y):
        df = x.copy()
        df['target'] = y
        target = 'target'
        ct = pd.crosstab(df[f'{column1}'], df['target'])
        chi2, p, dof, expected = chi2_contingency(ct)
        return p

    def best_feature(self, x, y, bf=''):
        df = x.copy()
        df['target'] = y
        target = 'target'
        p_values = {}
        listofcolumns = list(df.columns)
        listofcolumns.remove(target)
        if bf not in listofcolumns:
            listofcolumns = listofcolumns
        elif bf in listofcolumns:
            listofcolumns.remove(bf)
        for i in listofcolumns:
            p = self.chisquare(x, i, y)
            if p <= self.alpha:
                p_values[i] = p
        if not p_values:
            return None
        feature = pd.DataFrame(p_values, index=['p_value']).T
        return feature['p_value'].idxmin()

    def hypothesis_testing(self, x, y, bf):
        df = x.copy()
        df['target'] = y
        target = 'target'
        ct = pd.crosstab(df[f'{bf}'], df['target'])
        chi2, p, dof, expected = chi2_contingency(ct)
        if p <= self.alpha:
            return True  # reject null hypothesis
        else:
            return False  # fail to reject null hypothesis
        
    def split(self, x, y, path=None, depth=0):
        df = x.copy()
        df['target'] = y
        target = 'target'
        if path is None:
            path = []
        bf = self.best_feature(x, y)
        if bf is None or (self.max_depth is not None and depth >= self.max_depth):
            # If no best feature is found or the max depth is reached, return the majority class of the parent node
            majority_class = df[target].mode()[0]
            self.route.append((path, majority_class))
            print(f"{path} => {majority_class}")
            return None
        else:
            node = {'feature': bf, 'children': {}}
            path.append(bf)
            for val in df[bf].unique():
                node['children'][val] = self.split(df[df[bf] == val].drop(columns=[bf, target]), 
                                                    df[df[bf] == val][target],
                                                    path=path,
                                                    depth=depth+1)
            path.pop()
            return node
        
    def predict(self, x):
        predictions = []
        for i, row in x.iterrows():
            for path, decision in self.route:
                a = 0
                for branch in path:
                    col, val = branch.split('->')
                    if str(row[col]) != val:
                        a = 1
                        break
                if a == 0:
                    predictions.append(decision)
                    break
            else:
                counts = {}
                for path, decision in self.route:
                    last_branch = path[-1]
                    _, last_val = last_branch.split('->')
                    if last_val not in counts:
                        counts[last_val] = 0
                    counts[last_val] += 1
                mode = max(counts, key=counts.get)
                predictions.append(mode)
        return predictions