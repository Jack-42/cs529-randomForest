from copy import deepcopy


class TreeNode:

    def __init__(self, target, attribute):
        self.target = target
        self.branches = {}
        self.attribute = attribute

    def __init__(self, target):
        self.target = target
        self.attribute = None

    def __init__(self):
        self.target = None
        self.attribute = None
        self.branches = {}

    def addBranch(self, value, node):
        self.branches[value] = node

    def branchValues(self):
        return deepcopy(list(self.branches.keys()))

    def next(self, value):
        if value in self.branches.keys():
            return self.branches[value]
        else:
            return None

    def isLeaf(self):
        return self.attribute is None
