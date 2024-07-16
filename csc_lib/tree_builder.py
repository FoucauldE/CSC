from itertools import combinations

class TreeNode:
    def __init__(self, annotation, dico_anns_filtered, parent=None):
        self.annotation = annotation
        self.dico_anns_filtered = dico_anns_filtered
        self.docs = self.get_docs_containing_combination(parent)
        self.children = []
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

    def get_docs_containing_combination(self, parent):
        if parent is None:
            return set(self.dico_anns_filtered[self.annotation])
        else:
            return parent.docs.intersection(set(self.dico_anns_filtered[self.annotation]))

    def to_dict(self):
        return {
            'annotation': self.annotation,
            'docs': list(self.docs),
            'children': [child.to_dict() for child in self.children]
        }


def build_tree_recursive(annotations, parent, current_depth, max_depth, threshold_number_docs, dico_anns_filtered):
    """Recursively builds a tree representing combinations of annotations where each node represents an annotation"""
    
    if current_depth >= max_depth:
        return

    for annotation in annotations:
        node = TreeNode(annotation, dico_anns_filtered, parent)
        if len(node.docs)>0:
            parent.add_child(node)
            if len(node.docs) >= threshold_number_docs:
                build_tree_recursive([a for a in annotations if a!=annotation], node, current_depth+1, max_depth, threshold_number_docs, dico_anns_filtered)
            
                 

def build_tree(annotations, max_depth, threshold_number_docs, dico_anns_filtered):
    """Builds the root of the tree and starts calls the function to recursively complete it"""

    annotations = list(annotations)
    root = TreeNode(annotations[0], dico_anns_filtered)
    build_tree_recursive(annotations[1:], root, 0, max_depth, threshold_number_docs, dico_anns_filtered)
    return root


def is_subset_flagged(current_combination, flagged_combinations):
    """Checks if any subset of the current combination is already flagged"""

    for i in range(1, len(current_combination)):
        for subset in combinations(current_combination, i):
            if tuple(sorted(subset)) in flagged_combinations:
                return True
    return False


def get_rare_combinations(tree_dict, threshold_nb_docs, max_combination_size, current_combination=None, current_docs=None, results=None, flagged_combinations=None):
    """Recursively retrieves rare combinations of annotations from the tree"""

    if current_combination is None:
        current_combination = set()
    if current_docs is None:
        current_docs = []
    if results is None:
        results = []
    if flagged_combinations is None:
        flagged_combinations = set()

    current_combination.add(tree_dict['annotation'])
    current_docs = tree_dict['docs']

    if len(current_combination) > max_combination_size + 1:
        return # Stop if the combination size is greater than the limit
    
    # Ensure that the combination is present in less documents than the requested threshold and that no subset of the combination has already been flagged
    if len(current_docs) <= threshold_nb_docs and not is_subset_flagged(current_combination, flagged_combinations):
        combination_tuple = tuple(sorted(current_combination))
        if combination_tuple not in flagged_combinations:
            flagged_combinations.add(combination_tuple)
            results.append({'combination': list(current_combination)[1:], 'docs': current_docs})

    # Recursively process each child node in the tree        
    for child in tree_dict.get('children', []):
        get_rare_combinations(child, threshold_nb_docs, max_combination_size, current_combination.copy(), current_docs, results, flagged_combinations)

    return results