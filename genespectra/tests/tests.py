import pandas as pd

# a test data for gene classification
test_data = pd.DataFrame({
    'gene': ['Gene1', 'Gene1', 'Gene1', 'Gene1', 'Gene1', 'Gene2', 'Gene2', 'Gene2', 'Gene2', 'Gene2', 'Gene3', 'Gene3',
             'Gene3', 'Gene3', 'Gene3', 'Gene4', 'Gene4', 'Gene4', 'Gene4', 'Gene4', 'Gene5', 'Gene5', 'Gene5', 'Gene5',
             'Gene5', 'Gene6', 'Gene6', 'Gene6', 'Gene6', 'Gene6', 'Gene7', 'Gene7', 'Gene7', 'Gene7', 'Gene7'],
    'tissue': ['Tissue1', 'Tissue2', 'Tissue3', 'Tissue4', 'Tissue5', 'Tissue1', 'Tissue2', 'Tissue3', 'Tissue4',
               'Tissue5', 'Tissue1', 'Tissue2', 'Tissue3', 'Tissue4', 'Tissue5', 'Tissue1', 'Tissue2', 'Tissue3',
               'Tissue4', 'Tissue5', 'Tissue1', 'Tissue2', 'Tissue3', 'Tissue4', 'Tissue5', 'Tissue1', 'Tissue2',
               'Tissue3', 'Tissue4', 'Tissue5', 'Tissue1', 'Tissue2', 'Tissue3', 'Tissue4', 'Tissue5'],
    'expression': [188, 150, 119, 28, 163, 107, 145, 53, 139, 133, 57, 140, 1, 136, 176, 13, 59, 83, 195, 86, 56, 3, 91,
                   120, 26, 56, 3, 90, 75, 26, 4, 3, 2, 1, 0]
})
