import pandas as pd
import pytest
from genespectra.gene_classification.classify_genes import GeneClassificationResult


@pytest.fixture
def example_expression_data_long():

    num_cell_types = 10
    exp_lim = 1
    enr_fold = 4

    # Initialize the DataFrame
    data = []

    # 2. Lowly expressed: All values below exp_lim
    gene = "Gene_LowlyExpressed"
    for j in range(num_cell_types):
        data.append([gene, f"CellType_{j+1}", 0.5])

    # 3. Cell type enhanced: One cell type has expression enr_fold times the mean
    gene = "Gene_CellTypeEnhanced"

    data.append([gene, f"CellType_{0+1}", 55])  # enhanced in CellType_1
    data.append([gene, f"CellType_{1+1}", 20])  # enhanced in CellType_1
    data.append([gene, f"CellType_{2+1}", 18])  # enhanced in CellType_1
    data.append([gene, f"CellType_{3+1}", 15])  # enhanced in CellType_1
    data.append([gene, f"CellType_{4+1}", 10])  # enhanced in CellType_1
    data.append([gene, f"CellType_{5+1}", 5])  # enhanced in CellType_1
    data.append([gene, f"CellType_{6+1}", 4])  # enhanced in CellType_1
    data.append([gene, f"CellType_{7+1}", 3])  # enhanced in CellType_1
    data.append([gene, f"CellType_{8+1}", 2])  # enhanced in CellType_1
    data.append([gene, f"CellType_{9+1}", 1])  # enhanced in CellType_1

    # 4. Group enhanced: Few cell types have expression enr_fold times the mean
    # TODO fix group enhanced example - hard to find
    gene = "Gene_GroupEnhanced"
    data.append([gene, f"CellType_{0+1}", 22])  # enhanced in CellType_1 and 2
    data.append([gene, f"CellType_{1+1}", 22])
    data.append([gene, f"CellType_{2+1}", 2])
    data.append([gene, f"CellType_{3+1}", 3])
    data.append([gene, f"CellType_{4+1}", 6])
    data.append([gene, f"CellType_{5+1}", 0])
    data.append([gene, f"CellType_{6+1}", 0])
    data.append([gene, f"CellType_{7+1}", 0])
    data.append([gene, f"CellType_{8+1}", 0])
    data.append([gene, f"CellType_{9+1}", 0])

    # 5. Cell type enriched: One cell type has expression enr_fold times the max of others
    gene = "Gene_CellTypeEnriched"
    max_other_expression = exp_lim
    for j in range(num_cell_types):
        if j == 0:
            data.append(
                [gene, f"CellType_{j+1}", enr_fold * max_other_expression + 1]
            )  # enriched in CellType_1
        else:
            data.append([gene, f"CellType_{j+1}", max_other_expression])

    # 6. Group enriched: Few cell types have expression enr_fold times the max of others
    gene = "Gene_GroupEnriched"
    max_other_expression = exp_lim
    for j in range(num_cell_types):
        if j in [0, 1]:
            data.append(
                [gene, f"CellType_{j+1}", enr_fold * max_other_expression + 1]
            )  # enriched in CellType_1, CellType_2
        else:
            data.append([gene, f"CellType_{j+1}", max_other_expression])

    # 7. Low cell type specificity: Random values
    gene = "Gene_LowSpecificity"
    for j in range(num_cell_types):
        data.append(
            [gene, f"CellType_{j+1}", 2]
        )  # values that don't fit into other categories

    # Create the DataFrame
    df = pd.DataFrame(data, columns=["gene", "group", "expression"])

    # Ensure the DataFrame is ready for classification
    df["expression"] = df["expression"].astype(float)
    df["gene"] = df["gene"].astype(str)
    df["group"] = df["group"].astype(str)

    return df


def expected_gene_classes():

    spec_classes = {
        "gene": [
            "Gene_CellTypeEnhanced",
            "Gene_CellTypeEnriched",
            "Gene_GroupEnhanced",
            "Gene_GroupEnriched",
            "Gene_LowSpecificity",
            "Gene_LowlyExpressed",
        ],
        "spec_category": [
            "cell type enhanced",
            "cell type enriched",
            "group enhanced",
            "group enriched",
            "low cell type specificity",
            "lowly expressed",
        ],
    }

    # Create the DataFrame
    df = pd.DataFrame(spec_classes)

    # Set the 'gene' column as the index
    df.set_index("gene", inplace=True)

    return df


def test_cell_type_enhanced(example_expression_data_long):

    result = GeneClassificationResult.create_from_expression_data_long(
        example_expression_data_long, max_group_n=None, exp_lim=1, enr_fold=4
    )

    assert (
        result[["spec_category"]].loc["Gene_CellTypeEnhanced"]["spec_category"]
        == expected_gene_classes().loc["Gene_CellTypeEnhanced"]["spec_category"]
    )


def test_cell_type_enriched(example_expression_data_long):

    result = GeneClassificationResult.create_from_expression_data_long(
        example_expression_data_long, max_group_n=None, exp_lim=1, enr_fold=4
    )

    assert (
        result[["spec_category"]].loc["Gene_CellTypeEnriched"]["spec_category"]
        == expected_gene_classes().loc["Gene_CellTypeEnriched"]["spec_category"]
    )


def test_group_enhanced(example_expression_data_long):

    result = GeneClassificationResult.create_from_expression_data_long(
        example_expression_data_long, max_group_n=None, exp_lim=1, enr_fold=4
    )

    assert (
        result[["spec_category"]].loc["Gene_GroupEnhanced"]["spec_category"]
        == expected_gene_classes().loc["Gene_GroupEnhanced"]["spec_category"]
    )


def test_cell_type_enhanced(example_expression_data_long):

    result = GeneClassificationResult.create_from_expression_data_long(
        example_expression_data_long, max_group_n=None, exp_lim=1, enr_fold=4
    )

    assert (
        result[["spec_category"]].loc["Gene_CellTypeEnhanced"]["spec_category"]
        == expected_gene_classes().loc["Gene_CellTypeEnhanced"]["spec_category"]
    )


def test_group_enriched(example_expression_data_long):

    result = GeneClassificationResult.create_from_expression_data_long(
        example_expression_data_long, max_group_n=None, exp_lim=1, enr_fold=4
    )

    assert (
        result[["spec_category"]].loc["Gene_GroupEnriched"]["spec_category"]
        == expected_gene_classes().loc["Gene_GroupEnriched"]["spec_category"]
    )


def test_low_specificity(example_expression_data_long):

    result = GeneClassificationResult.create_from_expression_data_long(
        example_expression_data_long, max_group_n=None, exp_lim=1, enr_fold=4
    )

    assert (
        result[["spec_category"]].loc["Gene_LowSpecificity"]["spec_category"]
        == expected_gene_classes().loc["Gene_LowSpecificity"]["spec_category"]
    )


def test_lowly_expressed(example_expression_data_long):

    result = GeneClassificationResult.create_from_expression_data_long(
        example_expression_data_long, max_group_n=None, exp_lim=1, enr_fold=4
    )

    assert (
        result[["spec_category"]].loc["Gene_LowlyExpressed"]["spec_category"]
        == expected_gene_classes().loc["Gene_LowlyExpressed"]["spec_category"]
    )
