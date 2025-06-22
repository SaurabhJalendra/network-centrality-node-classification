# Network Centrality Analysis and Node Classification

## Project Overview

This project analyzes the **Email-Eu-core network** dataset from Stanford SNAP to perform comprehensive network analysis and develop machine learning models for node classification. The analysis includes calculating various centrality measures and building SVM-based classifiers to predict department labels.

## Dataset

- **Source**: [Stanford SNAP - Email-Eu-core network](https://snap.stanford.edu/data/email-Eu-core.html)
- **Network**: Email communication network from a European research institution
- **Edges**: 25,571 email communications
- **Labeled Nodes**: 1,005 nodes with department labels (42 departments)

## Assignment Tasks

### 1. Network Diameter Analysis
- Calculate approximate diameter of the network
- Explain computational challenges with exact diameter calculation
- Discuss O(n³) complexity issues for large networks

### 2. Centrality Measures
**Connection-based centrality (who you are connected to):**
- Degree Centrality
- Eigenvector Centrality  
- Katz Centrality
- PageRank

**Bridging centrality (how you connect others):**
- Betweenness Centrality

**Distance-based centrality (how fast you can reach others):**
- Closeness Centrality

**Local structure:**
- Local Clustering Coefficient (LCC)

### 3. Machine Learning Classification
- Develop SVM-based classifier for department prediction
- Use centrality measures as features
- Implement 5-fold cross-validation
- Reserve 20% of data for final testing

## Files Structure

```
network-centrality-node-classification/
├── network_centrality_analysis.ipynb    # Main analysis notebook
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
├── email-Eu-core.txt/
│   └── email-Eu-core.txt              # Network edge list
└── email-Eu-core-department-labels.txt/
    └── email-Eu-core-department-labels.txt  # Node department labels
```

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd network-centrality-node-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook network_centrality_analysis.ipynb
   ```

## Key Features

### Network Analysis
- **Connectivity Analysis**: Identifies largest connected component
- **Diameter Calculation**: Uses approximation algorithms for efficiency
- **Centrality Computation**: Calculates 7 different centrality measures
- **Visualization**: Comprehensive plots and correlation analysis

### Machine Learning
- **Feature Engineering**: Uses centrality measures as classification features
- **Model Selection**: Tests multiple SVM configurations (RBF, Linear, Polynomial)
- **Cross-Validation**: 5-fold CV for robust model evaluation
- **Performance Metrics**: Accuracy, precision, recall, F1-score, confusion matrix

### Methodology Highlights
- **Stratified Sampling**: Maintains class distribution in train/test splits
- **Feature Scaling**: StandardScaler for SVM optimization
- **Approximation Methods**: Efficient computation for large networks
- **Robust Evaluation**: Separate test set for unbiased performance assessment

## Expected Results

The notebook will provide:
1. **Network Statistics**: Nodes, edges, connectivity, diameter
2. **Centrality Rankings**: Top nodes by each centrality measure
3. **Feature Correlations**: Relationships between different centrality measures
4. **Classification Performance**: SVM accuracy and detailed metrics
5. **Feature Importance**: Most predictive centrality measures

## Technical Notes

### Computational Complexity
- **Exact Diameter**: O(n³) - prohibitive for large networks
- **Approximate Diameter**: O(n log n) - practical solution
- **Betweenness Centrality**: O(nm) - uses sampling for large networks

### Dependencies
- **NetworkX**: Network analysis and centrality calculations
- **Scikit-learn**: Machine learning models and evaluation
- **Pandas/NumPy**: Data manipulation and numerical computations
- **Matplotlib/Seaborn**: Visualization and plotting

## Usage

Simply run all cells in the Jupyter notebook sequentially. The analysis is fully automated and will:
1. Load and preprocess the network data
2. Calculate all centrality measures
3. Perform feature engineering
4. Train and evaluate SVM models
5. Generate comprehensive results and visualizations

## Academic Context

This project demonstrates:
- **Network Science**: Understanding of centrality measures and their interpretations
- **Graph Theory**: Practical application of graph algorithms
- **Machine Learning**: Feature engineering and model evaluation
- **Data Science**: End-to-end analysis pipeline from raw data to insights