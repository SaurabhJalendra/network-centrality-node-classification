# Network Centrality Analysis and Node Classification

## Project Description

This project implements a comprehensive network analysis and machine learning pipeline for the Email-Eu-core dataset from Stanford SNAP. The project calculates various network centrality measures and develops SVM-based classifiers to predict node department labels using centrality features.

## Dataset Information

**Source**: Stanford SNAP - Email-Eu-core network
**URL**: https://snap.stanford.edu/data/email-Eu-core.html

**Dataset Characteristics**:
- Network represents email communication within a European research institution
- Total edges: 25,571 email communications
- Total nodes: 1,005 individuals
- Labeled nodes: 1,005 nodes with department assignments
- Number of departments: 42 different departments
- Network structure: Undirected graph representing bidirectional communication

## Assignment Requirements

### 1. Network Diameter Analysis
- Calculate approximate diameter of the network
- Provide detailed explanation of computational issues with exact diameter calculation
- Discuss why approximation methods are necessary for large networks

### 2. Centrality Measures Computation

**2.1 Connection-based Centrality (who you are connected to)**
- Degree Centrality
- Eigenvector Centrality
- Katz Centrality
- PageRank

**2.2 Bridging Centrality (how you connect others)**
- Betweenness Centrality

**2.3 Distance-based Centrality (how fast you can reach others)**
- Closeness Centrality

**2.4 Local Network Structure**
- Local Clustering Coefficient (LCC)

### 3. SVM-based Classification
- Develop Support Vector Machine classifier for department prediction
- Use all calculated centrality measures as feature set
- Implement comprehensive model evaluation

### 4. K-fold Cross-Validation Method
- Divide dataset into 5 equal parts (20% each)
- Reserve 1 part (20%) for final testing
- Use 4-fold cross-validation on remaining 4 parts (80%) for training and validation

## Project Structure

```
network-centrality-node-classification/
├── network_centrality_analysis.ipynb           # Main analysis notebook
├── requirements.txt                            # Python package dependencies
├── README.md                                   # Project documentation
├── email-Eu-core.txt/
│   └── email-Eu-core.txt                      # Network edge list data
└── email-Eu-core-department-labels.txt/
    └── email-Eu-core-department-labels.txt    # Node department labels
```

## Installation Instructions

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd network-centrality-node-classification
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook network_centrality_analysis.ipynb
   ```

## Dependencies

### Core Libraries
- **networkx**: Network analysis and graph algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and evaluation
- **matplotlib**: Basic plotting and visualization
- **seaborn**: Statistical data visualization

### Specific Versions
All package versions are specified in `requirements.txt` for reproducibility.

## Implementation Details

### Network Analysis
- **Data Loading**: Reads edge list and node labels from text files
- **Graph Construction**: Creates NetworkX graph object from edge data
- **Connectivity Analysis**: Identifies and works with largest connected component
- **Centrality Calculation**: Implements all 7 required centrality measures
- **Approximation Methods**: Uses efficient algorithms for computationally expensive measures

### Machine Learning Pipeline
- **Feature Engineering**: Converts centrality measures into feature matrix
- **Data Preprocessing**: Standardizes features using StandardScaler
- **Model Selection**: Tests multiple SVM configurations (RBF, Linear, Polynomial kernels)
- **Cross-Validation**: Implements 4-fold cross-validation as per assignment requirements
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score

### Computational Optimizations
- **Diameter Approximation**: Uses NetworkX approximation algorithm (O(n log n) vs O(n³))
- **Betweenness Sampling**: Applies sampling for betweenness centrality on large networks
- **Error Handling**: Robust implementation with fallback methods for convergence issues
- **Progress Indicators**: Informative output for long-running computations

## Expected Output

### Network Statistics
- Total nodes and edges in the network
- Connectivity analysis and largest component identification
- Approximate network diameter with computational complexity explanation

### Centrality Analysis
- Complete centrality scores for all nodes across 7 different measures
- Statistical summaries (min, max, mean, standard deviation) for each centrality type
- Correlation analysis between different centrality measures
- Visualization of centrality distributions and relationships

### Classification Results
- Cross-validation performance across multiple SVM configurations
- Best model selection based on validation performance
- Final test set evaluation with detailed classification report
- Feature importance analysis showing most predictive centrality measures
- Confusion matrix and performance metrics visualization

### Visualizations
- Network centrality distributions (histograms)
- Correlation heatmap between centrality measures
- Cross-validation performance comparison (bar plots)
- Feature importance rankings

## Technical Specifications

### Computational Complexity
- **Exact Diameter Calculation**: O(n³) using Floyd-Warshall or O(n²m) using repeated BFS
- **Approximate Diameter**: O(n log n) using sampling-based estimation
- **Centrality Measures**: Ranges from O(n+m) for degree centrality to O(nm) for betweenness
- **SVM Training**: Depends on kernel choice and dataset size

### Memory Requirements
- Network storage: O(n+m) for adjacency representation
- Centrality computation: O(n) for storing centrality scores
- Feature matrix: O(n×k) where k is number of features (7 centrality measures)

### Algorithmic Choices
- **NetworkX Library**: Leverages optimized graph algorithms
- **Scikit-learn**: Uses efficient SVM implementations with multiple kernel options
- **Stratified Sampling**: Maintains class distribution in train/test splits
- **Cross-Validation**: Ensures robust model evaluation and selection

## Usage Instructions

### Running the Analysis
1. Ensure all dependencies are installed
2. Open the Jupyter notebook: `network_centrality_analysis.ipynb`
3. Execute cells sequentially from top to bottom
4. The analysis is fully automated and self-contained

### Customization Options
- Modify SVM parameters in the configuration section
- Adjust approximation parameters for large networks
- Change visualization styles and color schemes
- Add additional centrality measures if needed

## Results Interpretation

### Centrality Measures
- **High Degree Centrality**: Nodes with many direct connections
- **High Eigenvector Centrality**: Nodes connected to other well-connected nodes
- **High Betweenness Centrality**: Nodes that act as bridges between different parts of the network
- **High Closeness Centrality**: Nodes that can quickly reach other nodes in the network
- **High Clustering Coefficient**: Nodes whose neighbors are also connected to each other

### Classification Performance
- **Accuracy**: Overall correctness of department predictions
- **Precision/Recall**: Class-specific performance metrics
- **F1-Score**: Balanced measure combining precision and recall
- **Feature Importance**: Which centrality measures are most predictive of department membership

## Academic Context

This project demonstrates proficiency in:
- **Network Science**: Understanding and application of centrality concepts
- **Graph Theory**: Practical implementation of graph algorithms
- **Machine Learning**: Feature engineering, model selection, and evaluation
- **Data Science**: Complete analysis pipeline from raw data to insights
- **Scientific Computing**: Efficient algorithms and computational considerations

## Troubleshooting

### Common Issues
- **Memory Errors**: Reduce network size or use approximation methods
- **Convergence Failures**: Adjust algorithm parameters or use alternative methods
- **Installation Problems**: Ensure Python version compatibility and virtual environment setup
- **Data Loading Errors**: Verify file paths and data format consistency

### Performance Optimization
- Use approximation algorithms for large networks
- Consider sampling methods for computationally expensive measures
- Implement parallel processing where applicable
- Monitor memory usage during computation

## License and Attribution

This project is developed for academic purposes. The Email-Eu-core dataset is provided by Stanford SNAP and should be cited appropriately in any academic work using this analysis.