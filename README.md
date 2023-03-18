## Code of our paper "Bi-directional Learning of Logical Rules with Implicit Type Constraints for Knowledge Graph Completion"
## Prerequisites

 * Python 3.8
 * pytorch==1.10.0


### Datasets
We used Kinship, UMLS, WN18RR, FB15k-237, YAGO26K906, AirGraph, FamilyIC in our experiments.

| Datasets           | Download Links                                                       |
|--------------------|----------------------------------------------------------------------|
| Kinship            | https://github.com/DeepGraphLearning/RNNLogic                                |
| UMLS               | https://github.com/DeepGraphLearning/RNNLogic                            |
| WN18RR             | https://github.com/DeepGraphLearning/RNNLogic   |
| FB15k-237          | https://github.com/DeepGraphLearning/RNNLogic   |
| YAGO26K906         | https://github.com/Rainbow0625/TyRuLe   |
| AirGraph           | https://github.com/Rainbow0625/TyRuLe   |
| FamilyIC           | This work.   |


## Use examples
For link prediction, you can run our models as bellow:

``bash run.sh`` for Kinship, UMLS, WN18RR and FB15k-237.

``bash run_type.sh`` for YAGO26K906 and AirGraph.

``bash run_FamilyIC.sh`` for FamilyIC.

For triple classification, you can run our models as bellow:

``python triple_classification.py`` for head queries.

``python triple_classification_inv.py`` for tail queries.



