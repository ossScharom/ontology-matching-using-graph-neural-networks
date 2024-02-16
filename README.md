# Ontology Matching using Graph Neural Networks (OMUNET)
[![DOI](https://zenodo.org/badge/757712669.svg)](https://zenodo.org/doi/10.5281/zenodo.10659449)

The biomedical domain requires experts and information systems to have a common understanding
of terms and concepts used during the exchange of information. Ontologies are a common way to
describe these terms and concepts. Research within the biomedical domain constantly pushes the
boundaries of knowledge and thus changes the semantics of concepts or enriches ontologies with new
concepts. This dynamic environment requires a constant curation of existing ontologies. Ontology
matching enables an automation of this curation, allowing different ontologies to integrate. In recent
years, the field of ontology matching has seen a rise in machine learning-based matching systems.
Systems exploiting the textual representations of concepts within the ontologies dominate this new
trend, motivating us to propose Ontology Matching using Graph Neural Networks (OMUNET) a
system that enriches textual representations with graph structural information. Our system mod-
els the ontology matching problem as a relation prediction between two graphs. Additionally, we
investigate techniques for negative node sampling during the network training. We evaluate our
system on the 2023 version of the Bio-ML track of the Ontology Alignment Evaluation Initiative.
Our results show that our proposed system cannot outperform existing systems and that one of our
negative node sampling strategies exploiting the graph structure of an ontology is only performing
slightly better than a baseline.

## Contact

Jerome Wuerf (jw20qave@uni-leipzig.de)
