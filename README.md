# LOGICS: Learning optimal generative distribution for designing de novo chemical structures
Bae, B., Bae, H. & Nam, H. LOGICS: Learning optimal generative distribution for designing de novo chemical structures. J Cheminform 15, 77 (2023). https://doi.org/10.1186/s13321-023-00747-3

# Contact
Bongsung Bae (bsbae402@gist.ac.kr)

Haelee Bae (haeleeeeleah@gm.gist.ac.kr)

Hojung Nam* (hjnam@gist.ac.kr)

*Corresponding Author

# Abstract
In recent years, the field of computational drug design has made significant strides in the development of artificial intelligence (AI) models for the generation of de novo chemical compounds with desired properties and biological activities, such as enhanced binding affinity to target proteins. These high-affinity compounds have the potential to be developed into more potent therapeutics for a broad spectrum of diseases. Due to the lack of data required for the training of deep generative models, however, some of these approaches have fine-tuned their molecular generators using data obtained from a separate predictor. While these studies show that generative models can produce structures with the desired target properties, it remains unclear whether the diversity of the generated structures and the span of their chemical space align with the distribution of the intended target molecules. In this study, we present a novel generative framework, LOGICS, a framework for Learning Optimal Generative distribution Iteratively for designing target-focused Chemical Structures. We address the exploration—exploitation dilemma, which weighs the choice between exploring new options and exploiting current knowledge. To tackle this issue, we incorporate experience memory and employ a layered tournament selection approach to refine the fine-tuning process. The proposed method was applied to the binding affinity optimization of two target proteins of different protein classes, κ-opioid receptors, and PIK3CA, and the quality and the distribution of the generative molecules were evaluated. The results showed that LOGICS outperforms competing state-of-the-art models and generates more diverse de novo chemical structures with optimized properties. 
