# LOGICS
LOGICS: Learning optimal generative distribution for designing de novo chemical structures

# Contact
Bongsung Bae (bsbae402@gist.ac.kr)

Hojung Nam* (hjnam@gist.ac.kr)

*Corresponding Author

# Abstract
In recent years, the field of computational drug design has made significant strides in the development of artificial intelligence (AI) models for the generation of de novo chemical compounds with desired properties and biological activities, such as enhanced binding affinity to target proteins. These high-affinity compounds have the potential to be developed into more effective therapeutics for a broad spectrum of diseases. Due to the lack of data required for the training of deep generative models, however, some of these approaches have fine-tuned their molecular generators using data obtained from a separate predictor. These studies demonstrated that generative models can generate structures with the desired target properties at a high frequency; nevertheless, it is uncertain whether the diversity of generations and chemical space covered correspond to the distribution of the desired targeted molecules. In this study, we present a novel generative framework, LOGICS, a framework for Learning Optimal Generative distribution Iteratively for designing target-focused Chemical Structures. We address the exploration-exploitation dilemma (exploring new options or exploiting current knowledge) and tackle this problem by applying experience memory and the layered tournament selection process for a sophisticated fine-tuning process. The proposed method was applied to the binding affinity optimization of two target proteins of different protein classes, κ-opioid receptors and PIK3CA, and the quality and the distribution of the generative molecules were evaluated. The results showed that LOGICS outperforms competing state-of-the-art models and generates more diverse de novo chemical structures with optimized properties. 
