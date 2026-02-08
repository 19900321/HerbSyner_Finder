# syn_net_CVA is a novel network-based framework for synergistic ingredients discovery from herb medicines

<p align="center">
  <img src="Figure1.png" alt="Overview Figure" width="800">
  <br>
  <em>Overview of network framework. (A) Systematic comparison of five prescriptions for CVA on their therapeutic targets. (B) Identify synergistic ingredients of CVA prescriptions through a comprehensive combination landscape (C) Experimental validation of synergistic ingredients.</em>
</p>

 Herbal medicine presents a promising alternative, offering enhanced synergistic efficacy and reduced side effects through combined herbal formulations. 
 
 However, the synergistic mechanisms of action (MOAs) of these herbal medicines remain largely unexplored. 
 
 Given the complexity of herbal systems, it is impractical to evaluate all possible drug/ingredient pairs experimentally.
 
To develop an innovative network model to consider synergistic ingredients or herbs in herbal medicine and identify their MOAs in the treatment of CVA, we developed an innovative network model to prioritize synergistic ingredients or herbs using advanced network proximity and community analysis. 

Our framework constructs a combinational atlas for each herbal medicine, quantifying interactions among herb-disease, ingredient-disease, herb-herb, and ingredient-ingredient relationships, thereby identifying synergistic components and their MOAs. 

Our network model offers novel insights into combinational strategies for treating CVA and has successfully identified effective herbal drug combinations from herbal medicines, specifically berberine and luteolin.



# HerbSyner_Finder

A General Tool for Synergistic Ingredient Discovery in Herbal Medicine

## Overview

HerbSyner_Finder is a computational pipeline designed to discover synergistic ingredient combinations in herbal medicine using network-based approaches and community detection algorithms. This tool helps identify key therapeutic modules and potential synergistic herb combinations for specific diseases through advanced network analysis.

## Features

- **Network-based distance calculation**: Compute distances between herbs and ingredients in biological networks
- **Community detection**: Identify communities in herb-ingredient-disease networks using Louvain algorithm
- **Key module identification**: Detect synergistic modules with main therapeutic effects
- **ADMET property filtering**: Filter ingredients based on drug-likeness properties
- **Multi-disease support**: Analyze multiple disease targets simultaneously
- **Visualization tools**: Generate comprehensive synergy landscapes

## Installation

### Prerequisites

- Python 3.10.11
- Required Python packages:

```bash
pip install communities==2.1.1
pip install networkx==2.8.4
pip install scikit-learn==1.2.2
pip install pandas numpy matplotlib seaborn openpyxl
