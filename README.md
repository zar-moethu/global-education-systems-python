# Global Higher Education Clustering with Python: Clustering Analyis Algorithm


### Project Overview
This project aims to understand how international student mobility is related to the level of globalisation and the structure of higher education systems across countries in 2019.
We apply unsupervised learning, specifically clustering analysis, to group countries based on similarities in:

**Education system characteristics**, including:  
- ISCED Participation Levels (Levels 5 to 8)
- QS World University Rankings (Top 50, 100, 500, 1000)

**Globalisation indices**, including:
- Political Globalisation Index (KOFPoGI)
- Economic Globalisation Index (KOFEcGI)
- Social Globalisation Index (KOFSoGI)

#### Why do we use unsupervised learning algorithm?
- Unsupervised learning algorithm is used to group similar data points into clusters without any predefined labels or categories.
- In this project, it helps uncover natural groupings of countries that show similar patterns in education systems and level of globalisation indices.
- This allows us to explore and compare international education trends in a data-driven way.

### Data Dictionary 
The following features/variables are used in this analysis:

| Variable Name            | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `country_x`              | Country name                                     |
| `InboundRatio`           | Inbound mobility rate for international students (% of tertiary enrolment) |
| `InternationalStudentsNO`| Total number of international students enrolled                             |
| `KOFPoGI`                | Political Globalisation Index (from KOF Globalisation Index)                |
| `KOFEcGI`                | Economic Globalisation Index                                                |
| `KOFSoGI`                | Social Globalisation Index                                                  |
| `ISCED5 Percentage`      | Participation in short-cycle tertiary education (ISCED Level 5)             |
| `ISCED6 Percentage`      | Participation in Bachelor's level tertiary education (ISCED Level 6)        |
| `ISCED7 Percentage`      | Participation in Master's level tertiary education (ISCED Level 7)          |
| `ISCED8 Percentage`      | Participation in Doctoral level tertiary education (ISCED Level 8)          |
| `top 50 count`           | Number of universities ranked in the QS Top 50                              |
| `top 100 count`          | Number of universities ranked in the QS Top 100                             |
| `top 500 count`          | Number of universities ranked in the QS Top 500                             |
| `top 1000 count`         | Number of universities ranked in the QS Top 1000                            |
| `WESP`                   | World Economic Situation and Prospects 2021 grouping or economic tier       |

**Note:**  
Other decomposed KOF indicators (e.g. KOFGIdf, KOFTrGIdj, etc.) exist in the full dataset but are not used in this study for simplicity and relevance. This project focuses on understanding how globalisation and higher education metrics relate to inbound student mobility and university rankings.

### Data Cleaning and Preparation
- Checked for missing values across selected features/variables
- Selected features/variables based on the defined data dictionary.
- Removed rows with missing values to ensure data integrity for clustering
- Standardised all numerical features using StandardScaler to ensure equal contribution during clustering.
