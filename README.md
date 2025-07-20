## Global Higher Education Clustering Analysis with Python: Unsupervised Learning Algorithm

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
| `country_x`              | Country name                                                                |
| `InboundRatio`           | Inbound mobility rate for international students                            |
| `InternationalStudentsNO`| Total number of inbound international students                              |
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
| `WESP`                   | World Economic Situation and Prospects|

**Note:**  
Other decomposed KOF indicators (e.g. KOFGIdf, KOFTrGIdj, etc.) exist in the full dataset but are not used in this study for simplicity and relevance. This project focuses on understanding how globalisation and higher education metrics relate to inbound student mobility and university rankings.

### Data Cleaning and Preparation
- Checked for missing values across selected features/variables
- Selected features/variables based on the defined data dictionary.
- Removed rows with missing values to ensure data integrity for clustering
- Standardised all numerical features using StandardScaler to ensure equal contribution during clustering.

### Exploratory Data Analysis (EDA)

To understand the data structure and identify patterns before clustering, we performed:

**Summary statistics:**  

Across the 42 countries studied, several key patterns emerge in relation to international student engagement:

- On average, 9% of all students enrolled in tertiary education are international students, though this varies widely — from as low as 0.22% to as high as 35.29%, depending on the country. The number of international students also differs significantly: while the average is 117,317 students, some countries host nearly one million (maximum = 976,562), while others have only a few thousand.
- Bachelor’s level (ISCED 6) dominates tertiary education systems, with an average share of 45.24%, while Master’s level (ISCED 7) follows at 14.23%.
- Doctoral-level participation (ISCED 8) is the smallest segment, averaging just 2.10%. Most countries fall between 0.00% and 5.15%, suggesting this level remains specialised and selective due to longer study duration, competitive entry and limited institutional capacity or funding.
- Overall globalisation indices are relatively high across the sample, with political = 84.95, economic = 71.98 and social = 79.98 on average.

<img width="790" height="408" alt="Screenshot 2025-07-20 at 13 54 10" src="https://github.com/user-attachments/assets/862a6d1f-e139-40d7-9038-2d353d5b6c34" />

**Visualisation - Percentage Distribution of Education Levels by Country:**

- Compared participation rates in ISCED Levels 5 to 8 across countries using a vertical stacked bar chart to show the distribution of tertiary education participation by level.

- The visual shows that some countries have a more balanced distribution across all ISCED levels, while others are heavily concentrated in certain tiers (e.g. bachelor’s or master’s level).

- Australia and Sweden, both classified as developed countries (WESP), show high overall tertiary participation with significant contributions across multiple ISCED levels, indicating a more developed and layered higher education system.

- Turkey shows relatively even participation across short-cycle tertiary (e.g. associate degree or foundation degree), bachelor’s and master’s levels (ISCED Levels 5, 6 and 7 respectively), but comparatively limited engagement at the doctoral level (ISCED 8).

<img width="1490" height="590" alt="output_9_0" src="https://github.com/user-attachments/assets/0a46fba4-0544-4245-93fa-eecc3faeb075" />

**Visulisation - Percentage Distribution of University Rankings by Country:**  

Compared the number of universities ranked in the Top 50, Top 100, Top 500, and Top 1000 across countries.

Used a vertical stacked bar chart to show how each country contributes to global university rankings at different tiers.

🔍 Key Insight:
Countries like the United States, United Kingdom, and Australia dominate the top tiers, with multiple universities in the Top 50 and Top 100, highlighting their global academic reputation and strong higher education infrastructure.

<img width="987" height="1189" alt="output_10_0" src="https://github.com/user-attachments/assets/73229a78-f7e7-438d-955b-0bb234024aa2" />

