# The NIST Diverse Communities Data Excerpts

The Diverse Communities Data Excerpts are tabular demographic data from real household in the U.S. The Excerpts serve as benchmark data for the [SDNist v2: Deidentified Data Report Tool](https://github.com/usnistgov/SDNist/) and are the taget data for the [2023 NIST Collaborative Researtch Cycle](https://pages.nist.gov/privacy_collaborative_research_cycle/).

The Excerpts are designed to be a resource to investigate performance of data synthesizers and similar privacy-enhancing technologies when applied to data collected from three communities (Boston area, Dallas-Forth Worth area, and US national) with radically different demographics. The data are drawn from real records released in the 2019 [American Community Survey](https://www.census.gov/programs-surveys/acs), a product of the US Census Bureau. We also include 2018 sample as a control set. Currently, sdnist is only configured to run against 2019 files. Any analysis of 2018 data is entirely up to the user. 

## Purpose and design

The Diverse Communities Data Excerpts were developed to address a recurring problem identified in the NIST Differential Privacy Synthetic Data Challenge, NIST Differential Privacy Temporal Map Challenge, and the HLG-MOS Synthetic Data Test Drive. Specifically, if the data are too complex, it’s hard to tell for sure what’s going on with it. These data are designed to support a deeper and more formal understanding of algorithm behavior on real human data.

The Diverse Communities Data Excerpts were designed with input from US Census Bureau experts in adaptive sampling design (consider [Coffey 2020](https://doi.org/10.1093/jssam/smz026)) and leveraged previous work on geographical differences in CART-modeled synthetic data to identify Public Use Microdata Areas (PUMAs) with challenging distributions (see Appendix B, [Abowd 2021](https://ui.adsabs.harvard.edu/link_gateway/2021arXiv211013239A/arxiv:2110.13239)).

Effectively, we attempt to provide a model-sized version of the problem that we were trying to understand. Like a ship in a bottle&mdash;pieces small enough that they can be assembled by hand&mdash;so that we can start to see how they fit together.

The requirements we sought to meet included the following:
- A feature set that is interesting, with complexity representative of real communities yet compact enough to allow for rigorous, penetrating analysis
- Geographic regions that include enough (but not an overwhelming amount) individuals and sufficient complexity to investigate relationships between features
- Geographic regions whose typical feature values vary radically to explore performance on diverse data sets

## Citing NIST Diverse Communities Data Excerpts
If you publish work that utilizes the NIST Diverse Communities Data Excerpts, please cite the resource. Citation recommendation:  
```
Task C., Bhagat K., Streat D., Simpson A., and Howarth G.S. (2023), 
NIST Diverse Communities Data Excerpts, National Institute of Standards and Technology, 
https://doi.org/10.18434/mds2-2895  

@misc{task_nist_2022,
	author = {Task, Christine and Bhagat, Karan and Damon, Streat and Howarth, Gary},
	collaborator = {Howarth, Gary},
	doi = {10.18434/MDS2-2895},
	publisher = {National Institute of Standards and Technology},
	title = {{NIST} {Diverse} {Community} {Excerpts} {Data}},
	url = {https://data.nist.gov/od/id/mds2-2895},
	year = {2023}}
```



## Data set description

We are currently offering three geographic regions, each contained within its own directory, containing the data and a description of each PUMA within data we call "postcards".

All of the data have identical schema of 22 features that are provided in a data dictionary with accompanying notes.

### Community descriptions
See the "postcards" and data dictionaries in each respective directory for more detailed information:
- `national`: 27254 records drawn from 20 PUMAs from across the United States
- `massachusetts`: 7634 records drawn from five PUMAs of communities from the North Shore to the west of the greater Boston, Massachusetts area.
- `texas`:  9276 records drawn from six PUMAs of communities surrounding Dallas-Fort Worth, Texas area


## Usage guidance

### Area definitions
Each of the data sets comprise data from multiple PUMAs. PUMAs are designated by the US Census Bureau and designed to be "non-overlapping, statistical geographic areas that partition each state or equivalent entity into geographic areas containing no fewer than 100,000 people each." [See the Census page for more information](https://www.census.gov/programs-surveys/geography/guidance/geo-areas/pumas.html).

### Data partitions
The data have been broken into three separate data sets: `massachusetts` (MA), `texas` (TX), and `national`. Algorithms can be run on each data set separately, or all three together (they use the same schema). The MA and TX data were chosen to be more homogeneous and regionally focused (North Shore of Boston and southwest Fort Worth), while the PUMAs included in the national data were chosen to represent an array of complex and diverse communities around the country.

Postcard descriptions are included in each data set folder and provide a snapshot introduction to each of the geographies included in the data set. When your algorithm performs well or poorly on a specific PUMA, you might find it interesting to check on the defining characteristics of that PUMA. For instance, the veteran disability variable (DVET) may have a more interesting distribution near military communities. These postcards make a good starting point for learning more about the real communities your algorithms are running on.


### Data dictionary and feature notes
There is a data dictionary in JSON format included with the data.

**Null values:** 
Different systems encode and handle null (empty) values differently; we use the literal character 'N' to represent null values throughout these data. Null values appear in columns representing categorical and numerical data. For synthesizing numerical features (PINCP, POVIP), you will likely want to convert 'N' to null or a numerical code before processing. The presence of a string in a column may cause a programing environment to interpret all values of said column as strings. To avoid this in the Pandas package of Python, for example, specify 'N' as the null value during CSV import: `df = pandas.read_csv('./national2019.csv', na_values='N')`. **Be sure to convert null values back to 'N' before submitting to the sdnist evaluator.** Again in Pandas, `df.to_csv('./synthetic_national2019.csv', na_rep='N')`.


**Feature notes:**
- Geography Feature: PUMA (see note above)
- Demographic Features: AGEP (age), SEX (sex), HISP (hispanic origin), RAC1P (race)
- Household Features: MSP (marital status), NOC (number of children), NPF (family size), HOUSING_TYPE (single housing unit or group quarters such as dorms, barracks, or nursing homes), OWN_RENT (own or rent housing)
- Regional Features: DENSITY (population density of PUMA, useful for distinguishing urban, suburban, and rural PUMAs)
- Financial Features: PINCP (personal income, including wages and investment), PINCP_DECILE (person’s income discretized as a 10% percentile bin relative to the income distribution in their PUMA), POVPIP (household income as a percentage of the poverty threshold for that household (dependent on family size and other factors&mdash;households with income more than 5x the property line are given the value 501)
- Industry and Education Features: EDU (highest educational attainment&mdash; it’s useful to consider this in combination with AGEP), INDP_CAT (general category of work, for example, agriculture or retail), INDP (specific category of work&mdash;this is an example of a categorical feature with very many possible values)
- Disability Features: DREM (cognitive difficulty, binary feature), DEYE (vision difficulty, binary feature), DEAR (hearing difficulty, binary feature), DPHY (walking/movement difficulty, binary feature), DVET (disability due to military service&mdash;this is a rating value rather than a binary feature)
- Weights: Person’s weight and housing weight can be used to generate a full sample. To make a full population persons data estimate, duplicate each record the number of times indicates by its PWGHT feature.
- There exist some deterministic relationships between features (consistency constraints):
   * NPF & NOC (NPF must be greater than NOC). 
   * HOUSING_TYPE & OWN_RENT (OWN_RENT is null for group quarters).
   * AGEP dependence on many features: children (~ AGEP < 18) will have null values for many adult features, such as financials, and a maximal possible value for educational attainment dependent on age)
- Other Notes: INDP_CAT & INDP (INDP_CAT is a partitioning of the codes in INDP), PUMA & DENSITY (DENSITY is the population density of a given PUMA),  PINCP & PINCP_DECILE (PINCP_DECILE is a percentile aggregation of PINCP by PUMA)


### Suggested feature combinations
**If you want …**
- a few small binary features, consider using OWN_RENT, DPHY, DEYE, DEAR, or SEX
- a numerical feature, consider using PINCP, POVPIP, or AGEP
- an ordinal feature, consider EDU, PINCP_DECILE, NOC, or NPF
- a large cardinality categorical feature, consider using INDP
- a smaller cardinality categorical feature, consider using INDP_CAT, HISP, or MSP
- diverse distributions, look at RAC1P, DVET, SEX, MRST, or RENT_OWN 

### Reduced-size feature sets
These smaller feature sets can be used to compare table-based deidentification methods such as cell suppression or differentially private histograms that are not intended for sparse data distributions on larger feature sets. Three possible feature sets are provided, derived from Aniruddha Sen’s internship work exploring distributional variations among subpopulations and their impact on tabular deidentification approaches. 

If you use these feature sets, consider evaluating utility separately on the indicated partitions (in addition to overall utility) to learn more about the interaction of privacy, utility, and equity for your deidentification method. Different subpopulations in the data can have different patterns of correlations over the same set of features, and this can cause deidentification methods to perform better or worse for these groups. The selected feature sets have interesting distributional variances with respect to their suggested partitioning features. Your deidentification method may be more or less sensitive to this effect; you can use these feature sets to explore its behavior.   


Feature Set 1: AGEP,* RAC1P, MSP, SEX, HOUSING_TYPE, OWN_RENT, DVET, DEYE

Suggested partitioning feature: Race 


Feature Set 2: AGEP,* SECTOR, RAC1P, SEX, HOUSING_TYPE, OWN_RENT, DVET, DEYE

Suggested partitioning feature: Sector
 

Feature Set 3: AGEP,* SEX, MSP, PINCP_DECILE, HISP, DVET, DREM, DEYE

Suggested partitioning feature: Sex + Disability 

*(For studying tabular/histogram methods you will likely want to bin AGE into a set of ranges: [0-10], [11-20], etc.)* 

## Credits 

- [Christine Task](mailto:christine.task@knexusresearch.com) - Project technical lead - christine.task@knexusresearch.com
- [Karan Bhagat](https://github.com/kbtriangulum) - Contributor
- [Aniruddha Sen](https://www.linkedin.com/in/senaniruddha/) - Contributor
- [Damon Streat](https://www.linkedin.com/in/damon-streat-244106190) - Contributor
- [Ashley Simpson](https://www.linkedin.com/in/ashleysimpson33) - Contributor
- [Gary Howarth](https://www.nist.gov/people/gary-howarth) - Project PI - gary.howarth@nist.gov



