
========================================================================
Q1. How many geographical units are there?
========================================================================
Geographical units (rows): 20640
Interpretation: Each row is one California census block group (district). There are 20,640 geographical units in this dataset.

========================================================================
Q2. Mean house value among all "ocean_proximity" categories
========================================================================
Overall mean median_house_value: $206,855.82
Interpretation: Across all categories, the average district median house value is about $206,856. This is the baseline against which coastal vs inland differences can be judged.

========================================================================
Q3. Mean house value in each "ocean_proximity" category
========================================================================
ocean_proximity
ISLAND        $380,440.00
NEAR BAY      $259,212.31
NEAR OCEAN    $249,433.98
<1H OCEAN     $240,084.29
INLAND        $124,805.39
Name: median_house_value, dtype: object
Interpretation: ISLAND districts are by far the most expensive on average (~$380k), followed by NEAR BAY, NEAR OCEAN, and <1H OCEAN. INLAND is clearly cheapest (~$125k), roughly half of coastal averages. Note ISLAND has only 5 districts, so that mean is unstable.

========================================================================
Q4. Mean vs median house value by ocean_proximity
========================================================================
                      mean    median  count  mean_minus_median
ocean_proximity                                               
ISLAND           380440.00  414700.0      5          -34260.00
NEAR BAY         259212.31  233800.0   2290           25412.31
NEAR OCEAN       249433.98  229450.0   2658           19983.98
<1H OCEAN        240084.29  214850.0   9136           25234.29
INLAND           124805.39  108500.0   6551           16305.39
Interpretation: For most categories, mean > median, which means the price distribution is right-skewed: a minority of high-value districts pull the mean upward. ISLAND is the exception (mean < median) because of its tiny sample (n=5). The gap between mean and median is largest for NEAR BAY and <1H OCEAN, suggesting more high-end outliers there.

========================================================================
Q5. Histograms
========================================================================
Saved histograms to: /Users/ishpun/Documents/training/Labs/lab1/figures/histograms.png
Interpretation: Four distributions are plotted for households, median_income, housing_median_age, and median_house_value (50 bins each).

========================================================================
Q6. What do the graphs show?
========================================================================
housing_median_age_max: 52.0
housing_median_age_capped_count: 1273
median_house_value_max: 500001.0
median_house_value_capped_count: 965
median_house_value_overall_mean: 206855.81690891474
median_house_value_overall_median: 179700.0
Interpretation:
- households and median_income are right-skewed: many smaller values, a long tail of large districts / high incomes.
- housing_median_age spikes at the maximum (52.0), with 1273 districts at that value. Ages were likely capped (clipped) during data collection.
- median_house_value also spikes at 500,001 (965 districts). Values appear capped around $500,001, so the true prices of the most expensive homes are unknown.
- Magnitudes: house values are in the hundreds of thousands of dollars (mean ~$207k). That scale is expected for California housing, but capping compresses the upper end and can bias models trained on this target.

========================================================================
Q7. Most / least expensive and largest / smallest houses
========================================================================
                 mean_house_value  mean_total_rooms  mean_rooms_per_household     n
ocean_proximity                                                                    
ISLAND                  380440.00           1574.60                      5.66     5
NEAR BAY                259212.31           2493.59                      5.22  2290
NEAR OCEAN              249433.98           2583.70                      5.21  2658
<1H OCEAN               240084.29           2628.34                      5.15  9136
INLAND                  124805.39           2717.74                      5.98  6551
Interpretation:
- Most expensive (by mean median_house_value): ISLAND, then NEAR BAY / NEAR OCEAN / <1H OCEAN. Coastal proximity tracks higher prices.
- Largest houses (rooms per household and total rooms): INLAND tends to have more rooms on average, so 'expensive' and 'large' do not coincide.
- Least expensive: INLAND (~$125k mean).
- Smallest by rooms-per-household: <1H OCEAN (denser coastal living); ISLAND has the fewest total_rooms but only 5 samples.
- Practical takeaway: coastal categories are typically more expensive; INLAND typically offers larger but cheaper housing.

========================================================================
Q8. House quality by ocean_proximity
========================================================================
                 mean_age  mean_total_rooms  ...  mean_rooms_per_household  mean_bedrooms_per_room
ocean_proximity                              ...                                                  
ISLAND             42.400          1574.600  ...                     5.657                   0.273
NEAR BAY           37.730          2493.590  ...                     5.222                   0.213
NEAR OCEAN         29.347          2583.701  ...                     5.206                   0.219
<1H OCEAN          29.279          2628.344  ...                     5.153                   0.218
INLAND             24.272          2717.743  ...                     5.977                   0.203

[5 rows x 5 columns]
Interpretation:
- Age: ISLAND and NEAR BAY homes are older on average; INLAND and <1H OCEAN are younger, suggesting more recent development inland / in suburban coastal belts.
- Rooms: INLAND has higher rooms_per_household, pointing to larger dwellings. Coastal categories are denser with fewer rooms per household.
- Bedrooms-per-room is fairly similar across categories, so room mix is less distinctive than size and age.
- Quality is mixed: coastal areas command higher prices despite older and smaller homes, implying location (not just dwelling size/age) drives value.

========================================================================
Q9. Demographics by ocean_proximity
========================================================================
                 mean_population  mean_households  mean_pop_per_household  mean_median_income     n
ocean_proximity                                                                                    
<1H OCEAN               1520.290          517.745                   3.052               4.231  9136
NEAR BAY                1230.317          488.616                   2.620               4.173  2290
NEAR OCEAN              1354.009          501.245                   2.952               4.006  2658
INLAND                  1391.046          477.448                   3.303               3.209  6551
ISLAND                   668.000          276.600                   2.383               2.744     5
Interpretation:
- Income: highest in <1H OCEAN and NEAR BAY; lowest inland. Income aligns with house values for the main categories.
- Population / households: <1H OCEAN and NEAR OCEAN districts are more populous on average; ISLAND is sparsely populated.
- People per household: highest inland (~3.3), lowest on ISLAND / NEAR BAY, consistent with larger inland family households vs denser or smaller coastal households.
- Overall: coastal categories look wealthier; inland looks more crowded per household and less affluent.

========================================================================
Q10. Other interesting observations / follow-up questions
========================================================================
Missing total_bedrooms: 207
Corr(median_income, median_house_value): 0.688
Unique lat/lon pairs: 12590
Category counts: {'<1H OCEAN': 9136, 'INLAND': 6551, 'NEAR OCEAN': 2658, 'NEAR BAY': 2290, 'ISLAND': 5}
Observations:
- Strong positive correlation between median_income and house value: income is likely the strongest simple predictor.
- 207 districts have missing total_bedrooms; any model using that feature needs imputation.
- ISLAND is almost a curiosity (n=5) and can dominate averages if not handled carefully.
- Capping of age and house value (Q6) will affect regression targets and residuals at the top end.

Follow-up questions you could ask:
- How much of house-value variation remains after controlling for income?
- Are inland 'large house' districts still cheap after adjusting for rooms?
- Do longitude/latitude clusters (Bay Area, LA, San Diego) explain ocean_proximity differences better than the category labels alone?
- How should capped $500,001 values be treated in predictive modeling?
ishpun@M6JC7T4JYK lab1 % 