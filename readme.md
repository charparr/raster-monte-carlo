# A Monte Carlo-esque method for sampling raster data

## Motivation
Your raster data is large, oddly shaped, and has areas where data is not available. You have a nifty function that describes or analyzes the data - but how do you apply this function to a sample of the raster in a way that
   a.) is faithful to the scope of the full dataset
   b.) accounts for spatial variability and bias
   c.) is not confounded by "no data", and
   d.) is reproducible

## Aim
Generate random subsamples of a raster dataset that in ensemble ensure a robust spatial sampling of the raster and meet the above conditions.

## Scope
Geotiff raster data.
