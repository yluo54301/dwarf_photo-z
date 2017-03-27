# Where to Start

## Getting Data
### COSMOS
I've included a version of the COSMOS catalog in this repo (`data/COSMOS_reference.sqlite`).

### HSC
Unfortunately the HSC dataset is too large to fit as a single file in this directory (and the exact feature set that I want to include is still in flux). So for now you'll have to download it yourself (see `get_data.ipynb`)

## Get an Overview of the Data
The next place to run `HSC_COSMOS_completeness.ipynb`. This'll show you a sample of the HSC and COSMOS footprints. It'll also show you about breakdown of COSMOS in (photo-z, stellar mass) space.  Finally, it'll display the estimated completeness of HSC, relative to the known COSMOS objects.

## Begin Filtering out Galaxies
Before we get to fancy image-capable machine learning techniques, we'll probably need to reduce the size of the dataset. Ideally we'd keep galaxies that are low mass + low redshift, while filtering out the rest.

To see the first attempts on this, check out `HSC_COSMOS_filtering.ipynb`