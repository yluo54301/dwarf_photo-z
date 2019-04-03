# Subproject Overview

We want to find galaxies within a particular redshift range using a narrowband filter. Given a target redshift (e.g. *z*= 0.10), we can design a filter for which a bright emission line (e.g. Hα) falls within the filter. The expectation is that galaxies at the targetted redshift will appear much brighter through this filter than galaxies at other redshifts, which only have continuum emission within the filter bandpass.

**The challenge** is that this filter will also identify higher redshift galaxies that have other emission lines (e.g. OIII) that have been redshifted to the same wavelength range.

![comparison of target and contaminant populations](https://github.com/yluo54301/dwarf_photo-z/raw/master/misc/linkable_images/2019_04/2019_04_02/comparison%20of%20narrow%20band%20target%20and%20contaminant%20galaxies.png)

**The goal** is to "de-confuse" these two populations: the low-*z* target galaxies, and the high-*z* contaminants. We will try **2 methods**:

1) **Catalog-only / traditional machine learning**: Just use the photometric magnitudes and fluxes. 

2) **Deep learning** on the actual images: train a Convolutional Neural Network to try to discriminate between the two populations. 

