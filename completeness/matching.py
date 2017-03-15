import numpy as np
import pandas as pd
import astropy

from astropy import units as u
from astropy import coordinates


# def match(dataset_1, dataset_2):
#   """Matches objects in dataset_1, to the closest object in dataset_2

#   This is basically a wrapper on 

#   """


class Matches(object):
    """Matches objects in `catalog_1` to the closest object in `catalog_2`

    This is largely a wrapper on `astropy.coordinates.SkyCoord.match_to_catalog_sky`
    but with the following value-adds:
        - set a 


    Inputs
    ------
    catalog_1 : an object defined in `datasets.py`
    catalog_2 : an object defined in `datasets.py`
    mask_catalog_1 : Optional(None or a boolean mask for catalog_1.df)
    mask_catalog_2 : Optional(None or a boolean mask for catalog_2.df)
    threshold_error : Optional(astropy.units.quantity.Quantity (angle-like))
        - default : 10**1.5 arcsec
        - must have angular units (e.g. arcsec)
    threshold_match : Optional(astropy.units.quantity.Quantity (angle-like))
        - matches closer than this are considered successful matches
        - default : 1 arcsec
        - must have angular units (e.g. arcsec)

    
    Variables
    ---------
        - catalog_1 : an object defined in `datasets.py`
            the "reference" dataset
        - catalog_2 : an object defined in `datasets.py`
            the dataset you want to test
        - mask_catalog_1 : a boolean mask for catalog_1.df
        - mask_catalog_2 : a boolean mask for catalog_2.df
        - idx : np.ndarray
            shape : (N,), where N is the length of catalog_1
            dtype : int
            the index in catalog_2 that most closely matches the object in catalog_1
            i.e. catalog_2[mask_catalog_2][j] = catalog_1[mask_catalog_1][idx[j]]
        - sep : astropy.coordinates.angles.Angle (array-like)
            shape : (N,), where N is the length of catalog_1
            dtype : float
            the angular separation between the objects in catalog_1 and their
            closest counterparts in catalog_2
        - threshold_error : astropy.units.quantity.Quantity (angle-like)
            assume any matches separated by more than this threshold to be due
            to non-matching survey footprints, not by completeness limits
        - threshold_match : astropy.units.quantity.Quantity (angle-like)
            assume any matched objects with separations closer than this value
            to be correctly matched objects
        - mask_error : numpy.ndarray (boolean)
            These matches are assumed to be totally erroneous
            (e.g. due to non-overlapping survey footprints)
        - mask_match : numpy.ndarray (masked)
            These matches are assumed to be totally erroneous
            (e.g. due to non-overlapping survey footprints)



    Methods
    -------
        - recompute_error(new_threshold_error)
            - overwrites threshold_error and recomputes `mask_error`
        - recompute_match(new_threshold_match)
            - overwrites threshold_match and recomputes `mask_match`


    Notes
    -----
    to do: create a standardized abstract base class in datasets

    Both catalogs *must* have an array-like df.ra, df.dec defined for each object

    Right now I'm *only* matching on position. The matched objects might have 
    totally different fluxes, sizes, shapes, etc.  As `catalog_2` becomes more
    complete (and deeper), you're more likely to get false positives


    """
    @u.quantity_input(threshold_error=u.arcsec, threshold_match=u.arcsec)
    def __init__(self, catalog_1, catalog_2,
        mask_catalog_1 = None,
        mask_catalog_2 = None,
        threshold_error = 15 * u.arcsec,
        threshold_match =  1 * u.arcsec):

        super(Matches, self).__init__()
        self.catalog_1 = catalog_1
        self.catalog_2 = catalog_2

        if mask_catalog_1 is None:
            mask_catalog_1 = np.isfinite(catalog_1.df.ra) * np.isfinite(catalog_1.df.dec)
        self.mask_catalog_1 = mask_catalog_1

        if mask_catalog_2 is None:
            mask_catalog_2 = np.isfinite(catalog_2.df.ra) * np.isfinite(catalog_2.df.dec)
        self.mask_catalog_2 = mask_catalog_2

        coords_1 = coordinates.SkyCoord(catalog_1.df[self.mask_catalog_1].ra,
                                        catalog_1.df[self.mask_catalog_1].dec,
                                        unit=u.deg)

        coords_2 = coordinates.SkyCoord(catalog_2.df[self.mask_catalog_2].ra,
                                        catalog_2.df[self.mask_catalog_2].dec,
                                        unit=u.deg)

        self.idx, self.sep, _ = coords_1.match_to_catalog_sky(coords_2)

        self.recompute_error(threshold_error)
        self.recompute_match(threshold_match)

    @u.quantity_input(threshold_error=u.arcsec)
    def recompute_error(self, new_threshold_error):
        """Given a new error threshold, update the error mask

        Inputs
        ------
        new_threshold_error : astropy.units.quantity.Quantity (angle-like)

        Outputs
        -------
        None

        Side Effects
        ------------
        Overwrites self.threshold_error, self.mask_error

        """

        self.threshold_error = new_threshold_error
        self.mask_error = (self.sep > self.threshold_error)

    @u.quantity_input(threshold_match=u.arcsec)
    def recompute_match(self, new_threshold_match):
        """Given a new match threshold, update the match mask

        Inputs
        ------
        new_threshold_match : astropy.units.quantity.Quantity (angle-like)

        Outputs
        -------
        None

        Side Effects
        ------------
        Overwrites self.threshold_match, self.mask_match

        """

        self.threshold_match = new_threshold_match
        self.mask_match = (self.sep < self.threshold_match)



