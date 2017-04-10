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
    catalog_[1,2] : pd.DataFrame
        - must contain columns `ra` and `dec`
        - doesn't need to be the entire dataset; could be a masked subset
    threshold_error : Optional(astropy.units.quantity.Quantity (angle-like))
        - default : 10**1.5 arcsec
        - must have angular units (e.g. arcsec)
    threshold_match : Optional(astropy.units.quantity.Quantity (angle-like))
        - matches closer than this are considered successful matches
        - default : 1 arcsec
        - must have angular units (e.g. arcsec)
    save : Optional(bool)
        - save matches to disk? (overwrites `data/matches.sqlite3`)

    
    Variables
    ---------
        - catalog_1 : pd.DataFrame
            the "reference" dataset
            each record in here will be matched once and only once
        - catalog_2 : pd.DataFrame
            the dataset you want to test
        - df : pd.DataFrame
            - index column: `catalog_1_ids` : int (unique)
            - other columns: 
                - `catalog_2_ids` : int (not necessarily unique)
                - `sep` : float
                     - in units of arcsec
                     - the angular separation between the objects in catalog_1 
                       and their closest counterparts in catalog_2
                - match : bool
                    True if `sep` < threshold_match
                    These matches are assumed to be true matches, 
                    but could be coincidental
                - error : bool
                    True if `sep` > threshold_error
                    These matches are assumed to be totally erroneous
                    (e.g. due to non-overlapping survey footprints)

    Methods
    -------
        - recompute_error(new_threshold_error)
            - overwrites threshold_error and recomputes `df.error`
        - recompute_match(new_threshold_match)
            - overwrites threshold_match and recomputes `df.match`
        - save_to_filename(filename)
            - saves the dataframe to disk as a sqlite3 table

    Class Methods
    -------------
        - load_from_filename(filename, table="matches")
            - loads and returns the dataframe from a sqlite3 table on disk 
              *Does not* return a `Matches` object


    Notes
    -----
    Right now I'm *only* matching on position. The matched objects might have 
    totally different fluxes, sizes, shapes, etc.  As `catalog_2` becomes more
    complete (and deeper), you're more likely to get false positives


    """
    sql_table_name = "matches"
    threshold_error_default = 15 * u.arcsec
    threshold_match_default = 1 * u.arcsec

    @u.quantity_input(threshold_error=u.arcsec, threshold_match=u.arcsec)
    def __init__(self, catalog_1, catalog_2,
        threshold_error = threshold_error_default,
        threshold_match = threshold_match_default,
        save = True
        ):
        super(Matches, self).__init__()
        self.catalog_1 = catalog_1
        self.catalog_2 = catalog_2

        coords_1 = coordinates.SkyCoord(catalog_1.ra,
                                        catalog_1.dec,
                                        unit=u.deg)

        coords_2 = coordinates.SkyCoord(catalog_2.ra,
                                        catalog_2.dec,
                                        unit=u.deg)

        idx, sep, _ = coords_1.match_to_catalog_sky(coords_2)

        catalog_1_ids = catalog_1.index
        catalog_2_ids = catalog_2.iloc[idx].index

        self.df = pd.DataFrame({
            "catalog_1_ids" :  catalog_1_ids,
            "catalog_2_ids" : catalog_2_ids,
            "sep" : sep.to(u.arcsec).value ,
            }, 
        )

        self.df.set_index("catalog_1_ids", inplace=True)

        self.df["match"] = False
        self.df["error"] = True

        self.recompute_error(threshold_error)
        self.recompute_match(threshold_match)

        if save:
            self.save_to_filename("data/matches.sqlite3")


    @u.quantity_input(threshold_error=u.arcsec)
    def recompute_error(self, new_threshold_error):
        """Given a new error threshold, update the error mask
`
        Inputs
        ------
        new_threshold_error : astropy.units.quantity.Quantity (angle-like)

        Outputs
        -------
        None

        Side Effects
        ------------
        Overwrites self.threshold_error, self.df.error

        """

        self.threshold_error = new_threshold_error
        self.df.error = (self.df.sep > self.threshold_error.to(u.arcsec))

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
        Overwrites self.threshold_match, self.df.match

        """

        self.threshold_match = new_threshold_match
        self.df.match = (self.df.sep < self.threshold_match.to(u.arcsec))

    def save_to_filename(self, filename):
        self.df.to_sql(self.sql_table_name, "sqlite:///{}".format(filename),
            if_exists="replace",
            )


    @classmethod
    def load_from_filename(cls, filename):
        df = pd.read_sql_table(cls.sql_table_name, 
                "sqlite:///{}".format(filename), 
                 index_col="catalog_1_ids",
            )
        return df




