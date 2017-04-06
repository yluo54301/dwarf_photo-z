import pandas as pd


# catalog abstract parent class?

class HSC(object):
    """HSC 
    Variables
    ---------
    filename : str
        the path of the origin of the data (should be a sqlite database)
    df : pandas.DataFrame
        contains :
            - object_id : int
            - ra : float
                degrees
            - dec : float
                degrees
            - detect_is_patch_inner  : boolean
                True if object not duplicated on two patches
            - detect_is_tract_inner  : boolean
                True if object not duplicated on two tracts
            - detect_is_primary : boolean
                True if object is not duplicated on 2 patches or tracts, 
                and if object is already deblended (i.e. has no children)
            - *cmodel_flux : float
                flux measurement for a particular band
            - *cmodel_flux_err : float
                flux measurement uncertianty for a particular band
            - *cmodel_flux_flags : boolean
                True if flux measurement failed or is untrustworthy for a particular band

    label : str
        a label for plot legends


    Methods
    -------

    
    Notes
    -----
    Available flux bands are {g,r,i,z,y} (see `*cmodel_flux*` variables in `df`)

    The required sqlite file isn't included in the github repo.
    In order to get this data, run the notebook `data/get_fluxes.ipynb`

    """
    def __init__(self, filename):
        """Read in, filter, create an HSC catalog object

        Inputs
        ------
            filename : str
                - should point to a sqlite database


        Notes
        -----
        The `filename` database should include the data within `table_1`
        The expected data fields are:
            - object_id : Optional(int)
            - ra : float
                - degrees
            - dec : float
                - degrees

        """
        super(HSC, self).__init__()
        self.filename = filename

        self.df = pd.read_sql_table("hsc", "sqlite:///data/{}".format(filename),
                                    index_col="object_id")

        self.df = self.df[self.df.detect_is_primary]

        self.label = "HSC"


        


class COSMOS(object):
    """COSMOS 
    Variables
    ---------
    filename : str
        the path of the origin of the data (should be a sqlite database)
    df : pandas.DataFrame
        contains :
            - id : int
            - alpha : float
                degrees
            - delta : float
                degrees
            - flag_Capak : int
            - flag_UVISTA : int
            - flag_deep : int
            - flag_shallow : int
            - photo_z : float
            - classification : int
            - flag_Capak : int
            - mass_med : float
                - log10 of median posterior estimate of stellar mass / M_sun
            - mass_med_min68 : float
                - log10 of 16th percentile posterior estimate of stellar mass
            - mass_med_max68 : float
                - log10 of 83th percentile posterior estimate of stellar mass
    label : str
        a label for plot legend


    Methods
    -------

    
    Notes
    -----
        df.ra is a clone of df.alpha
        df.dec is a copy of df.delta

    """
    def __init__(self, filename):
        """Read in, filter, create an COSMOS catalog object

        Inputs
        ------
            filename : str
                - should point to a sqlite database


        Notes
        -----
        The `filename` database should include the data within `COSMOS`
        The expected data fields are:
            - id : int
            - alpha : float
                degrees
            - delta : float
                degrees
            - flag_Capak : int
            - flag_UVISTA : int
            - flag_deep : int
            - flag_shallow : int
            - photo_z : float
            - classification : int
            - flag_Capak : int
            - mass_med : float
                - log10 of median posterior estimate of stellar mass / M_sun
            - mass_med_min68 : float
                - log10 of 16th percentile posterior estimate of stellar mass
            - mass_med_max68 : float
                - log10 of 83th percentile posterior estimate of stellar mass

        """
        super(COSMOS, self).__init__()
        self.filename = filename

        self.df = pd.read_sql_table("COSMOS", "sqlite:///data/{}".format(filename))

        # filter out bad data / non-galaxies
        self.df = self.df[self.df["photo_z"] > 0]
        self.df = self.df[self.df["photo_z"] < 8]
        self.df = self.df[self.df["mass_med"] > 0]

        self.df["ra"] = self.df.alpha
        self.df["dec"] = self.df.delta

        self.label = "COSMOS"


