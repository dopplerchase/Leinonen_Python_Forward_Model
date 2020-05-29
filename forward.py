import pandas as pd 
import numpy as np 
import scipy.interpolate

class forward_Z():
    """ 
    Class to forward model reflectivity using Leinonen and Szrymer 2015. 
    
    Currenly, only outputs Ku-band Z. Can add X,Ka and W in the future if needed. 
    
    """    
    
    def set_PSD(self,PSD = None,D=None,dD=None,n_sims=6):
        """
        This sets the PSD objects in the class. It expects the following: 
        
        PSD: Matrix, (n_samples,n_bins); units: m^-4
        D: Array, (n_bins,); units m
        dD: Array, (n_bins,); units m 
        
        """
        #reshape PSD to have a third dimension. 1 for each L15 sim
        PSD = np.reshape(PSD,[PSD.shape[0],PSD.shape[1],1])
        PSD = np.tile(PSD,(1,1,n_sims))
        
        self.PSD = PSD 
        self.dD = dD
        self.D = D
        
        
        #time to reshape things
        psd_shape = self.PSD.shape
        
        #rescale to match shape of PSD. This allows fast computations through vectorization 
        self.dD = np.reshape(self.dD,[1,psd_shape[1]])
        self.dD = np.tile(self.dD,(psd_shape[0],1))
        self.dD = np.reshape(self.dD,[psd_shape[0],psd_shape[1],1])
        self.dD = np.tile(self.dD,(1,1,psd_shape[2]))
        
        #rescale to match shape of PSD. This allows fast computations through vectorization 
        self.D = np.reshape(self.D,[1,psd_shape[1]])
        self.D = np.tile(self.D,(psd_shape[0],1)) 
        self.D = np.reshape(self.D,[psd_shape[0],psd_shape[1],1])
        self.D = np.tile(self.D,(1,1,psd_shape[2]))
        
        #set a placeholder for the backscatter cross-section 
        self.sigma_x = np.zeros(psd_shape)
        self.sigma_ku = np.zeros(psd_shape)
        self.sigma_ka = np.zeros(psd_shape) 
        self.sigma_w = np.zeros(psd_shape)
    
    def load_split_L15(self):
        """ 
        This method loads the results from Leinonen and Szyrmer 2015 and then splits the particles into each rimed category.
        There are 6 categories. Each category of partilces were exposed to a larger amount of supercooled liquid water path. 
        The order of less rimed to heavily rimes is: No riming; 0.1 kg/m^2; 0.2 kg/m^2; 0.5 kg/m^2; 1.0 kg/m^2; 2.0 kg/m^2.
        """
        #load text file
        header = ['rimemodel','lwp','mass','Dmax','rad_gy','axis_ratio','rimed_fraction','Xchh','Xvv','Kuchh','Kucvv','Kachh','Kacvv','Wchh','Wcvv']
        leinonen = pd.read_csv('/data/gpm/a/randyjc2/Leinonen_2015_rimed.tex',delim_whitespace=True,names=header,header=None,index_col=None)
        
        #split methods 
        leinonen_A = leinonen.where(leinonen.rimemodel == 'A')
        leinonen_B = leinonen.where(leinonen.rimemodel == 'B')
        
        #grab all the rimed instances
        bins = np.arange(-0.05,2.05,0.1)
        bin_i = np.digitize(leinonen_B.lwp,bins=bins)
        leinonen_B['bin_i'] = bin_i
        grouped = leinonen_B.groupby('bin_i')
        groups = grouped.groups
        list_of_keys = list(groups.keys())
        list_of_subsetted_data = []
        for i in list_of_keys:
                g_i = np.asarray(groups[i].values,dtype=int)
                d = leinonen_B.iloc[g_i]
                list_of_subsetted_data.append(d)

        L01 = list_of_subsetted_data[0]
        L02 = list_of_subsetted_data[1]
        L05 = list_of_subsetted_data[2]
        L10 = list_of_subsetted_data[3]
        L20 =list_of_subsetted_data[4].dropna()
        
        #grab the non-rimed situation 
        bin_i = np.digitize(leinonen_A.lwp,bins=bins)
        leinonen_A['bin_i'] = bin_i
        grouped = leinonen_A.groupby('bin_i')
        groups = grouped.groups
        list_of_keys = list(groups.keys())
        list_of_subsetted_data = []
        for i in list_of_keys:
                g_i = np.asarray(groups[i].values,dtype=int)
                d = leinonen_A.iloc[g_i]
                list_of_subsetted_data.append(d)

        L00 = list_of_subsetted_data[0]
        
        #store them in the class. 
        self.L00 = L00
        self.L01 = L01
        self.L02 = L02
        self.L05 = L05
        self.L10 = L10
        self.L20 = L20
        
    def fit_sigmas(self):
        """ 
        This method is to fit a flexible function to the Leinonen and Szyrmer (2015) data. Essentially, it interpolates
        the backscatter cross-section to whatever values of D are inputed to the class. Please make sure you have the correct units.
        D should be in m.
        """
        
        #loop over the various degrees of riming 
        list_o_objects = [self.L00,self.L01,self.L02,self.L05,self.L10,self.L20]
        for i,ii in enumerate(list_o_objects):
            bins = np.append(np.linspace(1e-4,3e-3,5),np.linspace(3e-3,2.20e-2,7))
            whichbin = np.digitize(ii.Dmax,bins=bins)
            ii['bin_i'] = whichbin
            df = ii.groupby('bin_i').median()
            df = df.reindex(np.arange(0,len(bins)))
            df = df.interpolate()
            df = df.dropna(how='all')
            
            #fit the functions for each frequency 
            f_x = scipy.interpolate.interp1d(np.log10(df.Dmax.values[:-1]),np.log10(df.Xchh.values[:-1]),fill_value='extrapolate',kind='linear',bounds_error=False)
            sigma_x = 10**f_x(np.log10(self.D[0,:,0]))

            f_ku = scipy.interpolate.interp1d(np.log10(df.Dmax.values[:-1]),np.log10(df.Kuchh.values[:-1]),fill_value='extrapolate',kind='linear',bounds_error=False)
            sigma_ku = 10**f_ku(np.log10(self.D[0,:,0]))

            f_ka = scipy.interpolate.interp1d(np.log10(df.Dmax.values[:-1]),np.log10(df.Kachh.values[:-1]),fill_value='extrapolate',kind='linear',bounds_error=False)
            sigma_ka = 10**f_ka(np.log10(self.D[0,:,0]))

            f_w = scipy.interpolate.interp1d(np.log10(df.Dmax.values[:-1]),np.log10(df.Wchh.values[:-1]),fill_value='extrapolate',kind='linear',bounds_error=False)
            sigma_w = 10**f_w(np.log10(self.D[0,:,0]))

            #time to reshape things again so we can have vectorized calculations 
            psd_shape = self.PSD.shape

            sigma_x = np.reshape(sigma_x,[1,psd_shape[1]])
            sigma_x = np.tile(sigma_x,(psd_shape[0],1))

            sigma_ku = np.reshape(sigma_ku,[1,psd_shape[1]])
            sigma_ku = np.tile(sigma_ku,(psd_shape[0],1))

            sigma_ka = np.reshape(sigma_ka,[1,psd_shape[1]])
            sigma_ka = np.tile(sigma_ka,(psd_shape[0],1))

            sigma_w = np.reshape(sigma_w,[1,psd_shape[1]])
            sigma_w = np.tile(sigma_w,(psd_shape[0],1))

            #store it into the class, the 3rd dimension is now the various degrees of riming. 
            self.sigma_x[:,:,i] = np.copy(sigma_x)*1e6 #convert to mm^2 
            self.sigma_ku[:,:,i] = np.copy(sigma_ku)*1e6 #convert to mm^2 
            self.sigma_ka[:,:,i] = np.copy(sigma_ka)*1e6 #convert to mm^2 
            self.sigma_w[:,:,i] = np.copy(sigma_w)*1e6 #convert to mm^2 

    def calc_Z(self):
        """
        Here is the method that actualy calculates Z. Output is in dBZ. 
        
        The resulting shape is 2d. Axis 1 is still 
        """
        
        #create the coeficients in equation
        from pytmatrix import tmatrix_aux
        #X-band
        lamb = tmatrix_aux.wl_X #is in mm
        K = tmatrix_aux.K_w_sqr[lamb]
        coef = (lamb**4)/(np.pi**5*K) #mm^4
        #Ku-band
        lamb = tmatrix_aux.wl_Ku #is in mm
        K = tmatrix_aux.K_w_sqr[lamb]
        coef2 = (lamb**4)/(np.pi**5*K) #mm^4
        #Ka-band
        lamb = tmatrix_aux.wl_Ka #is in mm
        K = tmatrix_aux.K_w_sqr[lamb]
        coef3 = (lamb**4)/(np.pi**5*K) #mm^4
        #W-band
        lamb = tmatrix_aux.wl_W #is in mm
        K = tmatrix_aux.K_w_sqr[lamb]
        coef4 = (lamb**4)/(np.pi**5*K) #mm^4
        
        #calculate, output is in dBZ
        Z_x = 10*np.log10(coef*np.nansum(self.sigma_x*self.PSD*self.dD,axis=1))
        Z_ku = 10*np.log10(coef2*np.nansum(self.sigma_ku*self.PSD*self.dD,axis=1))
        Z_ka = 10*np.log10(coef3*np.nansum(self.sigma_ka*self.PSD*self.dD,axis=1))
        Z_w = 10*np.log10(coef4*np.nansum(self.sigma_w*self.PSD*self.dD,axis=1))
        
        #eliminate any missing values
        Z_x[np.isinf(Z_x)] = np.nan
        Z_ku[np.isinf(Z_ku)] = np.nan
        Z_ka[np.isinf(Z_ka)] = np.nan
        Z_w[np.isinf(Z_w)] = np.nan

        self.Z_x = Z_x 
        self.Z_ku = Z_ku 
        self.Z_ka = Z_ka 
        self.Z_w = Z_w 
