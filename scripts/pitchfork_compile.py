from .compile_from_dict import numpy_compile
import numpy as np

"""
compile_pitchfork.py

compiles the pitchork model from a dictionary which was saved using tf_to_dict.py
applies the relevant data scaling and inverse-pca projections 

predict functions mean we can pass inputs in grid dimensions and 
receive outputs in expected dimensions too!
"""

# =======================
# class to compile pitchfork
# =======================
class pitchfork_compile():
    def __init__(self, model_dict, model_info):
        """
        class to compile pitchfork using compile_from_dict and known pre- and post-
        prediction scalings
        """
        
        ### compile from dict
        self.model = numpy_compile(model_dict)

        ### load relevant info for scaling
        self.log_inputs_mean = np.array(model_info["data_scaling"]["inp_mean"][0])        
        self.log_inputs_std = np.array(model_info["data_scaling"]["inp_std"][0])
        self.log_outputs_mean = np.array(model_info["data_scaling"]["classical_out_mean"][0] + model_info["data_scaling"]["astero_out_mean"][0])        
        self.log_outputs_std = np.array(model_info["data_scaling"]["classical_out_std"][0] + model_info["data_scaling"]["astero_out_std"][0])
        self.pca_comps = np.array(model_info['custom_objects']['inverse_pca']['pca_comps'])
        self.pca_mean = np.array(model_info['custom_objects']['inverse_pca']['pca_mean'])        

        ### def constants
        self.L_sun = 3.828e+26
        self.R_sun = 6.957e+8
        self.SB_sigma = 5.670374419e-8

    def predict(self, inputs, n_min=6, n_max=40):
        ## indexing for radial order slice according to n_min and n_max
        n_slice_index = np.r_[0, 1, 2, np.arange(n_min-3, n_max-2)]
        
        log_inputs = np.log10(inputs)
        
        standardised_log_inputs = (log_inputs - self.log_inputs_mean)/self.log_inputs_std
        
        preds = self.model.forward_pass(standardised_log_inputs)

        pca_preds = preds[1] @ self.pca_comps + self.pca_mean
        
        standardised_log_outputs = np.concatenate((preds[0], pca_preds), axis=1)

        log_outputs = (standardised_log_outputs*self.log_outputs_std) + self.log_outputs_mean

        outputs = np.empty_like(log_outputs)
        
        outputs[:, :2] = 10**log_outputs[:, :2]

        outputs[:, 2] = log_outputs[:, 2]##we want star_feh in dex

        outputs[:, 3:] = 10**log_outputs[:, 3:] 

        teff = np.array(((outputs[:,1]*self.L_sun) / (4*np.pi*self.SB_sigma*((outputs[:,0]*self.R_sun)**2)))**0.25)
        
        outputs[:,0] = teff
        
        outputs = outputs[:, n_slice_index]

        return outputs




















        