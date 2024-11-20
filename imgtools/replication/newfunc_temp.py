def sliding_window_run(self) -> None:
    """ Run the sliding window analysis.
    Treats each cell, locus independently.
    The estimations are performed by taking sliding windows of a given size centered at each locus.
    The efficiency and bias for each cell, locus and z quantile are calculated from the previous steps.
    Estimates:
        - eps_ic, detection efficiency. shape: (ncells, nloci, ncopies),
        - beta_ic, bias rate. shape: (ncells, nloci, ncopies),
        - p_ic, replication probability. shape: (ncells, nloci, ncopies).
    """
    
    print('SLIDING WINDOW RUN')
    print('------------------')

    # Initialize the efficiency and bias tensors of shape (ncells, nloci, ncopies)
    eps_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
    alpha_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
    
    # Loop over cells, z and rad quantiles to assign the efficiency and bias
    for c, state in enumerate(self.states):
        for z in self.zquants:
            
            # Get the masks for the quantile z in the cell c
            mask_cz = self.zq_ic[c, :, :] == z  # shape: (nloci, ncopies)
            
            # Assign the efficiency values
            if state == 'G1':
                eps_ic[c, :, :][mask_cz] = np.tile(self.eps_iz_G1[:, z][:, np.newaxis], (1, self.ncopies))[mask_cz]
            elif state == 'S':
                eps_ic[c, :, :][mask_cz] = np.tile(self.eps_iz_S[:, z][:, np.newaxis], (1, self.ncopies))[mask_cz]
            elif state == 'G2':
                eps_ic[c, :, :][mask_cz] = np.tile(self.eps_iz_G2[:, z][:, np.newaxis], (1, self.ncopies))[mask_cz]
            
            for d in self.radquants:
                
                # Get the masks for the quantile d in the cell c and copy h
                mask_cd = self.radq_ic[c, :, :] == d  # shape: (nloci)
                # Combine the masks
                mask_czd = np.logical_and(mask_cz, mask_cd)  # shape: (nloci)
                
                # Assign the bias values
                alpha_ic[c, :, :][mask_czd] = self.alpha_czd[c, z, d] * np.ones((self.nloci, self.ncopies))[mask_czd]
    
    # Loop over cells, z and rad quantiles to correct the efficiency and bias
    for c in range(self.ncells):
        for z in self.zquants:
            
            # Get the masks for the quantile z in the cell c
            mask_cz = self.zq_ic[c, :, :] == z  # shape: (nloci, ncopies)
            
            # Correct for every cell / z / rad
            for d in self.radquants:
                
                # Get the masks for the quantile d in the cell c
                mask_cd = self.radq_ic[c, :, :] == d  # shape: (nloci, ncopies)
                mask_czd = np.logical_and(mask_cz, mask_cd)  # shape: (nloci, ncopies)
                
                # Get the efficiency for the current cell, z and rad
                eps_czd_i = eps_ic[c, :, :][mask_czd]  # shape: (nloci_zd)
                # Rescale the efficiency by the cell_n_z_n_rad run estimates
                eps_czd_i = eps_czd_i + self.eps_czd[c, z, d] - np.nanmean(eps_czd_i)
                # Assign the corrected values
                eps_ic[c, :, :][mask_czd] = eps_czd_i
            
            # Correct for every cell / z
            eps_cz_i = eps_ic[c, :, :][mask_cz]  # shape: (nloci_z)
            alpha_cz_i = alpha_ic[c, :, :][mask_cz]  # shape: (nloci_z)
            # Rescale the efficiency and bias by cell_n_z estimates
            eps_cz_i = eps_cz_i + self.eps_cz[c, z] - np.nanmean(eps_cz_i)
            alpha_cz_i = alpha_cz_i + self.alpha_cz[c, z] - np.nanmean(alpha_cz_i)
            # Assign the corrected values
            eps_ic[c, :, :][mask_cz] = eps_cz_i
            alpha_ic[c, :, :][mask_cz] = alpha_cz_i
        
        # Correct for every cell
        eps_ic[c, :, :] = eps_ic[c, :, :] + self.eps_c[c] - np.nanmean(eps_ic[c, :, :])
        alpha_ic[c, :, :] = alpha_ic[c, :, :] + self.alpha_c[c] - np.nanmean(alpha_ic[c, :, :])
    
    """# Initialize the efficiency and bias tensors of shape (ncells, nloci, ncopies)
    eps_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
    alpha_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
    for c, state in enumerate(self.states):
        if state == 'G1':
            eps_ic[c, :, :] = np.tile(self.eps_i_G1[:, np.newaxis], (1, self.ncopies))
            alpha_ic[c, :, :] = self.alpha_G1 * np.ones((self.nloci, self.ncopies))
        elif state == 'S':
            eps_ic[c, :, :] = np.tile(self.eps_i_S[:, np.newaxis], (1, self.ncopies))
            alpha_ic[c, :, :] = self.alpha_S * np.ones((self.nloci, self.ncopies))
        elif state == 'G2':
            eps_ic[c, :, :] = np.tile(self.eps_i_G2[:, np.newaxis], (1, self.ncopies))
            alpha_ic[c, :, :] = self.alpha_G2 * np.ones((self.nloci, self.ncopies))
    
    # Correct the efficiency and bias for every cell, z and rad quantile
    for c, state in enumerate(self.states):
        for z in self.zquants:
            
            # Get the masks for the quantile z in the cell c
            mask_cz = self.zq_ic[c, :, :] == z  # shape: (nloci, ncopies)
            
            for d in self.radquants:
                
                # Get the masks for the quantile d in the cell c
                mask_cd = self.radq_ic[c, :, :] == d  # shape: (nloci, ncopies)
                mask_czd = np.logical_and(mask_cz, mask_cd)  # shape: (nloci, ncopies)
                
                # Get the efficiency and bias for the current cell, z and rad
                eps_czd_i = eps_ic[c, :, :][mask_czd]  # shape: (nloci_zd)
                alpha_czd_i = alpha_ic[c, :, :][mask_czd]  # shape: (nloci_zd)
                # Rescale the efficiency and bias by the z_n_rad estimates
                if state == 'G1':
                    eps_czd_i = eps_czd_i + self.eps_zd_G1[z, d] - np.nanmean(eps_czd_i)
                    alpha_czd_i = alpha_czd_i + self.alpha_zd_G1[z, d] - np.nanmean(alpha_czd_i)
                elif state == 'S':
                    eps_czd_i = eps_czd_i + self.eps_zd_S[z, d] - np.nanmean(eps_czd_i)
                    alpha_czd_i = alpha_czd_i + self.alpha_zd_S[z, d] - np.nanmean(alpha_czd_i)
                elif state == 'G2':
                    eps_czd_i = eps_czd_i + self.eps_zd_G2[z, d] - np.nanmean(eps_czd_i)
                    alpha_czd_i = alpha_czd_i + self.alpha_zd_G2[z, d] - np.nanmean(alpha_czd_i)
                # Assign the corrected values
                eps_ic[c, :, :][mask_czd] = eps_czd_i
                alpha_ic[c, :, :][mask_czd] = alpha_czd_i
            
            # Correct for every cell / copy / z
            eps_cz_i = eps_ic[c, :, :][mask_cz]  # shape: (nloci_z)
            alpha_cz_i = alpha_ic[c, :, :][mask_cz]  # shape: (nloci_z)
            # Rescale the efficiency and bias by the z estimates
            if state == 'G1':
                eps_cz_i = eps_cz_i + self.eps_z_G1[z] - np.nanmean(eps_cz_i)
                alpha_cz_i = alpha_cz_i + self.alpha_z_G1[z] - np.nanmean(alpha_cz_i)
            elif state == 'S':
                eps_cz_i = eps_cz_i + self.eps_z_S[z] - np.nanmean(eps_cz_i)
                alpha_cz_i = alpha_cz_i + self.alpha_z_S[z] - np.nanmean(alpha_cz_i)
            elif state == 'G2':
                eps_cz_i = eps_cz_i + self.eps_z_G2[z] - np.nanmean(eps_cz_i)
                alpha_cz_i = alpha_cz_i + self.alpha_z_G2[z] - np.nanmean(alpha_cz_i)
            # Assign the corrected values
            eps_ic[c, :, :][mask_cz] = eps_cz_i
            alpha_ic[c, :, :][mask_cz] = alpha_cz_i
        
        # Correct for every cell
        eps_ic[c, :, :] = eps_ic[c, :, :] + self.eps_c[c] - np.nanmean(eps_ic[c, :, :])
        alpha_ic[c, :, :] = alpha_ic[c, :, :] + self.alpha_c[c] - np.nanmean(alpha_ic[c, :, :])"""
    
    eps_ic = self.print_n_clip('eps_ic', eps_ic, 0, 1)
    alpha_ic = self.print_n_clip('alpha_ic', alpha_ic, 0, None)
    
    # Clip n_ic up to 4 to avoid large overestimations of the replication probability
    n_ic = self.print_n_clip('n_ic', self.n_ic, 0, 4)
    
    # Get the window size in units of loci
    window = int(np.ceil(self.config['sliding_window_size'] / self.index.resolution()))
    
    # Calculate the sliding window averages
    n_ic_SW = scf_utils.sliding_matrix(n_ic, self.index, window=window, method='mean')
    eps_ic_SW = scf_utils.sliding_matrix(eps_ic, self.index, window=window, method='mean')
    alpha_ic_SW = scf_utils.sliding_matrix(alpha_ic, self.index, window=window, method='mean')
    
    # Calculate the replication probability
    p_ic_SW = n_ic_SW / (eps_ic_SW * alpha_ic_SW) - 1
    
    # Set all the -1 values to NaN
    # p_ic_SW[p_ic_SW == -1] = np.nan

    # Store the results
    self.eps_ic = eps_ic_SW
    self.alpha_ic = alpha_ic_SW
    self.p_ic = p_ic_SW
    
    print('OVER.')
    print('\n\n')