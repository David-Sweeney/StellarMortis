import stellarmortis as sm
import numpy as np
import pandas as pd

if __name__ == '__main__':
    filename = './data/guw_f1e-3_bhm2.35.ebf'
    
    # Use case 1 - start from scratch
    guw = sm.Underworld(filename, species=['White Dwarf'], 
                        logging_file='test.log', verbose=1)
    guw.evolve()
    guw.save('./data/white_dwarfs_f1e-3.csv')
    
    
    all_data = [guw.data]
    for _ in range(100):
        guw.evolve(duration=1e-3)
        all_data.append(guw.data)

    all_data = pd.concat(all_data)
    all_data = all_data.reset_index(drop=True)
    guw.data = all_data
    guw.save('./data/white_dwarfs_f1e-1.csv')
        
    # guw.save('./increased_bhs_pre-kick.csv')
    # natal_kicks = sm.NatalKick()
    # guw.add_kicks(natal_kicks)
    # guw.save('./increased_bhs_pre-evolve.csv')
    # guw.evolve()
    # guw.save('./increased_bhs_test.csv')
    # guw.calculate_microlensing(run_name='Zofia', years=[int(1e4)], num_workers=-1, collate=True)
    
    
    