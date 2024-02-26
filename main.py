from multiprocessing import cpu_count

import stellarmortis as sm

if __name__ == '__main__':
    filename = './data/galaxia_f1e-4_bhm2.35.ebf'
    
    # # Use case 1 - start from scratch
    # natal_kicks = sm.NatalKick()
    # masses = sm.Mass()
    # guw = sm.Underworld(filename, natal_kicks=natal_kicks, masses=masses, 
    #                  species=['Black Hole'], logging_file='test.log', verbose=1)
    # guw.evolve()
    # guw.save('./test.csv')
    # guw.calculate_microlensing(run_name='test', years=[int(1e4)], num_workers=-1, collate=True)
    # sm.plot_microlensing({'Black Hole': 'testtest.ecsv'}, {'Black Hole': 1e4}, output_filepath='./test_events.ecsv')
    
    # # Use case - perform microlensing on a saved population
    # guw = Underworld('./test.csv', logging_file='test.log', verbose=1, append_logging=True)
    # guw.calculate_microlensing(run_name='test', years=[5], num_workers=-1, collate=True)
    
    # Use case - join existing microlensing run
    # sm.calculate_microlensing('test.csv', 'test', [int(1e4)], num_workers=-1, collate=True, 
    #                           logging_file='test.log', append_logging=True, 
    #                           output_filepath='./test_events.ecsv', verbose=1) # Joins previous run
    # sm.plot_microlensing({'Black Hole': 'test_test.ecsv'}, {'Black Hole': 1e4}, output_dir='.')
    
    # # Use case - fine grained control
    # natal_kicks = sm.NatalKick({'Neutron Star': 350, 'Black Hole': 100})
    # masses = sm.Mass({'Neutron Star': 1.4, 'Black Hole': 5.0})
    # guw = Underworld(filename, logging_file='test.log', verbose=1)
    # guw.filter_species(['Black Hole'])
    # guw.add_masses(masses)
    # guw.add_kicks(natal_kicks)
    # guw.save('test.csv')
    # guw.evolve()
    # guw.save('test.csv')
    # guw.calculate_microlensing(run_name='test', years=[5], num_workers=cpu_count()//2, collate=True)
    
    # Use case - plot microlensing run
    base_filepath = '/import/morgana2/snert/david/Gravitational_Lensing/Incidence/Data/'
    sm.plot_microlensing({
                        #   'Black Hole': base_filepath + '10000_year_lensing_results_all_GUW_paper_mag.ecsv',
                        #   'Neutron Star': base_filepath + '10000_year_lensing_results_all_GUW_paper_mag.ecsv',
                          'Star': base_filepath + '100000_year_lensing_results_milkyway_all_MW_paper_mag.ecsv'}, 
                         {'Black Hole': 1e3, 'Neutron Star': 1e3, 'Star': 1e6}, output_dir='../plots')
    
    