./bin/pygloop --clean generate
./bin/pygloop -ii gammaloop inspect -p 100.0 200.0 300.0
GL_DISPLAY_FILTER=gammalooprs=debug ./bin/pygloop -ii gammaloop inspect -p 100.0 200.0 300.0
./bin/pygloop -ii gammaloop plot --nb_cores 8 --fixed_x 0.0 -r -1000.0 1500.0 --xs 0 2
./bin/pygloop --m_top 1000.0 -ii gammaloop integrate --integrator naive --points_per_iteration 1000 --n_iterations 10 --target 6.56089133881216768e-4-4.17078968913725420e-6j --phase imag
# | > Central value : -3.4422817856807024e-04   +/- 8.21e-06     (2.384%)
./bin/pygloop --m_top 1000.0 -ii gammaloop integrate --integrator naive --points_per_iteration 100000 --n_iterations 2 --target 6.56089133881216768e-4-4.17078968913725420e-6j --phase real --n_cores 8
# | > Central value : +4.2043405979800793e-04   +/- 2.14e-06     (0.508%)
./bin/pygloop --m_top 1000.0 -ii gammaloop integrate --integrator symbolica --points_per_iteration 100000 --n_iterations 2 --target 6.56089133881216768e-4-4.17078968913725420e-6j --phase real --n_cores 8
# | > Central value : +4.2107673536711040e-04   +/- 1.64e-06     (0.389%)
./bin/pygloop --m_top 1000.0 -ii gammaloop integrate --integrator vegas --points_per_iteration 100000 --n_iterations 2 --target 6.56089133881216768e-4-4.17078968913725420e-6j --phase real --n_cores 8
# | > Central value : +4.2021620551215211e-04   +/- 1.40e-07     (0.033%)
./bin/pygloop --m_top 1000.0 -ii gammaloop integrate --integrator gammaloop --points_per_iteration 100000 --n_iterations 2 --target 6.56089133881216768e-4-4.17078968913725420e-6j --n_cores 8 --restart
# |  itg #1   re: 0.0004229(19)       0.440%    0.036 χ²/dof    mwi: 8.4445e-5
