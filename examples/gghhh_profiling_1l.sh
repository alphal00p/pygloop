echo "<<<<<<<"
echo "<<<<<<< 1-loop profiling: generation"
echo "<<<<<<<"
./bin/pygloop -s "set global kv global.generation.evaluator.iterative_orientation_optimization=false" "set global kv global.generation.threshold_subtraction.enable_thresholds=false" -o GGHHH1L_no_iterative_no_deformation --clean --n_loops 1 generate -t gammaloop
echo "<<<<<<<"
./bin/pygloop -s "set global kv global.generation.evaluator.iterative_orientation_optimization=true" "set global kv global.generation.threshold_subtraction.enable_thresholds=false" -o GGHHH1L_with_iterative_no_deformation --clean --n_loops 1 generate -t gammaloop
echo "<<<<<<<"
./bin/pygloop -s "set global kv global.generation.evaluator.iterative_orientation_optimization=false" "set global kv global.generation.threshold_subtraction.enable_thresholds=true" -o GGHHH1L_no_iterative_with_deformation --clean --n_loops 1 generate -t gammaloop
echo "<<<<<<<"
./bin/pygloop -s "set global kv global.generation.evaluator.iterative_orientation_optimization=true" "set global kv global.generation.threshold_subtraction.enable_thresholds=true" -o GGHHH1L_with_iterative_with_deformation --clean --n_loops 1 generate -t gammaloop
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_spenso_parametric --clean --n_loops 1 generate -t spenso
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_spenso_merging --clean --n_loops 1 generate -t spenso -g merging
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_spenso_function_map --clean --n_loops 1 generate -t spenso -g function_map
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_spenso_summing --clean --n_loops 1 generate -t spenso -g summing
echo "<<<<<<<"
echo "<<<<<<< 1-loop profiling: running"
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_no_iterative_no_deformation --n_loops 1 -ii gammaloop bench
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_with_iterative_no_deformation --n_loops 1 -ii gammaloop bench
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_no_iterative_with_deformation --n_loops 1 -ii gammaloop bench
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_with_iterative_with_deformation --n_loops 1 -ii gammaloop bench
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_spenso_parametric --n_loops 1 -ii spenso_parametric bench
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_spenso_merging --n_loops 1 -ii spenso_summed bench
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_spenso_function_map --n_loops 1 -ii spenso_summed bench
echo "<<<<<<<"
./bin/pygloop -o GGHHH1L_spenso_summing --n_loops 1 -ii spenso_summed bench
echo "<<<<<<<"
