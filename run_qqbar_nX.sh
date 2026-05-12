#!/usr/bin/env bash

set -euo pipefail

GL=/Users/vjhirsch/Documents/Work/gammaloop_dev/target/dev-optim/gammaloop
STATE=./outputs/gammaloop_states/qqbar_nX_standalone_qqbar_nX_d_dbar_h_h_h_2L_top_pentagon_isr_subtracted
CARD=./outputs/dot_files/qqbar_nX/qqbar_nX_d_dbar_h_h_h_2L_top_pentagon_isr_subtracted.toml

usage() {
    cat <<EOF
Usage: $0 [all|load|generate|inspect|inspect-f64|inspect-arb|integrate|integrate-pm|integrate-pp|open]...

Examples:
  $0
  $0 load generate inspect
  $0 generate inspect-f64 integrate-pm
  $0 integrate-pp
  $0 open
EOF
}

run_load() {
    "$GL" -o --clean-state "$CARD" run load_subtracted_dot -c 'quit -o true'
}

run_generate() {
    "$GL" -o -s "$STATE" run generate_subtracted_integrand -c 'quit -o true'
}

run_inspect_f64() {
    "$GL" -o -s "$STATE" run inspect_collinear_p1 -c 'quit -o true'
    "$GL" -o -s "$STATE" run inspect_collinear_p2 -c 'quit -o true'
}

run_inspect_arb() {
    "$GL" -o -s "$STATE" run inspect_collinear_p1_arb -c 'quit -o true'
    "$GL" -o -s "$STATE" run inspect_collinear_p2_arb -c 'quit -o true'
}

run_inspect() {
    run_inspect_f64
    run_inspect_arb
}

run_integrate_pm() {
    "$GL" -o -s "$STATE" run low_stat_integrate_pm -c 'quit -o true'
}

run_integrate_pp() {
    "$GL" -o -s "$STATE" run low_stat_integrate_pp -c 'quit -o true'
}

run_integrate() {
    run_integrate_pm
    run_integrate_pp
}

run_open() {
    exec "$GL" -o "$CARD"
}

run_all() {
    run_load
    run_generate
    run_inspect
    run_integrate
}

if [[ $# -eq 0 ]]; then
    set -- all
fi

for section in "$@"; do
    case "$section" in
        all)
            run_all
            ;;
        load)
            run_load
            ;;
        generate)
            run_generate
            ;;
        inspect)
            run_inspect
            ;;
        inspect-f64)
            run_inspect_f64
            ;;
        inspect-arb)
            run_inspect_arb
            ;;
        integrate)
            run_integrate
            ;;
        integrate-pm)
            run_integrate_pm
            ;;
        integrate-pp)
            run_integrate_pp
            ;;
        open)
            run_open
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown section: $section" >&2
            usage >&2
            exit 2
            ;;
    esac
done
