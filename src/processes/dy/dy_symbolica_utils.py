from symbolica import Expression, S  # pyright: ignore


def fold_momentum_components_into_gamma(expr: Expression) -> Expression:
    """Represent contracted Q/Qp components as gamma-chain vector arguments.

    Symbolica 2.1 collects open gamma products into ``spenso::chain`` objects,
    but custom two-argument momentum heads are not registered rank-one tensors.
    Folding the contracted component into the gamma argument before Clifford
    simplification lets idenso retain and contract that momentum structurally.
    """

    a_ = S("dy_gamma_a_")
    b_ = S("dy_gamma_b_")
    dim_ = S("dy_gamma_dim_")
    slot_ = S("dy_gamma_slot_")
    edge_ = S("dy_gamma_edge_")
    start_ = S("dy_chain_start_")
    end_ = S("dy_chain_end_")
    before___ = S("dy_chain_before___")
    after___ = S("dy_chain_after___")

    gamma = S("spenso::gamma")
    chain = S("spenso::chain")
    mink = S("spenso::mink")
    chain_in = S("spenso::in")
    chain_out = S("spenso::out")

    momentum_head_names = sorted(
        {
            symbol.get_name()
            for symbol in expr.get_all_symbols()
            if symbol.get_name().rsplit("::", 1)[-1] in {"Q", "Qp"}
        }
    )
    for head_name in momentum_head_names:
        head = S(head_name)
        component = head(edge_, mink(dim_, slot_))
        vector = head(edge_, mink(dim_))

        expr = expr.replace(
            gamma(a_, b_, mink(dim_, slot_)) * component,
            gamma(a_, b_, vector),
            repeat=True,
            allow_new_wildcards_on_rhs=True,
        )
        expr = expr.replace(
            chain(
                start_,
                end_,
                before___,
                gamma(chain_in, chain_out, mink(dim_, slot_)),
                after___,
            )
            * component,
            chain(
                start_,
                end_,
                before___,
                gamma(chain_in, chain_out, vector),
                after___,
            ),
            repeat=True,
            allow_new_wildcards_on_rhs=True,
        )

    return expr
