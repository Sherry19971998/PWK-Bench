#!/usr/bin/env python
"""
End-to-end PWK-Bench runner.

Offline demo (no API key, no downloads, FULLY SYNTHETIC toy cohort):
    python scripts/variant/run_benchmark.py --demo

Real run, on the already-bundled real 491-variant cohort (real ClinVar
labels + real PVS1/PM2/PP3/PM1; no fetch step needed -- see
data/sample/cohort_full_real.parquet and docs/variant/real_data.md):
    export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
    python scripts/variant/run_benchmark.py \
        --cohort data/sample/cohort_full_real.parquet \
        --models real_all_api --domain variant

Writes results/ (results.csv + results.json) and four figures to figures/
(budget_curve, single_slot, order_optimality, scaling_gap). On --demo the
numbers are ILLUSTRATIVE (fully synthetic toy cohort + mock agents), NOT the
paper's results -- --demo exists to verify the pipeline runs, not to
reproduce a number. `data/real/cohort_full.parquet` (built from scratch via
the fetch scripts in docs/variant/real_data.md) is an alternative to the
bundled cohort, e.g. to independently re-derive it, not a requirement.
"""

import os as _os, sys as _sys
# Make the package importable when the script is run directly from a
# checkout (scripts/<block>/x.py -> repo root is three dirnames up).
# Without this the script only works under `PYTHONPATH=.` or an
# installed package, which silently looks like the layout is broken.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import os
import pandas as pd
import yaml
from pwkbench.utils import set_seed
from pwkbench import harness, figures
from pwkbench.domains.variant import make_synthetic_cohort, VARIANT_DOMAIN


def _stratified_subsample(df, n, seed=0):
    """Down-sample to ~n rows keeping per-gene and per-label proportions, so a
    cheap smoke run stays representative (never silently drops a whole gene or
    one class). Returns the df unchanged if it already has <= n rows."""
    if n is None or len(df) <= n:
        return df
    frac = n / len(df)
    keys = ["gene", "label"] if "label" in df.columns else ["gene"]
    parts = [g.sample(max(1, round(len(g) * frac)), random_state=seed)
             for _, g in df.groupby(keys, sort=False)]
    return pd.concat(parts).reset_index(drop=True)


def load_cohort(args):
    if args.cohort:
        from pwkbench.domains.base import load_real_cohort
        df = pd.read_parquet(args.cohort)
        df = _stratified_subsample(df, getattr(args, "max_variants", None))
        dom = {"variant": VARIANT_DOMAIN}[args.domain]
        # Shared loader computes the empirical per-channel neutral so every
        # script scores the same parquet identically (see base.load_real_cohort).
        return load_real_cohort(df, dom)
    # demo synthetic
    coh = {"variant": make_synthetic_cohort}[args.domain]()
    n = getattr(args, "max_variants", None)
    if n is not None and len(coh.df) > n:
        # honour --max-variants on the demo path too (was silently ignored)
        from pwkbench.domains.base import attach_empirical_neutrals
        sub = _stratified_subsample(coh.df, n).reset_index(drop=True)
        coh = type(coh)(sub, attach_empirical_neutrals(sub, coh.domain))
    return coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline synthetic demo")
    ap.add_argument("--cohort", help="parquet cohort (real run)")
    ap.add_argument("--domain", choices=["variant"], default="variant")
    ap.add_argument("--models", default="demo", help="key in configs/models.yaml")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--with-ci", action="store_true",
                    help="add gene-clustered bootstrap CIs for each A(k) (slower)")
    ap.add_argument("--robustness", action="store_true",
                    help="also emit the paper's RQ3/RQ4 analyses: two-mask "
                         "memorization probe, paired agent-RelMax contrast CI, "
                         "disease-domain stratification, evidence-pool ablation")
    ap.add_argument("--max-variants", type=int, default=None,
                    help="cap the cohort to N variants (gene-stratified) for a "
                         "cheap smoke run before the full paid sweep. Real-model "
                         "cost scales with N x K calls per model, so e.g. "
                         "--max-variants 40 turns a ~2000-call run into ~160.")
    ap.add_argument("--outdir", default=None,
                    help="results dir (default: results/<domain>)")
    ap.add_argument("--figdir", default=None,
                    help="figures dir (default: figures/<domain>)")
    args = ap.parse_args()
    if args.demo and args.cohort:
        ap.error("--demo (synthetic) and --cohort (real parquet) are mutually "
                 "exclusive; pass exactly one.")
    if not args.demo and not args.cohort:
        ap.error("pass --demo (offline synthetic) or --cohort <parquet> (real).")
    # per-domain default so results/figures are namespaced under the domain name
    if args.outdir is None:
        args.outdir = f"results/{args.domain}"
    if args.figdir is None:
        args.figdir = f"figures/{args.domain}"
    set_seed()

    cohort = load_cohort(args)
    matrix = None
    if os.path.exists(args.config):
        with open(args.config) as _cfgf:
            cfg = yaml.safe_load(_cfgf)
        if args.models not in cfg:
            ap.error(f"--models '{args.models}' is not a key in {args.config}. "
                     f"Available: {', '.join(k for k in cfg)}")
        matrix = cfg.get(args.models)

    # Guard against silently running MOCK agents on a REAL cohort: --cohort
    # loads real data, but --models defaults to "demo" (mock adapters), which
    # would produce a plausible-looking results.csv that is NOT a real-model
    # run. Warn loudly so the output is not mistaken for real agent behaviour.
    if args.cohort and matrix and all(m.get("kind") == "mock" for m in matrix):
        import warnings
        warnings.warn(
            f"Running MOCK agents (--models {args.models}) on a REAL cohort "
            f"({args.cohort}). These results are illustrative, not real-model "
            f"behaviour. Pass --models real (with API keys) for a real run.",
            stacklevel=2)

    df = harness.run(cohort, model_matrix=matrix, with_ci=args.with_ci)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)
    df.to_csv(f"{args.outdir}/results.csv", index=False)
    json.dump(df.to_dict(orient="records"), open(f"{args.outdir}/results.json", "w"), indent=2)

    if args.robustness:
        import numpy as np
        from pwkbench import metrics as M
        from pwkbench.strategies import (relmax_order, oracle_order,
                                         heuristic_order, best_fixed_order,
                                         definedness_baseline_order,
                                         definedness_stratified_best_order)
        # fixed-order decomposition -> fixed_order_decomposition.csv
        # RelMax (label-free deployable prior) <= BestFixed (tightest fixed
        # order, label-conditioned) <= Oracle (per-instance ceiling).
        # Oracle-BestFixed isolates the gain from per-instance adaptation over
        # ANY fixed order; BestFixed-RelMax is the value of picking the right
        # fixed order.
        _bud = list(range(1, cohort.domain.K + 1))
        def _pe(o, leq_k=False):
            return float(np.mean(list(
                M.curve_A(cohort, o, budgets=_bud, leq_k=leq_k).values())))
        _pe_rel = _pe(relmax_order(cohort))
        _pe_bf = _pe(best_fixed_order(cohort))
        # Oracle follows the paper's |pi|<=k (skip-hurtful) rule -> monotone
        # ceiling; the fixed references cannot skip and use |pi|=k.
        _pe_ora = _pe(oracle_order(cohort), leq_k=True)
        pd.DataFrame([
            dict(reference="RelMax(label-free)", PE=_pe_rel),
            dict(reference="BestFixed(label-cond)", PE=_pe_bf),
            dict(reference="Oracle(per-instance)", PE=_pe_ora),
            dict(reference="gap: Oracle-BestFixed (adaptation)", PE=_pe_ora - _pe_bf),
            dict(reference="gap: BestFixed-RelMax (fixed-order choice)", PE=_pe_bf - _pe_rel),
            dict(reference="gap: Oracle-RelMax (total headroom)", PE=_pe_ora - _pe_rel),
        ]).to_csv(f"{args.outdir}/fixed_order_decomposition.csv", index=False)
        # definedness diagnosis -> definedness_diagnosis.csv
        # How much of Oracle-RelMax is schema lookup (knowing which channels are
        # defined for a variant) vs genuine per-instance planning (reading each
        # variant's channel values). This CSV reports whatever the CURRENT cohort
        # yields. On the real 491-variant ClinVar/gnomAD/AlphaMissense cohort it
        # split ~46% schema / ~54% genuine planning (see docs/variant/real_data.md); the
        # bundled synthetic demo gives a different, higher schema share.
        _pe_db = _pe(definedness_baseline_order(cohort))
        _pe_dsb = _pe(definedness_stratified_best_order(cohort))
        _tot = _pe_ora - _pe_rel
        _sch = _pe_dsb - _pe_rel
        _res = _pe_ora - _pe_dsb
        pd.DataFrame([
            dict(quantity="PE RelMax(label-free)", value=_pe_rel),
            dict(quantity="PE DefinednessBaseline(defined-first)", value=_pe_db),
            dict(quantity="PE DefinednessStratifiedBest(schema-optimal)", value=_pe_dsb),
            dict(quantity="PE Oracle(per-instance)", value=_pe_ora),
            dict(quantity="gap Oracle-RelMax (total)", value=_tot),
            dict(quantity="gap explained by definedness/schema", value=_sch),
            dict(quantity="gap residual = genuine per-instance planning", value=_res),
            dict(quantity="fraction schema-explained",
                 value=(_sch / _tot if abs(_tot) > 1e-9 else float("nan"))),
        ]).to_csv(f"{args.outdir}/definedness_diagnosis.csv", index=False)
        # (#2) two-mask memorization probe -> memorization_probe.csv
        pd.DataFrame([M.memorization_probe(cohort)]).to_csv(
            f"{args.outdir}/memorization_probe.csv", index=False)
        # (#3) paired agent-RelMax contrast CI per budget -> curve_contrast_ci.csv
        rel = relmax_order(cohort)
        if matrix:
            _, ag_order = harness.agent_and_order(cohort, matrix[0])
            rows = [dict(budget=k, **M.paired_contrast_ci(cohort, ag_order, rel, k))
                    for k in range(1, cohort.domain.K + 1)]
            pd.DataFrame(rows).to_csv(f"{args.outdir}/curve_contrast_ci.csv", index=False)
        # (#6) disease-domain stratification (variant domain only) -> domain_gap.csv
        if "domain" in cohort.df.columns and cohort.df["domain"].nunique() > 1:
            ds = M.domain_stratified_gap(cohort, k=1)
            pd.DataFrame([dict(domain=d, **v) for d, v in ds.items()]).to_csv(
                f"{args.outdir}/domain_gap.csv", index=False)
        # consequence stratification (variant domain): where does planning help?
        # Full cohort, split by is_missense, each stratum scored on its own
        # applicable action space, with a shuffled-label null per stratum.
        if "is_missense" in cohort.df.columns:
            cg = M.consequence_stratified_gap(cohort)
            if cg:
                def _fmt(k, val):
                    if k == "channels":
                        return ";".join(val)
                    if k == "per_k_gap":
                        return ";".join(f"{x:.4f}" for x in val)
                    return val
                pd.DataFrame([dict(consequence=c, **{k: _fmt(k, v[k]) for k in v})
                              for c, v in cg.items()]).to_csv(
                    f"{args.outdir}/consequence_gap.csv", index=False)
            # channel complementarity on the missense stratum: is any pair of
            # channels non-redundant (the precondition for planning to matter)?
            cc = M.channel_complementarity(cohort, restrict_missense=True)
            if cc:
                pd.DataFrame([dict(pair=p, **v) for p, v in cc.items()]).to_csv(
                    f"{args.outdir}/complementarity.csv", index=False)
        # (#7) evidence-pool ablation: drop each channel in turn -> ablation.csv
        ab_rows = []
        for ch in cohort.domain.channels:
            if len(cohort.domain.channels) > 1:
                a = M.evidence_pool_ablation(cohort, drop=[ch], k=1)
                ab_rows.append(dict(dropped=ch, **a))
        pd.DataFrame(ab_rows).to_csv(f"{args.outdir}/ablation.csv", index=False)
        # (#5) cost-model robustness -> cost_robustness.csv
        ref_orders = {"Oracle": oracle_order(cohort), "RelMax": rel,
                      "Heuristic": heuristic_order(cohort)}
        cr = M.cost_robustness(cohort, ref_orders)
        pd.DataFrame([dict(caliber=cal, rank="|".join(v["rank"]),
                           top_strategy_invariant=cr["ranking_invariant"],
                           full_order_invariant=cr["full_ranking_invariant"])
                      for cal, v in cr["per_caliber"].items()]
                     ).to_csv(f"{args.outdir}/cost_robustness.csv", index=False)
        # (#8) confidence-based stopping -> stopping.csv. Computed for Oracle
        # (the original reference) AND every real agent, so "how many queries
        # does this model actually need" is reported per agent, not only for
        # the oracle ceiling -- a query-count efficiency view that (unlike
        # cost_weighted_yield) needs no cost-tier assumption at all.
        stop_rows = []
        for strategy, ordr in [("Oracle", oracle_order(cohort))] + (
                [(m.get("slot"), harness.agent_and_order(cohort, m)[1]) for m in matrix]
                if matrix else []):
            cs = M.confidence_based_stopping(cohort, ordr)
            for margin, v in cs.items():
                stop_rows.append(dict(strategy=strategy, margin=margin, **v))
        pd.DataFrame(stop_rows).to_csv(f"{args.outdir}/stopping.csv", index=False)
        # (#4) multi-model agent-vs-RelMax contrasts, Holm-corrected across the
        # model family -> multimodel_contrast_holm.csv (bites once >=2 agents)
        if matrix:
            harness.multimodel_contrast_holm(cohort, matrix, budget=1).to_csv(
                f"{args.outdir}/multimodel_contrast_holm.csv", index=False)
        # (#10) trajectory diversity (of the agent if present, else oracle) -> trajectory.csv
        # recompute the order locally rather than reading ag_order from the
        # matrix branch above, so this block does not depend on that branch
        # having run (cache makes the re-fetch free).
        traj_order = (harness.agent_and_order(cohort, matrix[0])[1]
                      if matrix else oracle_order(cohort))
        td = M.trajectory_diversity(traj_order, cohort.domain.channels)
        pd.DataFrame([dict(n_variants=td["n_variants"],
                           n_distinct_orders=td["n_distinct_orders"],
                           **{f"first_{k}": v for k, v in td["first_query_counts"].items()})]
                     ).to_csv(f"{args.outdir}/trajectory.csv", index=False)
        # clinical-endpoint suite: diagnostic yield / VUS resolution,
        # guideline-native ACMG-points calls, and cost-weighted yield --
        # computed for RelMax (deployable reference) AND every real agent in
        # the matrix (not just RelMax), so the clinical numbers reflect actual
        # model behaviour. agent_and_order is memoized per cohort, so this
        # reuses whatever order was already elicited earlier in this same run
        # (main loop / curve_contrast_ci / multimodel_contrast_holm) instead of
        # re-calling the API -> clinical_yield.csv
        # BestFixed alongside RelMax: RelMax is the guideline-prior LOWER
        # reference (deployable, label-free, but not necessarily the tightest
        # fixed order on this specific cohort); BestFixed is the tightest
        # fixed order the metric itself can find (label-conditioned). Testing
        # agents against BestFixed too (not just RelMax) answers "does the
        # yield gain survive against the strongest non-adaptive baseline", not
        # just the naive guideline-prior one.
        clinical_targets = [("RelMax", rel), ("BestFixed", best_fixed_order(cohort))]
        if matrix:
            for m in matrix:
                _, ag_ord = harness.agent_and_order(cohort, m)
                clinical_targets.append((m.get("slot"), ag_ord))
        cy_rows = []
        cyc_by_strategy = {}   # keep the full curve dict (incl. per-variant
                               # `call` arrays) around for the significance/
                               # domain-breakdown blocks below, so those don't
                               # have to recompute clinical_yield_curve.
        for strategy, ordr in clinical_targets:
            cyc = M.clinical_yield_curve(cohort, ordr)
            apc = M.acmg_points_curve(cohort, ordr)
            # cost_weighted_yield: KEPT for transparency, but its PM1=10x
            # cost tier is an illustrative, unvalidated assumption that does
            # not match how PM1 is actually computed in this cohort (a
            # UniProt database lookup,
            # same cost class as PM2/PP3, not a wet-lab assay). Do NOT cite
            # cum_cost/resolved_per_cost as a deployment-cost finding until a
            # real, sourced cost model replaces the default cost_tiers.
            cwy = M.cost_weighted_yield(cohort, ordr)
            cyc_by_strategy[strategy] = (cyc, ordr)
            for k in sorted(cyc):
                cy_rows.append(dict(
                    strategy=strategy, budget=k,
                    logistic_resolved=cyc[k]["resolved_frac"],
                    logistic_call_acc=cyc[k]["call_accuracy"],
                    logistic_vus=cyc[k]["vus_frac"],
                    acmg_resolved=apc[k]["resolved_frac"],
                    acmg_call_acc=apc[k]["call_accuracy"],
                    cum_cost=cwy[k]["cum_cost"],
                    resolved_per_cost=cwy[k]["resolved_per_cost"]))
        pd.DataFrame(cy_rows).to_csv(f"{args.outdir}/clinical_yield.csv", index=False)

        # (#11) paired significance test of diagnostic-yield gain, every budget
        # -> yield_significance.csv. Closes the "point estimate, no CI" gap AND
        # the "is RelMax an easy strawman" concern: for each real agent,
        # McNemar-style paired test (on the `call` arrays clinical_yield_curve
        # now returns) against BOTH RelMax (the deployable guideline-prior
        # lower reference) and BestFixed (the tightest fixed order the metric
        # can find on this cohort) -- so "beats the naive baseline" and "beats
        # the best possible non-adaptive baseline" are reported separately.
        if matrix:
            ref_cycs = {"RelMax": cyc_by_strategy["RelMax"][0],
                       "BestFixed": cyc_by_strategy["BestFixed"][0]}
            sig_rows = []
            for m in matrix:
                slot = m.get("slot")
                agent_cyc = cyc_by_strategy[slot][0]
                for ref_name, ref_cyc in ref_cycs.items():
                    for k in sorted(agent_cyc):
                        t = M.paired_resolved_test(agent_cyc[k]["call"], ref_cyc[k]["call"])
                        sig_rows.append(dict(strategy=slot, reference=ref_name, budget=k,
                                             agent_resolved_frac=agent_cyc[k]["resolved_frac"],
                                             reference_resolved_frac=ref_cyc[k]["resolved_frac"],
                                             **t))
            pd.DataFrame(sig_rows).to_csv(f"{args.outdir}/yield_significance.csv", index=False)

        # (#12) confidence-threshold sensitivity at k=3 -> yield_sensitivity.csv.
        # Same operating point (confidence=0.90) drives every number above;
        # this checks the k=3 finding is not an artifact of that one cutoff.
        if matrix:
            sens_rows = []
            for strategy, (_, ordr) in cyc_by_strategy.items():
                cys = M.clinical_yield_sensitivity(cohort, ordr, k=3)
                for conf, v in cys.items():
                    v = {kk: vv for kk, vv in v.items() if kk != "call"}
                    sens_rows.append(dict(strategy=strategy, confidence=conf, **v))
            pd.DataFrame(sens_rows).to_csv(f"{args.outdir}/yield_sensitivity.csv", index=False)

        # (#13) diagnostic yield by disease domain, k=3, per real agent
        # -> yield_by_domain.csv. Agent-specific analogue of domain_gap.csv
        # (which is Oracle-vs-RelMax only): does THIS model's actual yield
        # differ by disease domain?
        if matrix and "domain" in cohort.df.columns and cohort.df["domain"].nunique() > 1:
            dom_rows = []
            for strategy, (_, ordr) in cyc_by_strategy.items():
                byd = M.clinical_yield_by_domain(cohort, ordr, k=3)
                for d, v in byd.items():
                    dom_rows.append(dict(strategy=strategy, domain=d, **v))
            pd.DataFrame(dom_rows).to_csv(f"{args.outdir}/yield_by_domain.csv", index=False)

        print(f"[{args.domain}] robustness -> memorization_probe.csv, "
              "curve_contrast_ci.csv, domain_gap.csv, consequence_gap.csv, "
              "complementarity.csv, ablation.csv, cost_robustness.csv, "
              "stopping.csv, trajectory.csv, clinical_yield.csv, "
              "yield_significance.csv, yield_sensitivity.csv, yield_by_domain.csv")

    # figures (reuse the already-built agent + order; no re-elicitation)
    agent_order = None
    if matrix:
        _, agent_order = harness.agent_and_order(cohort, matrix[0])
    budgets = list(range(1, cohort.domain.K + 1))
    figures.fig_budget_curve(cohort, agent_order, budgets, f"{args.figdir}/budget_curve.png")
    figures.fig_single_slot(cohort, f"{args.figdir}/single_slot.png")
    if agent_order is not None:
        figures.fig_order_hist(cohort, agent_order, f"{args.figdir}/order_optimality.png")
        # watermark the scaling panel when the matrix is mock/demo: the mock's
        # capability axis is a stand-in and produces an inverted trend that must
        # not be mistaken for a real finding.
        mock_matrix = all(m.get("kind") == "mock" for m in matrix)
        figures.fig_scaling(df, f"{args.figdir}/scaling_gap.png", watermark=mock_matrix)

    print(f"[{args.domain}] wrote results -> {args.outdir}/  figures -> {args.figdir}/")
    show = ["strategy", "model_id", "A_k1", "PE", "delta_oracle",
            "po_vs_relevance", "po_vs_oracle",
            "rho_vs_relevance", "rho_vs_oracle"]
    print(df[[c for c in show if c in df.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
