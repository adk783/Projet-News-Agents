"""
run_eval.py - Point d'entree unifie du framework d'evaluation.
Usage:
  python eval/run_eval.py --layer 1             # Tests unitaires, zero API, <3s
  python eval/run_eval.py --layer 2 --limit 5   # Pipeline complet, 5 articles
  python eval/run_eval.py --layer 3             # Validation marche via yfinance
  python eval/run_eval.py --layer 4             # Robustesse (anonymisation, stress, reproductibilite)
  python eval/run_eval.py --layer 5             # Financier (AR/CAR, calibration, portefeuille)
  python eval/run_eval.py --layer 6             # Intégrité (faithfulness, causality)
  python eval/run_eval.py --layer 7             # Évaluations unitaires et retriever
  python eval/run_eval.py --layer 8 --limit 3   # Architecture multi-agents (FinGAIA + contrefactuel)
  python eval/run_eval.py --layer 9             # Signal vs Marché (court/moyen/long terme)
  python eval/run_eval.py --layer 10            # Frais d'exécution réels (slippage, spread, commission)
  python eval/run_eval.py --layer 11            # Latence décisionnelle et signal decay
  python eval/run_eval.py --layer 12            # Robustesse aux régimes de marché (Bull/Bear/HV/SW)
  python eval/run_eval.py --all                 # Tout
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def run_layer1() -> dict:
    from eval.benchmark_structural import run_structural_benchmark

    return run_structural_benchmark()


def run_layer2(limit: int = 0) -> dict:
    from eval.evaluate_pipeline import run_pipeline_benchmark

    return run_pipeline_benchmark(limit=limit)


def run_layer3(horizon: int = 5) -> dict:
    from eval.evaluate_market import run_market_validation

    return run_market_validation(horizon_days=horizon)


def run_layer4(n_articles: int = 3, n_runs: int = 3, tests=None) -> dict:
    from eval.benchmark_robustness import run_robustness_benchmark

    return run_robustness_benchmark(n_articles=n_articles, n_runs=n_runs, tests=tests)


def run_layer5(sub: list = None, n_runs: int = 3, n_articles: int = 2) -> dict:
    """
    Couche 5 : Analyse financiere avancee.
    Sub-modules : ar (rendements anormaux), calibration, portfolio, debate
    """
    sub = sub or ["ar", "calibration", "portfolio", "debate"]
    result = {"layer": 5}

    if "ar" in sub:
        from eval.evaluate_abnormal_returns import run_abnormal_returns_analysis

        result["ar"] = run_abnormal_returns_analysis()

    if "calibration" in sub:
        from eval.evaluate_calibration import run_calibration_analysis

        result["calibration"] = run_calibration_analysis(
            n_runs_consistency=n_runs,
            n_articles_consistency=n_articles,
        )

    if "portfolio" in sub:
        from eval.evaluate_portfolio import run_portfolio_analysis

        result["portfolio"] = run_portfolio_analysis()

    if "debate" in sub:
        from eval.evaluate_debate_dynamics import run_debate_dynamics_analysis

        result["debate"] = run_debate_dynamics_analysis()

    return result


def run_layer6(sub: list = None, limit: int = 10) -> dict:
    """
    Couche 6 : Hallucinations (Faithfulness NLI) et Propagation des erreurs.
    Sub-modules : faithfulness, causality
    """
    sub = sub or ["faithfulness", "causality"]
    result = {"layer": 6}

    if "faithfulness" in sub:
        from eval.evaluate_faithfulness import run_faithfulness_analysis

        result["faithfulness"] = run_faithfulness_analysis(limit=limit)

    if "causality" in sub:
        from eval.evaluate_error_propagation import analyze_causality

        result["causality"] = analyze_causality()

    return result


def run_layer7(sub: list = None, limit: int = 0) -> dict:
    """
    Couche 7 : Benchmarks unitaires et Retriever.
    Sub-modules : unit, retriever
    """
    sub = sub or ["unit", "retriever"]
    result = {"layer": 7}

    if "unit" in sub:
        from eval.evaluate_unit_llm import run_unit_llm_benchmark

        result["unit"] = run_unit_llm_benchmark(limit=limit)

    if "retriever" in sub:
        from eval.evaluate_retriever import run_retriever_evaluation

        result["retriever"] = run_retriever_evaluation(limit=limit)

    return result


def run_layer8(limit: int = 0) -> dict:
    """
    Couche 8 : Architecture multi-agents & Mise à l'échelle.
    Benchmark comparatif FinGAIA-inspired + évaluation contrefactuelle.
    """
    from eval.evaluate_multiagent_arch import run_multiagent_arch_analysis

    return run_multiagent_arch_analysis(limit=limit)


def run_layer9() -> dict:
    """
    Couche 9 : Comparaison Signal Pipeline vs Réalité Marché.
    Analyse statistique sur 3 horizons : court (1-5j), moyen (10-30j), long (60-120j).
    Métriques : Accuracy, IC Spearman, Hit Rate, Confusion Matrix, P&L vs SPY.
    """
    from eval.evaluate_signal_vs_market import run_signal_vs_market_analysis

    return run_signal_vs_market_analysis()


def run_layer10(position_eur: float = 5000.0) -> dict:
    """
    Couche 10 : Rentabilité nette après frais d'exécution réels.
    Modélise commission + spread bid-ask + market impact + slippage.
    Compare 3 profils de courtage (Retail, Semi-Pro, Pro).
    Calcule le break-even accuracy et le turnover annualisé.
    Référence : Kissell & Glantz (2003), Almgren & Chriss (2001).
    """
    from eval.evaluate_execution_costs import run_execution_costs_analysis

    return run_execution_costs_analysis(position_eur=position_eur)


def run_layer11() -> dict:
    """
    Couche 11 : Latence décisionnelle et dégradation du signal (alpha decay).
    Analyse la distribution temporelle des publications, la performance par
    tranche horaire, et ajuste un modèle exponentiel de décroissance de l'alpha.
    Référence : Mitchell & Mulherin (1994), Fama (1970).
    """
    from eval.evaluate_latency import run_latency_analysis

    return run_latency_analysis()


def run_layer12() -> dict:
    """
    Couche 12 : Robustesse aux régimes de marché.
    Classifie chaque trade en régime Bull / Bear / Haute Volatilité / Latéral
    sur la base du VIX et du return SPY 20j, puis calcule Sharpe, Sortino,
    MaxDD et alpha par régime. Analyse la stabilité inter-régimes.
    Référence : Hamilton (1989), Lo (2004), Ang & Bekaert (2002).
    """
    from eval.evaluate_market_regimes import run_market_regime_analysis

    return run_market_regime_analysis()


def _print_summary(results: dict) -> None:
    """Affiche un résumé cross-couches si plusieurs couches ont été lancées."""
    print("\n" + "=" * 70)
    print("RÉSUMÉ GLOBAL DU FRAMEWORK D'ÉVALUATION")
    print("=" * 70)

    if 1 in results:
        r = results[1]
        status = "PASS" if r["failed"] == 0 else f"FAIL ({r['failed']} échec(s))"
        print(f"  L1 — Structurel : {r['passed']}/{r['total']} tests | {status}")

    if 2 in results:
        r = results[2]
        acc = r.get("accuracy")
        acc_str = f"{acc:.0%}" if acc is not None else "N/A"
        print(f"  L2 — Pipeline   : Accuracy signaux = {acc_str}")
        for cls in ["Achat", "Vente", "Neutre"]:
            f1 = r.get(cls, {}).get("f1")
            if f1 is not None:
                print(f"       F1 {cls:<6} = {f1:.2f}")

    if 3 in results:
        r = results[3]
        acc = r.get("global_accuracy", 0)
        n = r.get("n_validations", 0)
        ret = sum(r.get("returns_faible", [])) if r.get("returns_faible") else None
        print(
            f"  L3 — Marché     : Accuracy={acc:.0%} ({n} décisions validées) "
            + (f"| P&L FAIBLE={ret:+.2f}%" if ret is not None else "| P&L: N/A")
        )

    if 4 in results:
        r = results[4]
        bias = r.get("anonymization", {}).get("bias_rate", "N/A")
        stab = r.get("stress", {}).get("avg_stability", "N/A")
        var = r.get("reproducibility", {}).get("avg_variance", "N/A")
        bias_str = f"{bias:.0%}" if isinstance(bias, float) else str(bias)
        stab_str = f"{stab:.0%}" if isinstance(stab, float) else str(stab)
        var_str = f"{var:.0%}" if isinstance(var, float) else str(var)
        print(f"  L4 -- Robustesse : Biais={bias_str} | Stabilite stress={stab_str} | Variance={var_str}")

    if 5 in results:
        r = results[5]
        ar = r.get("ar", {}).get("n_significant", "N/A")
        ar_n = r.get("ar", {}).get("n_predictions", 0)
        bs = r.get("calibration", {}).get("brier_benchmark", {}).get("brier_score", "N/A")
        sh = r.get("portfolio", {}).get("portfolio_faible", {}).get("sharpe", "N/A")
        mdd = r.get("portfolio", {}).get("portfolio_faible", {}).get("max_drawdown", "N/A")
        bs_str = f"{bs:.3f}" if isinstance(bs, float) else str(bs)
        sh_str = f"{sh:.2f}" if isinstance(sh, float) else str(sh)
        mdd_str = f"{mdd:.1f}%" if isinstance(mdd, float) else str(mdd)
        ar_str = f"{ar}/{ar_n}" if isinstance(ar, int) else str(ar)

        deb_q = r.get("debate", {}).get("avg_quality", "N/A")
        deb_q_str = f"{deb_q:.2f}" if isinstance(deb_q, float) else str(deb_q)

        print(
            f"  L5 -- Financier  : CAR signif={ar_str} | BrierScore={bs_str} | Sharpe={sh_str} | MaxDD={mdd_str} | Debat={deb_q_str}"
        )

    if 6 in results:
        r = results[6]
        f_score = r.get("faithfulness", {}).get("faithfulness_score", "N/A")
        f_score_str = f"{f_score:.2f}" if isinstance(f_score, float) else str(f_score)
        err_tot = r.get("causality", {}).get("total_errors", "N/A")
        err_fil = r.get("causality", {}).get("cause_filtrage", 0)
        err_abs = r.get("causality", {}).get("cause_absa", 0)
        err_deb = r.get("causality", {}).get("cause_debate", 0)
        err_con = r.get("causality", {}).get("cause_consensus", 0)

        print(
            f"  L6 -- Integrite  : Faithfulness={f_score_str} | FailTot={err_tot} [Filt:{err_fil} ABSA:{err_abs} Debat:{err_deb} Cons:{err_con}]"
        )

    if 7 in results:
        r = results[7]
        unit = r.get("unit", {})
        acc_unit = unit.get("accuracy", "N/A")
        f1_unit = unit.get("weighted_f1", "N/A")
        kappa = unit.get("cohen_kappa", "N/A")
        kl = unit.get("kl_divergence", "N/A")
        acc_unit_str = f"{acc_unit:.0%}" if isinstance(acc_unit, float) else str(acc_unit)
        f1_unit_str = f"{f1_unit:.3f}" if isinstance(f1_unit, float) else str(f1_unit)
        kappa_str = f"{kappa:.3f}" if isinstance(kappa, float) else str(kappa)
        kl_str = f"{kl:.3f}" if isinstance(kl, float) else str(kl)

        ret = r.get("retriever", {})
        prec = ret.get("context_precision", "N/A")
        rec = ret.get("context_recall", "N/A")
        ndcg = ret.get("ndcg", "N/A")
        prec_str = f"{prec:.3f}" if isinstance(prec, float) else str(prec)
        rec_str = f"{rec:.3f}" if isinstance(rec, float) else str(rec)
        ndcg_str = f"{ndcg:.3f}" if isinstance(ndcg, float) else str(ndcg)

        print(
            f"  L7 -- Unit & Retr: LLM[Acc={acc_unit_str} F1={f1_unit_str} Kappa={kappa_str} KL={kl_str}] | Retr[Prec={prec_str} Rec={rec_str} NDCG={ndcg_str}]"
        )

    if 8 in results:
        r = results[8]
        fg = r.get("fingaia", {})
        cf = r.get("counterfactual", {})
        fg_score = fg.get("fingaia_score", "N/A")
        best_cfg = fg.get("best_config", "N/A")
        fae = fg.get("fae_solo_to_d3", "N/A")
        most_crit = cf.get("most_critical_agent", "N/A")
        fg_str = f"{fg_score:.3f}" if isinstance(fg_score, float) else str(fg_score)
        fae_str = f"{fae:.3f}" if isinstance(fae, float) else str(fae)
        print(
            f"  L8 -- Multi-Agent: FinGAIA={fg_str} | Meilleure config={best_cfg} | FAE={fae_str} | Agent critique={most_crit}"
        )

    if 9 in results:
        r = results[9]
        h = r.get("horizons", {})
        lines = []
        for hk, label in [("COURT_TERME", "CT"), ("MOYEN_TERME", "MT"), ("LONG_TERME", "LT")]:
            hd = h.get(hk, {})
            acc = hd.get("accuracy")
            ic = hd.get("ic")
            acc_s = f"{acc:.1%}" if acc is not None else "N/A"
            ic_s = f"{ic:+.3f}" if ic is not None else "N/A"
            lines.append(f"{label}[Acc={acc_s} IC={ic_s}]")
        print(f"  L9 -- Sig vs Mkt : {' | '.join(lines)}")

    if 10 in results:
        r = results[10]
        rb = r.get("results_by_broker", {})
        parts = []
        for k, v in rb.items():
            net = v.get("net_total")
            sh = v.get("net_sharpe")
            parts.append(f"{k}[Net={f'{net:+.2f}%' if net is not None else 'N/A'} Sh={sh or 'N/A'}]")
        print(f"  L10 - Exec Costs : {' | '.join(parts)}")

    if 11 in results:
        r = results[11]
        n_art = r.get("n_articles", 0)
        decay = r.get("decay_analysis", {}).get("decay_model", {})
        hl = decay.get("half_life_hours")
        hl_str = f"{hl:.1f}sem" if hl is not None else "N/A"
        print(f"  L11 - Latence    : {n_art} articles analysés | Alpha half-life = {hl_str}")

    if 12 in results:
        r = results[12]
        rob = r.get("robustness", {})
        verdict = rob.get("verdict", "N/A")
        best = rob.get("best_regime", "N/A")
        worst = rob.get("worst_regime", "N/A")
        sh_std = rob.get("sharpe_std", "N/A")
        sh_std_s = f"{sh_std:.3f}" if isinstance(sh_std, float) else str(sh_std)
        print(f"  L12 - Régimes    : {verdict} | Best={best} | Worst={worst} | SharpeStd={sh_std_s}")

    print("=" * 70)

    # Sauvegarde du résumé
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(__file__).parent / "eval_results" / f"{timestamp}_full_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # Convertit les clés int en str pour JSON
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, ensure_ascii=False)
    print(f"  Résumé sauvegardé : {out_path}")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework d'évaluation multi-couches de l'architecture financière.")
    parser.add_argument(
        "--layer",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help=(
            "Couche a lancer : 1=structurel, 2=pipeline, 3=marche, 4=robustesse, "
            "5=financier, 6=hallucination/causalite, 7=unitaire/retriever, "
            "8=architecture-multi-agents, 9=signal-vs-marche, "
            "10=frais-execution, 11=latence-signal-decay, 12=regimes-marche"
        ),
    )
    parser.add_argument(
        "--position_eur",
        type=float,
        default=5000.0,
        help="Taille de position par trade en € pour la couche 10 (défaut: 5000)",
    )
    parser.add_argument("--all", action="store_true", help="Lance toutes les couches")
    parser.add_argument("--limit", type=int, default=0, help="Nb d'articles pour la couche 2 (0=tous)")
    parser.add_argument("--horizon", type=int, default=5, help="Horizon en jours pour la couche 3")
    parser.add_argument("--n-articles", type=int, default=3, help="Nb d'articles pour les couches 4 et 5")
    parser.add_argument(
        "--n-runs", type=int, default=3, help="Nb de runs par article (L4 reproductibilite, L5 calibration)"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["anonymization", "temporal", "stress", "reproducibility"],
        default=None,
        help="Tests specifiques pour --layer 4",
    )
    parser.add_argument(
        "--sub",
        nargs="+",
        choices=["ar", "calibration", "portfolio", "debate"],
        default=None,
        help="Sous-modules pour --layer 5 (ar=rendements anormaux, calibration, portfolio, debate)",
    )
    parser.add_argument(
        "--sub6",
        nargs="+",
        choices=["faithfulness", "causality"],
        default=None,
        help="Sous-modules pour --layer 6 (faithfulness, causality)",
    )
    parser.add_argument(
        "--sub7",
        nargs="+",
        choices=["unit", "retriever"],
        default=None,
        help="Sous-modules pour --layer 7 (unit, retriever)",
    )
    args = parser.parse_args()

    if not args.layer and not args.all:
        parser.print_help()
        sys.exit(0)

    results = {}

    if args.all or args.layer == 1:
        results[1] = run_layer1()

    if args.all or args.layer == 2:
        results[2] = run_layer2(limit=args.limit)

    if args.all or args.layer == 3:
        results[3] = run_layer3(horizon=args.horizon)

    if args.all or args.layer == 4:
        results[4] = run_layer4(
            n_articles=args.n_articles,
            n_runs=args.n_runs,
            tests=args.tests,
        )

    if args.all or args.layer == 5:
        results[5] = run_layer5(
            sub=args.sub,
            n_runs=args.n_runs,
            n_articles=args.n_articles,
        )

    if args.all or args.layer == 6:
        results[6] = run_layer6(
            sub=args.sub6,
            limit=10 if args.limit == 0 else args.limit,
        )

    if args.all or args.layer == 7:
        results[7] = run_layer7(
            sub=args.sub7,
            limit=args.limit,
        )

    if args.all or args.layer == 8:
        results[8] = run_layer8(limit=args.limit)

    if args.all or args.layer == 9:
        results[9] = run_layer9()

    if args.all or args.layer == 10:
        results[10] = run_layer10(position_eur=args.position_eur)

    if args.all or args.layer == 11:
        results[11] = run_layer11()

    if args.all or args.layer == 12:
        results[12] = run_layer12()

    if len(results) > 1:
        _print_summary(results)
    elif len(results) == 1:
        # Un seul résultat — pas besoin d'un résumé cross-couches
        pass

    # Code de sortie : 1 si des tests L1 ont échoué
    if 1 in results and results[1]["failed"] > 0:
        sys.exit(1)
