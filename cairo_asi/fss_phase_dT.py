"""(d, T) plane near the A boundary from the PT runs: d = 0.145 FSS (n <= 24) and n = 12 runs at d = 0.140, 0.1475, 0.150, 0.1527.
-> fss_dT_phase.png, fss_dscan_n12.json"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = json.load(open("fss_summary_d0.1450.json"))
out = {}
for d in ["0.1400", "0.1475", "0.1500", "0.1527"]:
    R = json.load(open(f"fss_d{d}_n12.json")); a, c = R["runs"]["A"], R["runs"]["C"]
    out[d] = dict(T_Cmax=R["T_Cmax"], Cmax=R["Cmax"], Tstar=R["Tstar"], dF=R["dF"], A_vs_C_maxabsz=R["A_vs_C_maxabsz"],
                  Tstar_A_start=a["reweighted"]["Tstar"], dF_A_start=a["reweighted"]["dF"], Tstar_C_start=c["reweighted"]["Tstar"],
                  trips=[a["round_trips"], c["round_trips"]], min_n_over_tau=[min(a["n_over_tau"]), min(c["n_over_tau"])],
                  T=a["T"], psiA_A=a["psiA_abs"], psiC_A=a["psiC_abs"], psiA_C=c["psiA_abs"], psiC_C=c["psiC_abs"])
json.dump(out, open("fss_dscan_n12.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(13, 6.4))
a = ax[0]
a.errorbar([0.145], [S["Tstar_inf"][0]], [0.0003], fmt="o", color="k", ms=8, label="d=0.145: first-order A|X, T*(∞) (n ≤ 24)")
a.plot([0.140], [out["0.1400"]["T_Cmax"]], "v", color="C0", ms=9, label="d=0.140: A melting, C_max (no bimodal P(E), n=12)")
a.plot([0.1475], [out["0.1475"]["Tstar_A_start"]], "s", color="C3", ms=8, mfc="none", label="d=0.1475: A|C-like T* (A-start, n=12; C-start trapped)")
a.plot([0.150], [out["0.1500"]["T_Cmax"]], "D", color="C2", ms=8, mfc="none", label="d=0.150: C → mixed A/C superstructure (C_max, n=12, glassy)")
a.plot([0.1475, 0.145, 0.140], [out["0.1475"]["Tstar_A_start"], S["Tstar_inf"][0], out["0.1400"]["T_Cmax"]], ":", color="0.5")
a.axvspan(0.1475, 0.150, color="0.9", zorder=0); a.text(0.1478, 0.03, "A lost at T→0\n(n=12)", fontsize=8)
a.axvline(0.153, color="C1", ls="--", lw=0.8); a.text(0.1532, 0.16, "d_c = 0.153\n(8x8, T=0)", fontsize=8, color="C1")
a.text(0.1405, 0.07, "A", fontsize=16); a.text(0.1505, 0.13, "C-like", fontsize=13); a.text(0.1405, 0.175, "fluctuating A/C\n(C order only at T ≳ 0.2)", fontsize=8)
a.set_xlabel("d"); a.set_ylabel("T"); a.set_xlim(0.139, 0.154); a.set_ylim(0, 0.2); a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
a = ax[1]
for d, col in zip(out, ["C0", "C1", "C2", "C3"]):
    T = np.array(out[d]["T"]); a.plot(T, out[d]["psiA_A"], "-", color=col, label=f"d={d} ⟨|ψ_A|⟩"); a.plot(T, out[d]["psiC_A"], "--", color=col, label=f"d={d} ⟨|ψ_C|⟩")
R = json.load(open("fss_d0.1450_n12.json"))["runs"]["A"]; a.plot(R["T"], R["psiA_abs"], "-", color="k", lw=0.8, label="d=0.145 ⟨|ψ_A|⟩"); a.plot(R["T"], R["psiC_abs"], "--", color="k", lw=0.8)
a.set_xlabel("T"); a.set_ylabel("order parameters (A-start runs, n = 12)"); a.set_xlim(0, 0.3); a.legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig("fss_dT_phase.png", dpi=110)
