"""
IFRS 9 Loan-Level Synthetic Data Generator — v5
Changes from v4:
  (1) 60 months (5 years) of observation.
  (2) Originations staggered over 36 months (2022-2024).
  (3) Systemic Macro Recession shock in 2025 H1 (Months 37-43).
      - Unemployment +4.5%
      - Repo Rate +2.5%
      - GDP drops -2.0%
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json, warnings
warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────
SEED, N_LOANS, N_MONTHS = 42, 20_000, 60         # 60 months (5 years)
START_DATE = pd.Timestamp("2022-01-01")          # Starts 2022
BASE = Path(r"C:\CreditRiskCourse\01_microcredit_portfolio")
RAW_PQ, RAW_XL, META = BASE/"data/raw/parquet", BASE/"data/raw/excel", BASE/"metadata"
for p in [RAW_PQ, RAW_XL, META, BASE/"scripts", BASE/"outputs/summaries", BASE/"notebooks"]:
    p.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# ── 1. LOAN MASTER ──────────────────────────────────────────
print("[1/6] Loan master...")
regions = ["Gauteng","Western Cape","KwaZulu-Natal","Eastern Cape","Limpopo",
           "Mpumalanga","North West","Free State","Northern Cape"]
region_w = [0.28,0.15,0.15,0.12,0.08,0.07,0.06,0.05,0.04]
region_eff = dict(zip(regions,[-0.10,-0.15,0.05,0.20,0.30,0.10,0.15,0.05,0.25]))
sectors = ["Agriculture","Agro-processing","Trade/Retail","Manufacturing",
           "Services","Transport","Construction","Tourism","ICT"]
sector_w = [0.14,0.08,0.22,0.10,0.20,0.08,0.08,0.05,0.05]
sector_eff = dict(zip(sectors,[0.35,0.15,0.05,-0.05,-0.10,0.10,0.25,0.20,-0.20]))
borrowers = ["Individual micro","Micro-enterprise","Small enterprise","Medium enterprise","Cooperative"]
borrower_w = [0.25,0.30,0.25,0.12,0.08]
borrower_eff = dict(zip(borrowers,[0.20,0.10,-0.05,-0.25,0.05]))
borrower_size = dict(zip(borrowers,[0.4,0.8,1.5,4.0,1.2]))
purposes = ["Working capital","Asset finance","Trade finance","Expansion capex",
            "Emergency/consumption","Agricultural input"]
purpose_w = [0.32,0.18,0.10,0.15,0.10,0.15]
collaterals = ["None","Movable assets","Property","Cash/Receivables","DFI guarantee","Group guarantee"]
collateral_w = [0.35,0.20,0.10,0.10,0.15,0.10]
collateral_eff = dict(zip(collaterals,[0.20,0.00,-0.20,-0.10,-0.25,-0.15]))
collateral_recovery = dict(zip(collaterals,[0.05,0.30,0.55,0.40,0.70,0.35]))

n = N_LOANS
df = pd.DataFrame({
    "loan_id": [f"L{i:06d}" for i in range(1,n+1)],
    "borrower_id": [f"B{i:06d}" for i in range(1,n+1)],
    "region": np.random.choice(regions,n,p=region_w),
    "sector": np.random.choice(sectors,n,p=sector_w),
    "borrower_type": np.random.choice(borrowers,n,p=borrower_w),
    "loan_purpose": np.random.choice(purposes,n,p=purpose_w),
    "collateral_type": np.random.choice(collaterals,n,p=collateral_w),
})
df["loan_amount_initial"] = np.round(
    np.random.lognormal(np.log(2000),0.7,n) * df["borrower_type"].map(borrower_size).values, -2
).clip(200, 50000)
df["term_months"] = np.random.choice([6,12,18,24,36,48,60],n,p=[0.10,0.20,0.15,0.25,0.15,0.10,0.05])
premium = (df["sector"].map(sector_eff).fillna(0)*0.15
           + df["collateral_type"].map(collateral_recovery).fillna(0.02)*0.05)
df["interest_rate_annual"] = (0.12 + premium + np.random.normal(0,0.01,n)).clip(0.08,0.28).round(4)

# 36 months of staggered origination
offset = np.random.randint(0, 36, n)
df["origination_date"] = START_DATE + pd.to_timedelta(offset*30, unit="D")

df["dfi_funded"]           = (np.random.rand(n)<0.25).astype(int)
df["green_finance"]        = (np.random.rand(n)<0.12).astype(int)
df["women_owned"]          = (np.random.rand(n)<0.35).astype(int)
df["youth_owned"]          = (np.random.rand(n)<0.20).astype(int)
df["first_time_borrower"]  = (np.random.rand(n)<0.18).astype(int)
df["group_lending"]        = ((df["collateral_type"]=="Group guarantee") | (np.random.rand(n)<0.10)).astype(int)
df["climate_exposed"]      = ((df["sector"].isin(["Agriculture","Tourism"])) &
                              (df["region"].isin(["Limpopo","Eastern Cape","Northern Cape","Free State"]))).astype(int)

log_odds = (-3.5
    + df["region"].map(region_eff).fillna(0)
    + df["sector"].map(sector_eff).fillna(0)
    + df["borrower_type"].map(borrower_eff).fillna(0)
    + df["collateral_type"].map(collateral_eff).fillna(0)
    + 0.15*df["dfi_funded"] + 0.25*df["first_time_borrower"]
    + 0.20*df["climate_exposed"]
    + 0.10*(df["loan_purpose"]=="Emergency/consumption").astype(int)
    + 0.08*(df["loan_purpose"]=="Agricultural input").astype(int)
    - 0.00015*df["loan_amount_initial"]
    + np.random.normal(0,0.35,n))
df["true_annual_pd"]  = (1/(1+np.exp(-log_odds))).round(4)
df["true_monthly_pd"] = (1-(1-df["true_annual_pd"])**(1/12)).round(6)
df.to_parquet(RAW_PQ/"loan_master.parquet", index=False)
print(f"  {len(df):,} loans | mean PD {df['true_annual_pd'].mean():.2%}")

# ── 2. MACRO ────────────────────────────────────────────────
print("[2/6] Macro scenarios...")
months = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")
macro = pd.DataFrame({"month": months, "month_id": range(1,N_MONTHS+1),
                      "year": months.year, "month_num": months.month})
macro["repo_rate"] = (np.linspace(7.0,8.25,N_MONTHS) + np.random.normal(0,0.15,N_MONTHS)).round(2)
macro["prime_rate"] = (macro["repo_rate"]+3.5).round(2)
macro["gdp_growth_yoy"] = (1.2 + 0.6*np.sin(np.arange(N_MONTHS)/6) + np.random.normal(0,0.2,N_MONTHS)).round(2)
macro["unemployment"] = (32.5 + 0.3*np.sin(np.arange(N_MONTHS)/8) + np.random.normal(0,0.3,N_MONTHS)).round(2)
macro["agri_commodity_index"] = (100 + np.random.normal(0,3,N_MONTHS).cumsum()*0.5).round(2)
macro["tourism_index"]        = (100 + np.random.normal(0,4,N_MONTHS).cumsum()*0.6).round(2)
macro["retail_index"]         = (100 + np.random.normal(0,2,N_MONTHS).cumsum()*0.4).round(2)

# 2025 Macro Recession (Months 37-43)
macro["recession_flag"] = 0
recession_months = macro["month_id"].between(37, 43)
macro.loc[recession_months, "recession_flag"] = 1

# Systemic Shocks
macro.loc[recession_months, "unemployment"] = (macro.loc[recession_months, "unemployment"] + 4.5).round(2)
macro.loc[recession_months, "repo_rate"]    = (macro.loc[recession_months, "repo_rate"] + 2.5).round(2)
macro.loc[recession_months, "prime_rate"]   = (macro.loc[recession_months, "repo_rate"] + 3.5).round(2)
macro.loc[recession_months, "gdp_growth_yoy"] = (macro.loc[recession_months, "gdp_growth_yoy"] - 2.0).round(2)

macro["scenario"] = np.where(macro["recession_flag"]==1, "Macro Recession (2025 H1)", "Base")
macro.to_parquet(RAW_PQ/"macro_scenarios.parquet", index=False)
print(f"  {len(macro)} months; {macro['recession_flag'].sum()} recession months")

# ── 3. MONTHLY PERFORMANCE ──────────────────────────────────
print("[3/6] Simulating monthly performance (this takes ~2-3 mins)...")
loan_ids  = df["loan_id"].values
sectors_a = df["sector"].values
true_pd_m = df["true_monthly_pd"].values
terms     = df["term_months"].values
rates     = df["interest_rate_annual"].values/12
amounts   = df["loan_amount_initial"].values
climate   = df["climate_exposed"].values.astype(bool)
agri_mask    = np.isin(sectors_a, ["Agriculture","Agro-processing"])
tourism_mask = (sectors_a == "Tourism")

monthly_payment = np.array([
    amounts[i]*rates[i]/(1-(1+rates[i])**(-int(terms[i]))) if rates[i]>0 else amounts[i]/int(terms[i])
    for i in range(n)
])

dpd_state   = np.zeros(n, dtype=int)
dpd_days    = np.zeros(n, dtype=int)
outstanding = amounts.astype(float).copy()
status      = np.ones(n, dtype=int)       # 1=active, 0=closed/written-off, -1=in-workout
restructure = np.zeros(n, dtype=int)
debt_review = np.zeros(n, dtype=int)
months_active       = np.zeros(n, dtype=int)
workout_months_left = np.zeros(n, dtype=int)
has_been_defaulted  = np.zeros(n, dtype=int)
orig_month_ids = ((pd.to_datetime(df["origination_date"])-START_DATE).dt.days/30).round().astype(int).values

state_default_mult = np.array([0.22, 1.09, 2.18, 4.35])
WORKOUT_CURE_P  = 0.03
WORKOUT_MIN, WORKOUT_MAX = 6, 24

records = []
for m_idx in range(N_MONTHS):
    mrow = macro.iloc[m_idx]; month_id = mrow["month_id"]
    
    # Calculate Macro Multiplier (mm)
    mm = np.ones(n)
    mm *= np.exp(-0.15*(mrow["gdp_growth_yoy"]-1.2))
    mm *= np.exp( 0.08*(mrow["unemployment"]-32.5))
    mm[agri_mask]    *= np.exp(-0.05*(mrow["agri_commodity_index"]-100))
    mm[tourism_mask] *= np.exp(-0.06*(mrow["tourism_index"]-100))
    
    for i in range(n):
        if month_id < orig_month_ids[i]: continue
        if status[i] == 0: continue

        # ── WORKOUT BOOK ──────────────────────
        if status[i] == -1:
            workout_months_left[i] -= 1
            if np.random.rand() < WORKOUT_CURE_P:
                status[i] = 1
                if np.random.rand() < 0.85:
                    dpd_state[i] = 3; dpd_days[i] = np.random.randint(60, 90)
                else:
                    dpd_state[i] = 2; dpd_days[i] = np.random.randint(30, 60)
                records.append(dict(
                    loan_id=loan_ids[i], month_id=month_id, month=mrow["month"],
                    dpd_bucket=dpd_state[i], dpd_days=dpd_days[i],
                    outstanding_balance=round(outstanding[i], 2),
                    payment_due=round(monthly_payment[i], 2), payment_made=0.0,
                    state_before=4, state_after=dpd_state[i], stage=2,
                    restructure_flag=restructure[i], debt_review_flag=debt_review[i],
                    default_flag=0, status="Active (cured)"))
            elif workout_months_left[i] <= 0:
                status[i] = 0
                records.append(dict(
                    loan_id=loan_ids[i], month_id=month_id, month=mrow["month"],
                    dpd_bucket=4, dpd_days=max(dpd_days[i], 90),
                    outstanding_balance=round(outstanding[i], 2),
                    payment_due=0.0, payment_made=0.0,
                    state_before=4, state_after=5, stage=None,
                    restructure_flag=restructure[i], debt_review_flag=debt_review[i],
                    default_flag=0, status="Written-off"))
            else:
                records.append(dict(
                    loan_id=loan_ids[i], month_id=month_id, month=mrow["month"],
                    dpd_bucket=4, dpd_days=max(dpd_days[i], 90),
                    outstanding_balance=round(outstanding[i], 2),
                    payment_due=round(monthly_payment[i], 2), payment_made=0.0,
                    state_before=4, state_after=4, stage=3,
                    restructure_flag=restructure[i], debt_review_flag=debt_review[i],
                    default_flag=0, status="In-workout"))
            continue

        # ── ACTIVE LOAN ─────────────────────────────────
        months_active[i] += 1

        if months_active[i] > int(terms[i]):
            if dpd_state[i] == 0 or has_been_defaulted[i] == 1:
                status[i] = 0
                sb = dpd_state[i]
                records.append(dict(
                    loan_id=loan_ids[i], month_id=month_id, month=mrow["month"],
                    dpd_bucket=dpd_state[i], dpd_days=dpd_days[i],
                    outstanding_balance=0.0,
                    payment_due=round(monthly_payment[i], 2), payment_made=0.0,
                    state_before=sb, state_after=5, stage=None,
                    restructure_flag=restructure[i], debt_review_flag=debt_review[i],
                    default_flag=0, status="Closed"))
                continue
            else:
                sb = dpd_state[i]
                status[i] = -1
                dpd_state[i] = 4
                dpd_days[i] = max(dpd_days[i], 90)
                has_been_defaulted[i] = 1
                workout_months_left[i] = np.random.randint(6, 13)
                records.append(dict(
                    loan_id=loan_ids[i], month_id=month_id, month=mrow["month"],
                    dpd_bucket=4, dpd_days=dpd_days[i],
                    outstanding_balance=round(outstanding[i], 2),
                    payment_due=round(monthly_payment[i], 2), payment_made=0.0,
                    state_before=sb, state_after=4, stage=3,
                    restructure_flag=restructure[i], debt_review_flag=debt_review[i],
                    default_flag=1, status="In-workout"))
                continue

        sb = dpd_state[i]; h = true_pd_m[i]*mm[i]; u = np.random.rand()

        if u < h * state_default_mult[sb]:
            dpd_state[i]=4; dpd_days[i]=max(dpd_days[i],90)
            status[i]=-1
            has_been_defaulted[i] = 1
            workout_months_left[i] = np.random.randint(WORKOUT_MIN, WORKOUT_MAX+1)
            default_flag = 1
        else:
            default_flag = 0
            u2 = np.random.rand()
            if sb == 0:
                if u2 < h*1.5:
                    dpd_state[i]=1; dpd_days[i]=np.random.randint(1,30)
                else: dpd_days[i]=0
            elif sb == 1:
                if   u2<0.35: dpd_state[i]=0; dpd_days[i]=0
                elif u2<0.75: dpd_days[i]=np.random.randint(1,30)
                else:         dpd_state[i]=2; dpd_days[i]=np.random.randint(30,60)
            elif sb == 2:
                if   u2<0.05: dpd_state[i]=0; dpd_days[i]=0
                elif u2<0.30: dpd_state[i]=1; dpd_days[i]=np.random.randint(1,30)
                elif u2<0.70: dpd_days[i]=np.random.randint(30,60)
                else:         dpd_state[i]=3; dpd_days[i]=np.random.randint(60,90)
            elif sb == 3:
                if   u2<0.02: dpd_state[i]=0; dpd_days[i]=0
                elif u2<0.05: dpd_state[i]=1; dpd_days[i]=np.random.randint(1,30)
                elif u2<0.20: dpd_state[i]=2; dpd_days[i]=np.random.randint(30,60)
                elif u2<0.70: dpd_days[i]=np.random.randint(60,90)
                else:
                    dpd_state[i]=4; dpd_days[i]=90; default_flag=1
                    status[i]=-1
                    has_been_defaulted[i]=1
                    workout_months_left[i]=np.random.randint(WORKOUT_MIN, WORKOUT_MAX+1)

            if dpd_state[i] in (2,3) and restructure[i]==0 and np.random.rand()<0.15: restructure[i]=1
            if dpd_state[i]>=2 and debt_review[i]==0 and np.random.rand()<0.05:       debt_review[i]=1

            if   dpd_state[i]==0: payment = monthly_payment[i]
            elif dpd_state[i]==1: payment = monthly_payment[i]*np.random.uniform(0,0.5)
            elif dpd_state[i]==2: payment = monthly_payment[i]*np.random.uniform(0,0.2)
            else:                 payment = 0.0
            interest  = outstanding[i]*rates[i]
            principal = max(0, payment-interest)
            outstanding[i] = max(0, outstanding[i]-principal)

        if status[i] == -1:                                        stage = 3
        elif dpd_state[i]>=2 or restructure[i] or debt_review[i]:  stage = 2
        else:                                                      stage = 1

        records.append(dict(
            loan_id=loan_ids[i], month_id=month_id, month=mrow["month"],
            dpd_bucket=dpd_state[i], dpd_days=dpd_days[i],
            outstanding_balance=round(outstanding[i], 2),
            payment_due=round(monthly_payment[i], 2), payment_made=round(payment, 2),
            state_before=sb, state_after=dpd_state[i], stage=stage,
            restructure_flag=restructure[i], debt_review_flag=debt_review[i],
            default_flag=default_flag,
            status=("Active" if status[i]==1 else "In-workout")))

perf = pd.DataFrame(records)
perf["year"] = perf["month"].dt.year
perf.to_parquet(RAW_PQ/"loan_monthly_performance.parquet", index=False)
n_unique_defaults = perf.loc[perf["default_flag"]==1, "loan_id"].nunique()
n_events = int(perf["default_flag"].sum())
print(f"  {len(perf):,} loan-months | "
      f"{n_unique_defaults:,} defaulted loans | "
      f"{n_unique_defaults/n:.2%} realized (events: {n_events:,})")

# ── 4. PAYMENT SCHEDULE ────────────────────────────────────
print("[4/6] Payment schedule...")
sched = []
for i in range(n):
    P, r, t, mp = amounts[i], rates[i], int(terms[i]), monthly_payment[i]
    bal = P
    for m in range(1, t+1):
        interest = bal*r
        principal = bal if m==t else mp-interest
        mp_actual = principal+interest
        bal = max(0, bal-principal)
        sched.append(dict(loan_id=loan_ids[i], installment_num=m,
            scheduled_due=round(mp_actual,2), scheduled_principal=round(principal,2),
            scheduled_interest=round(interest,2), scheduled_balance=round(bal,2)))
sched_df = pd.DataFrame(sched)
sched_df.to_parquet(RAW_PQ/"payment_schedule.parquet", index=False)
print(f"  {len(sched_df):,} installments")

# ── 5. DEFAULT EVENTS + RECOVERY ────────────────────────────
print("[5/6] Default events & recovery...")
def_ev = (perf[perf["default_flag"]==1]
    .sort_values("month_id")
    .groupby("loan_id", as_index=False)
    .agg(default_month=("month_id","first"), ead_at_default=("outstanding_balance","first")))
def_ev = def_ev.merge(
    df[["loan_id","collateral_type","sector","region","dfi_funded","loan_amount_initial"]],
    on="loan_id")
def_ev["recovery_rate_true"] = (def_ev["collateral_type"].map(collateral_recovery)
                                + np.random.normal(0,0.10,len(def_ev))).clip(0,1)
def_ev["recovery_amount"] = (def_ev["ead_at_default"]*def_ev["recovery_rate_true"]).round(2)
def_ev["time_to_recovery_months"] = np.random.randint(3,25,len(def_ev))
def_ev["lgd_true"] = (1-def_ev["recovery_rate_true"]).round(4)
def_ev.to_parquet(RAW_PQ/"default_events_recovery.parquet", index=False)
print(f"  {len(def_ev):,} unique defaulted loans | mean LGD {def_ev['lgd_true'].mean():.2%}")

# ── 6. SUMMARIES + EXCEL ───────────────────────────────────
print("[6/6] Summaries & Excel export...")
trans = perf[perf["state_before"].isin([0,1,2,3,4]) & (perf["state_after"]>=0)]
tmat_p = (pd.crosstab(trans["state_before"], trans["state_after"], normalize="index")
          .reindex(index=[0,1,2,3,4], columns=[0,1,2,3,4,5], fill_value=0.0))
tmat_n = (pd.crosstab(trans["state_before"], trans["state_after"])
          .reindex(index=[0,1,2,3,4], columns=[0,1,2,3,4,5], fill_value=0))

on_book    = perf[perf["stage"].notna()]
stage_dist = on_book.groupby(["month","stage"]).size().unstack(fill_value=0)
stage_pct  = stage_dist.div(stage_dist.sum(axis=1), axis=0)*100

kpis = perf.groupby("month").agg(
    n_active=("loan_id", lambda x: (perf.loc[x.index,"status"]=="Active").sum()),
    n_defaults=("default_flag","sum"),
    n_workout=("status", lambda x: (x=="In-workout").sum()),
    total_outstanding=("outstanding_balance","sum"),
    avg_dpd=("dpd_days","mean"),
    n_restructured=("restructure_flag","sum"),
    n_debt_review=("debt_review_flag","sum")).reset_index()
npl = perf[perf["dpd_bucket"]>=3].groupby("month").size()
kpis["npl_count"] = kpis["month"].map(npl).fillna(0).astype(int)
kpis["npl_ratio_pct"] = (kpis["npl_count"]/kpis["n_active"].replace(0,np.nan)*100).round(2)

df.to_excel(RAW_XL/"loan_master.xlsx", index=False, sheet_name="Loan Master")
for yr in sorted(perf["year"].unique()):
    perf[perf["year"]==yr].sample(frac=0.10, random_state=SEED).to_excel(
        RAW_XL/f"performance_{yr}_sample10pct.xlsx", index=False, sheet_name=f"Perf {yr}")
with pd.ExcelWriter(RAW_XL/"ifrs9_summaries.xlsx", engine="openpyxl") as w:
    stage_pct.round(2).to_excel(w, sheet_name="Stage %")
    stage_dist.to_excel(w, sheet_name="Stage Counts")
    kpis.to_excel(w, sheet_name="Monthly KPIs", index=False)
    tmat_p.round(4).to_excel(w, sheet_name="Transition Matrix P")
    tmat_n.to_excel(w, sheet_name="Transition Counts")
    macro.to_excel(w, sheet_name="Macro Scenarios", index=False)
    def_ev.describe().round(3).to_excel(w, sheet_name="Recovery Stats")
def_ev.sample(min(500,len(def_ev)), random_state=SEED).to_excel(RAW_XL/"recovery_sample.xlsx", index=False)
sched_df[sched_df["loan_id"].isin(np.random.choice(loan_ids, n//100, replace=False))].to_excel(
    RAW_XL/"payment_schedule_sample.xlsx", index=False)
macro.to_excel(RAW_XL/"macro_scenarios.xlsx", index=False)

meta = {"generated_at": datetime.now().isoformat(), "seed": SEED,
    "version": "v5 - 60 months, 3-year vintage, systemic recession shock",
    "n_loans": N_LOANS, "n_months": N_MONTHS,
    "start_date": str(START_DATE.date()), "end_date": str(months[-1].date()),
    "mean_true_annual_pd": float(df["true_annual_pd"].mean()),
    "total_exposure": float(df["loan_amount_initial"].sum()),
    "realized_default_rate": float(len(def_ev)/N_LOANS),
    "default_events_count": n_events,
    "calibration_target": 0.03,
    "mean_lgd": float(def_ev["lgd_true"].mean()),
    "performance_rows": int(len(perf)),
    "notes": "Systemic recession in 2025 H1 (Months 37-43). Unemployment +4.5%, Repo +2.5%, GDP -2.0%"}
with open(META/"dataset_metadata.json","w") as f: json.dump(meta, f, indent=2, default=str)

print("\n" + "="*70); print("COMPLETE — v5 (Systemic Recession)"); print("="*70)
print(f"Portfolio:  {N_LOANS:,} loans | mean PD {df['true_annual_pd'].mean():.2%}")
print(f"Realized:   {len(def_ev)/N_LOANS:.2%} default rate")
print(f"Mean LGD:   {def_ev['lgd_true'].mean():.2%}")
print(f"Events:     {n_events:,} (unique loans: {len(def_ev):,})")
print(f"\nStage distribution (last month, on-book):")
for s, pct in stage_pct.iloc[-1].items():
    if pd.notna(s): print(f"  Stage {int(s)}: {pct:5.2f}%")
print(f"\nTransition matrix:")
print(tmat_p.round(3).to_string())