import statsmodels.formula.api as smf

from analysis.core import pre

df = pre("results/main_results/all.jsonl")
df["install_success"] = (df["status"] == "install").astype(int)

# Binary flag for having *any* reference
df["has_reference"] = (df["reference"] != "none").astype(int)

# Convert booleans cleanly
df["raw_buildsys"] = df["raw_buildsys"].astype(bool)

# Drop rows missing numeric scores
df = df.dropna(subset=["dependency_score", "variants_score"]).copy()

SUCCESS_STAGE = "concretize"
metric = "avg_attempts_to_success"

# your metric: attempts only on success rows (NaN elsewhere)
# df[metric] = df["attempt_num"].where(df["status"] == SUCCESS_STAGE).astype(float)

# # same RHS you had before, just OLS instead of logit
# mod = smf.ols(f"{metric} ~ C(model) + C(reference) + C(raw_buildsys)", data=df).fit()
# print(mod.summary())
# breakpoint()

model_has_ref = smf.logit("install_success ~ C(reference)", data=df).fit()
print(model_has_ref.summary())

# model = smf.logit(
#     "install_success ~ C(model) + C(reference) + C(raw_buildsys)",
#     data=df
# ).fit()

# print(model.summary())

# model = smf.logit(
#     "variants_score ~ C(model) + C(reference) + C(raw_buildsys)", data=df
# ).fit()

# print(model.summary())
#
baseline_label = "none"
model = smf.logit("install_success ~ has_reference", data=df).fit()

print(model.summary())
