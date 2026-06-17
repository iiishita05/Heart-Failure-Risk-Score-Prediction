from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import dice_ml
from dice_ml import Dice

app = Flask(__name__)
CORS(app)

# ======================
# LOAD DATA + MODELS
# ======================
df = pd.read_csv("cleaned_train.csv")

TARGET = "target"
X = df.drop(columns=[TARGET])

rf = joblib.load("rf_model.pkl")
catboost_model = joblib.load("catboost_model.pkl")


def predict_proba(model, df):
    return model.predict_proba(df)[:, 1]


# ======================
# DICE SETUP (GENETIC)
# ======================
data_dice = dice_ml.Data(
    dataframe=df,
    continuous_features=list(X.columns),
    outcome_name=TARGET
)

model_dice = dice_ml.Model(model=rf, backend="sklearn")
dice = Dice(data_dice, model_dice, method="genetic")


# ======================
# SETTINGS
# ======================
permitted_range = {
    'trestbps': [60, 220],
    'oldpeak': [0, 7],
    'thalch': [50, 210],
    'chol': [80, 500],
    'ca': [0, 4]
}

# important features only (clean output)
important_features = ['oldpeak', 'thalch', 'ca', 'chol', 'trestbps']

# clamp unrealistic values
MAX_CHANGE = {
    "thalch": 30,
    "chol": 50,
    "trestbps": 30,
    "oldpeak": 2,
    "ca": 2
}

# ======================
# MAIN API
# ======================
@app.route("/counterfactual", methods=["POST"])
def generate_cf():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data"}), 400

        if data.get("risk") != "High":
            return jsonify({"counterfactuals": []})

        # -----------------
        # CLEAN INPUT
        # -----------------
        clean_data = {k: v for k, v in data.items() if k in X.columns}
        query = pd.DataFrame([clean_data])

        if "thalach" in query.columns:
            query.rename(columns={"thalach": "thalch"}, inplace=True)

        for col in X.columns:
            if col not in query.columns:
                query[col] = 0

        query = query[X.columns]

        # -----------------
        # GENERATE CFs
        # -----------------
        cf = dice.generate_counterfactuals(
            query,
            total_CFs=10,              # 🔥 ensures diversity
            desired_class=0,
            features_to_vary="all",
            permitted_range=permitted_range,
            diversity_weight=1.0
        )

        cf_list = cf.cf_examples_list[0].final_cfs_df

        if cf_list is None or cf_list.empty:
            return jsonify({"counterfactuals": []})

        # -----------------
        # RISK
        # -----------------
        original_prob = float(predict_proba(catboost_model, query)[0])

        results = []

        for i in range(len(cf_list)):
            cf_instance = cf_list.iloc[[i]]
            cf_instance = cf_instance[X.columns]

            cf_prob = float(predict_proba(catboost_model, cf_instance)[0])

            risk_reduction = 0 if original_prob == 0 else (
                (original_prob - cf_prob) / original_prob
            ) * 100

            changes = []

            for col in important_features:
                original_val = float(query.iloc[0][col])
                new_val = float(cf_instance.iloc[0][col])
                diff = new_val - original_val

                if abs(diff) > 0.1:
                    max_allowed = MAX_CHANGE.get(col, 50)

                    if abs(diff) > max_allowed:
                        diff = max_allowed if diff > 0 else -max_allowed

                    changes.append({
                        "feature": col,
                        "change": "increase" if diff > 0 else "decrease",
                        "amount": float(round(abs(diff), 2))
                    })

            if not changes:
                continue

            explanation = (
                f"Risk {original_prob:.2f} → {cf_prob:.2f} "
                f"({risk_reduction:.1f}% ↓)"
            )

            results.append({
                "recommendation": int(i + 1),
                "original_risk": float(round(original_prob, 2)),
                "new_risk": float(round(cf_prob, 2)),
                "risk_reduction": float(round(risk_reduction, 2)),
                "changes": changes,
                "explanation": explanation
            })

        # -----------------
        # REMOVE DUPLICATES
        # -----------------
        unique_results = []
        seen = set()

        for r in results:
            key = (
                round(r["new_risk"], 4),
                tuple(sorted(
                    (c["feature"], c["change"], round(c["amount"], 2))
                    for c in r["changes"]
                ))
            )

            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        # -----------------
        # SORT & TAKE TOP 3
        # -----------------
        unique_results.sort(key=lambda x: x["risk_reduction"], reverse=True)
        unique_results = unique_results[:3]

        return jsonify({
            "total_recommendations": len(unique_results),
            "counterfactuals": unique_results
        })

    except Exception as e:
        import traceback
        print("\n🔥 ERROR:\n")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ======================
# RUN SERVER
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    