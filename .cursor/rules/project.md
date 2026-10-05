# NBA Shot Quality Project Rules

- Keep the project simple and runnable.
- Do not overengineer.
- Implement one milestone at a time.
- Prefer readable Python over clever abstractions.
- Use pandas, scikit-learn, matplotlib/plotly, joblib, and Streamlit.
- Do not add new dependencies unless necessary.
- The target variable is SHOT_MADE or FGM depending on the dataset.
- The model predicts expected field goal percentage.
- Avoid using PLAYER_NAME or PLAYER_ID in the first model to prevent memorizing shooter skill.
- Use player/team columns for analysis, not for the initial pure shot-quality model.
- Evaluate probability predictions with log loss, Brier score, ROC-AUC, and calibration.
- Keep notebooks exploratory; put reusable logic in src/.