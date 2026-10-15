"""Build small notebook walkthroughs from the actual exported experiment tables."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(project, title, intro, sections, figure, limits):
    cells = []
    namespace = {"__builtins__": __builtins__}

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

    def code(source):
        tree = ast.parse(source)
        output = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(prefix, "<walkthrough>", "exec"), namespace)
            output = eval(compile(ast.Expression(tree.body[-1].value), "<walkthrough>", "eval"), namespace)
        else:
            exec(compile(tree, "<walkthrough>", "exec"), namespace)
        count = sum(c["cell_type"] == "code" for c in cells) + 1
        outputs = []
        if output is not None:
            data = {"text/plain": str(output)}
            if hasattr(output, "to_html"):
                data["text/html"] = output.to_html()
            outputs.append({"output_type": "execute_result", "execution_count": count,
                            "metadata": {}, "data": data})
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": count,
                      "source": source, "outputs": outputs})

    markdown(f"# {title}\n\n{intro}\n\nThis walkthrough reads the exported results from the reproduced Python experiment. It does not train models in its cells. To regenerate the results, run the command below from the repository root, then rerun these cells.\n\n```bash\npython -m decision_analytics.{ 'bank' if project=='bank-marketing' else 'donor' }\n```\n\nSee [case README](README.md) for the full protocol and interpretation.")
    code("from pathlib import Path\nimport pandas as pd\n\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]\n            if (p / 'pyproject.toml').exists() and (p / 'src/decision_analytics').exists())\nCASE = ROOT / 'projects' / " + repr(project))
    for heading, explanation, expression in sections:
        markdown(f"## {heading}\n\n{explanation}")
        code(expression)
    markdown(f"## Decision visualization\n\n![Case results]({figure})")
    markdown("## Limits\n\n" + limits)
    for i, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{project}:{i}".encode()).hexdigest()[:12]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                             "language_info": {"name": "python", "version": "3.12"}}}
    target = ROOT / "projects" / project / "analysis.ipynb"
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print("Built and executed table cells:", target.relative_to(ROOT))


if __name__ == "__main__":
    build("bank-marketing", "Bank marketing: prediction, cost, and capacity",
          "Select clients under a forecast call-time budget, then compare the plan with actual historical call duration.",
          [("Subscription models", "The classifier is selected by validation Brier score. The final row evaluates the selected model on the historical test rows.", "pd.read_csv(CASE / 'results/classification.csv').round(5)"),
           ("Duration baselines", "RMSE favors the training mean. MAE favors the log forest, illustrating why the error metric should match the decision objective.", "pd.read_csv(CASE / 'results/duration.csv').round(4)"),
           ("Forecast budgets and actual budgets", "Keep the original duration forest as an experimental policy. Buffer values were selected using validation rows, before applying them to test rows.", "policies = pd.read_csv(CASE / 'results/policies_test.csv')\nfocus = policies[(policies.scenario == 'baseline') & (policies.duration_model == 'Log-random forest') & policies.policy.isin(['Probability ranking', 'Duration-aware ranking'])]\nfocus[['policy', 'variant', 'buffer', 'contacts', 'predicted_minutes', 'actual_minutes', 'actual_overrun_pct']].round(3)")],
          "figures/calibration_and_capacity.svg",
          "These are retrospective random-split results on historical contacted customers. Profit uses assumptions, not causal lift. Predicted feasibility does not guarantee actual feasibility, and the original test rows had already been inspected in coursework.")
    build("donor-routing", "Donor targeting: proxy labels, subgroup transfer, and routing",
          "Combine model scores with distance constraints in a fictional nonprofit outreach scenario.",
          [("Training-only model selection", "Each stage selects its estimator on an internal holdout from the source training data, then refits the selected estimator.", "pd.read_csv(CASE / 'results/model_selection_validation.csv').round(4)"),
           ("Known and cold prospects", "The overall winner need not transfer best to cold prospects. The hybrid rule comes from the original course analysis, so this is a retrospective comparison.", "pd.read_csv(CASE / 'results/future_donation_test.csv').round(4)"),
           ("Route baselines", "All routes reserve enough distance to return to the depot. Income scores are not calibrated donation probabilities; score-based projections and historical simulated proceeds are kept separate.", "pd.read_csv(CASE / 'results/routes_test.csv').round(3)"),
           ("Travel-budget sensitivity", "Keep the score threshold fixed while changing the distance allowance.", "pd.read_csv(CASE / 'results/budget_sensitivity.csv').round(3)")],
          "figures/subgroups_and_route.svg",
          "The 200-person sample and scenario outcomes do not establish real fundraising impact. Haversine distances ignore road networks and visit time. The inherited hybrid rule needs a separate future population and response-probability calibration.")
