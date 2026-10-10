"""Recompute solar inference on freshly keyed CPU checkpoint predictions."""
from .audit_solar_statistics import run
from .replay_solar_checkpoints import OUTPUT

if __name__ == "__main__":
    result = run(OUTPUT / "statistical_audit", keyed_predictions=OUTPUT)
    for row in result["comparisons"]:
        print(row["comparator"], "p", row["p_value"], "Holm", row["holm_p_value"])
