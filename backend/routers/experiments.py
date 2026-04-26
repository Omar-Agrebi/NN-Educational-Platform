from fastapi import APIRouter, HTTPException
from models.schemas import ExperimentRequest, ModelConfig, TrainConfig, TrainRequest
from models.responses import ExperimentResult
from progression.experiment_configs import list_experiments, get_experiment
from routers.training import _run_training

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("")
def get_experiments():
    return list_experiments()


@router.post("/run")
def run_experiment(req: ExperimentRequest):
    try:
        exp = get_experiment(req.experiment_id)
        train_cfg = TrainConfig(**exp["train_config"])

        def make_req(model_dict, label):
            mc = ModelConfig(**model_dict)
            return TrainRequest(
                session_id=req.session_id + f"_{label}",
                dataset=exp["dataset"],
                nn_config=mc,
                train_config=train_cfg,
            )

        result_a = _run_training(make_req(exp["model_a"], "a"))
        result_b = _run_training(make_req(exp["model_b"], "b"))

        acc_a = result_a.final_metrics.get("accuracy", 0)
        acc_b = result_b.final_metrics.get("accuracy", 0)
        f1_a = result_a.final_metrics.get("f1_score", 0)
        f1_b = result_b.final_metrics.get("f1_score", 0)
        winner = "model_b" if (acc_b + f1_b) > (acc_a + f1_a) else "model_a"

        return ExperimentResult(
            experiment_id=req.experiment_id,
            model_a_result=result_a,
            model_b_result=result_b,
            lesson=exp["lesson"],
            winner=winner,
            metric_comparison={
                "accuracy": {"model_a": acc_a, "model_b": acc_b},
                "f1_score": {"model_a": f1_a, "model_b": f1_b},
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
