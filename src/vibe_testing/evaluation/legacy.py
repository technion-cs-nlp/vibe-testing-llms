"""Logic for evaluating model outputs on vibe-testing tasks."""

from typing import List
from src.vibe_testing.data_utils import VibeTask, EvaluationResult
from src.vibe_testing.models.base import BaseModel


class Evaluator:
    """
    Evaluates a model's output on a given VibeTask.
    """

    def __init__(self, judge_model: BaseModel):
        """
        Initializes the Evaluator.

        Args:
            judge_model (BaseModel): An LLM used for qualitative, "soft" metric
                                     evaluations (e.g., clarity, tone).
        """
        self._judge_model = judge_model
        print("Initialized Evaluator.")

    def evaluate(self,
                 response: str,
                 task: VibeTask) -> EvaluationResult:
        """
        Scores a model response along multiple dimensions defined in the task.

        This involves both quantitative checks (e.g., running code and checking
        correctness) and qualitative assessments using a judge LLM.

        Args:
            response (str): The raw output from the model being evaluated.
            task (VibeTask): The VibeTask containing the evaluation criteria.

        Returns:
            EvaluationResult: An object containing all the scores.
        """
        print(f"Evaluating response for task {task.task_id}...")

        # Placeholder logic
        quantitative_scores = {"pass_fail": 1.0} # Assume pass for skeleton
        qualitative_scores = {
            metric: 5.0 for metric in task.evaluation_metrics
        } # Max score for skeleton

        return EvaluationResult(
            task_id=task.task_id,
            model_name="unknown", # This should be set by the evaluation script
            raw_output=response,
            quantitative_scores=quantitative_scores,
            qualitative_scores=qualitative_scores,
        )

    def evaluate_all(self,
                     model_outputs: List[dict],
                     tasks: List[VibeTask]) -> List[EvaluationResult]:
        """
        Evaluates a batch of model outputs.

        Args:
            model_outputs (List[dict]): A list of dictionaries, each with
                                        'task_id' and 'output'.
            tasks (List[VibeTask]): The list of VibeTasks.

        Returns:
            List[EvaluationResult]: A list of evaluation results.
        """
        task_dict = {task.task_id: task for task in tasks}
        results = []
        for output in model_outputs:
            task_id = output['task_id']
            if task_id in task_dict:
                task = task_dict[task_id]
                result = self.evaluate(output['output'], task)
                results.append(result)
        return results
