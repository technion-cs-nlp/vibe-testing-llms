"""Logic for selecting benchmark samples based on a user profile."""

from asyncio.log import logger
import random
import sys
from typing import List
from .data_utils import UserProfile, BenchmarkSample


class SampleSelector:
    """
    Selects relevant benchmark samples that align with a user's profile.
    """

    def __init__(self, total_samples: int):
        """Initializes the SampleSelector."""
        self.total_samples = total_samples
        print("Initialized SampleSelector with total samples:", total_samples)

    def select(
        self, profile: UserProfile, benchmarks: List[BenchmarkSample]
    ) -> List[BenchmarkSample]:
        """
        Filters and selects samples from a list of benchmark items.

        The selection logic will be based on criteria in the UserProfile, such
        as task complexity, domain, etc.

        Args:
            profile (UserProfile): The structured user profile.
            benchmarks (List[BenchmarkSample]): A list of all candidate samples.

        Returns:
            List[BenchmarkSample]: A filtered list of samples relevant to the user.
        """
        print(f"Selecting samples for user {profile.user_id}...")
        # Sample from benchmarks while keeping a balance between different benchmarks
        benchmark_counts = {}
        for benchmark in benchmarks:
            benchmark_counts[benchmark.source_benchmark] = (
                benchmark_counts.get(benchmark.source_benchmark, 0) + 1
            )
        for benchmark, count in benchmark_counts.items():
            print(f"Benchmark {benchmark} has {count} samples.")
        selected_samples = []
        for benchmark, count in benchmark_counts.items():
            if self.total_samples <= len(selected_samples):
                raise ValueError(
                    f"Total samples {self.total_samples} is less than the number of selected samples {len(selected_samples)}"
                )
            # for mbpp_plus, make sure non of these task_ids are selected, resample new ones (94, 721, 722, 723, 754)
            if benchmark == "mbpp_plus":
                for sample in benchmarks:
                    if sample.sample_id in [94, 721, 722, 723, 754]:
                        logger.error(
                            f"Task_id {sample.sample_id} is problematic for mbpp_plus"
                        )
                        benchmarks = [
                            sample
                            for sample in benchmarks
                            if sample.sample_id not in [94, 721, 722, 723, 754]
                        ]
            selected_samples.extend(
                random.sample(benchmarks, self.total_samples // len(benchmark_counts))
            )

        print(f"Selected {len(selected_samples)} samples.")
        return selected_samples
