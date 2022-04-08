from dataclasses import dataclass
from enum import Enum

################################################################################
#  NLLB Job - Required Helper Enums and Class Definitions
# This file will contain Abstract NLLBJob implementation in future PR
################################################################################


class RegistryStatuses(Enum):
    """
    The job registry displays 5 essential high-level status of a job, shown below.
    """

    COMPLETED = "Completed"
    FAILED = "Failed"
    PENDING = "Pending"
    RUNNING = "Running"
    UNKNOWN = "Unknown"


class JobType(Enum):
    """
    Two types of jobs exist: single jobs and array jobs
    """

    SINGLE = "Single"
    ARRAY = "Array"


@dataclass
class ArrayJobInfo:
    """
    ArrayJobInfo stores additional info for jobs of array type (not single job). Stored info is:
        parent_job_id: (the id of the combined array of jobs; In slurm this is the job ID before the underscore _ (e.g for 1234567_00, it would be 1234567))
        array_index: (the index of this job within larger job array)
    """

    def __init__(self, parent_job_id: str, array_index: int):
        self.parent_job_id = parent_job_id
        self.array_index = array_index
        assert array_index >= 0, f"array_index: {array_index} must be >= 0"
        # could also be helpful to add total jobs in array in future