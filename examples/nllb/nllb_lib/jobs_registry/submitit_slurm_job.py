import logging
from pathlib import Path

import submitit
from submitit.slurm.slurm import read_job_id

from examples.nllb.nllb_lib.jobs_registry.nllb_job import (
    ArrayJobInfo,
    JobType,
    RegistryStatuses,
)
from examples.nllb.nllb_lib.nllb_module import NLLBModule

logger = logging.getLogger("submitit_slurm_job registry")


################################################################################
#  Submitit Launcher - SLURM Job Statuses Dictionary
################################################################################
# Submitit/Slurm jobs statuses can take a total of ~25 different values.
# These are documented here: https://slurm.schedmd.com/squeue.html under the "Job State Codes" Heading
# The job registry however displays only 5 high-level essential statuses. This can be seen in the RegistryStatuses Enum in examples/nllb/nllb_lib/jobs_registry/nllb_job.py
# Below is a dictionary to map each of the submitit job statuses 24 SLURM job + a few local Job statuses to the 5 accepted registry statuses

submitit_state_to_registry_state_dict = {
    # ############## RegistryStatuses.COMPLETED below ###########################
    "COMPLETED": RegistryStatuses.COMPLETED.value,
    "FINISHED": RegistryStatuses.COMPLETED.value,
    # ############## RegistryStatuses.FAILED below ##############################
    "FAILED": RegistryStatuses.FAILED.value,
    "CANCELLED": RegistryStatuses.FAILED.value,
    "STOPPED": RegistryStatuses.FAILED.value,
    "SUSPENDED": RegistryStatuses.FAILED.value,
    "TIMEOUT": RegistryStatuses.FAILED.value,
    "NODE_FAIL": RegistryStatuses.FAILED.value,
    "OUT_OF_MEMORY": RegistryStatuses.FAILED.value,
    "DEADLINE": RegistryStatuses.FAILED.value,
    "BOOT_FAIL": RegistryStatuses.FAILED.value,
    "RESV_DEL_HOLD": RegistryStatuses.FAILED.value,
    "REVOKED": RegistryStatuses.FAILED.value,
    "SIGNALING": RegistryStatuses.FAILED.value,  # I'm quite sure signalling means being cancelled
    "INTERRUPTED": RegistryStatuses.FAILED.value,  # from local submitit job statuses
    # ############## RegistryStatuses.PENDING below ##############################
    "PENDING": RegistryStatuses.PENDING.value,
    "REQUEUE_FED": RegistryStatuses.PENDING.value,
    "REQUEUE_HOLD": RegistryStatuses.PENDING.value,
    "REQUEUED": RegistryStatuses.PENDING.value,
    "RESIZING": RegistryStatuses.PENDING.value,
    "READY": RegistryStatuses.PENDING.value,  # from local submitit job statuses
    # ############## RegistryStatuses.RUNNING below ##############################
    "RUNNING": RegistryStatuses.RUNNING.value,
    "COMPLETING": RegistryStatuses.RUNNING.value,
    "CONFIGURING": RegistryStatuses.RUNNING.value,
    # ############## RegistryStatuses.UNKNOWN below ##############################
    "UNKNOWN": RegistryStatuses.UNKNOWN.value,
    "STAGE_OUT": RegistryStatuses.UNKNOWN.value,
    # STAGE_OUT status: Occurs once the job has completed or been cancelled, but Slurm has not released resources for the job yet. Source: https://slurm.schedmd.com/burst_buffer.html
    # Hence, setting it as unknown as this state doesn't specifify completion or cancellation.
    "SPECIAL_EXIT": RegistryStatuses.UNKNOWN.value,
    # SPECIAL_EXIT status: This occurs when a job exits with a specific pre-defined reason (e.g a specific error case).
    # This is useful when users want to automatically requeue and flag a job that exits with a specific error case. Source: https://slurm.schedmd.com/faq.html
    # Hence, setting as unknown as the status may change depending on automatic re-queue; without this requeue, job should be FAILED.
    "PREEMPTED": RegistryStatuses.UNKNOWN.value,
    # PREEMPTED Status: Different jobs react to preemption differently - some may automatically requeue and other's may not
    # (E.g if checkpoint method is implemented or not). Hence, setting as Unknown for now; without automatic re-queue, job should be FAILED
}


def convert_slurm_status_into_registry_job_status(
    slurm_status: str, job_id: str
) -> RegistryStatuses:
    """
    This function maps slurm's 24 different values to return 5 essential statuses understood by the registry,
    And logs a warning for unrecognized statuses.
    """
    try:
        job_status = submitit_state_to_registry_state_dict[slurm_status]
        return job_status
    except KeyError:  # Entering this except block means slurm_status doesn't exist in submitit_state_to_registry_state_dict
        logger.warning(
            f"Job with id: {job_id} has unrecognized slurm status: {slurm_status}. Please inspect and if suitable, add this status to the slurm_state_to_registry_state_map converter."
        )
        return RegistryStatuses.UNKNOWN.value


################################################################################
#  Submitit Launcher Job Class
################################################################################
class SubmititJob:
    """
    Job class for the submitit launcher.
    """

    def __init__(
        self,
        submitit_job_object: submitit.Job,
        job_index_in_registry: int,
        module: NLLBModule,
    ):
        assert isinstance(
            submitit_job_object, submitit.Job
        ), f"submitit_job_object must have type submitit.Job but was provided with type {type(submitit_job_object)}"

        # _launcher_specific_job_object is the underlying job object specific to each launcher; in this case it's the submitit_job_object
        self._launcher_specific_job_object = submitit_job_object
        self._job_id = self._launcher_specific_job_object.job_id
        self._index_in_registry = job_index_in_registry
        self._module = module
        self._std_out_file = self._launcher_specific_job_object.paths.stdout
        self._std_err_file = self._launcher_specific_job_object.paths.stderr

        # The read_job_id function reads a job id and returns a tuple with format:
        # [(parent_job_id, array_index] where array_index is only for array jobs; Hence, tuple length is 1 if job is single, and 2 if its an array
        read_job_id_output = read_job_id(self._job_id)[0]
        read_job_id_output_length = len(read_job_id_output)
        if read_job_id_output_length == 1:  # single job
            self._job_type = JobType.SINGLE
            self._array_job_info = None

        elif read_job_id_output_length == 2:  # array job
            self._job_type = JobType.ARRAY
            # Since it's an array job, set ArrayJobInfo to store parent_job_id and job_index_in_array
            parent_job_id = read_job_id_output[0]
            job_index_in_array = int(read_job_id_output[1])
            self._array_job_info = ArrayJobInfo(parent_job_id, job_index_in_array)

        else:
            raise ValueError(
                f"Could not identify job type for job with id: {self._job_id}"
            )

    # Getters:
    @property
    def launcher_specific_job_object(self) -> submitit.Job:
        return self._launcher_specific_job_object

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def index_in_registry(self) -> int:
        return self._index_in_registry

    @property
    def module(self) -> NLLBModule:
        return self._module

    @property
    def std_out_file(self) -> Path:
        return self._std_out_file

    @property
    def std_err_file(self) -> Path:
        return self._std_err_file

    @property
    def job_type(self) -> JobType:
        """
        Returns either JobType.SINGLE or JobType.ARRAY
        """
        job_type = self._job_type
        assert (
            job_type == JobType.SINGLE or job_type == JobType.ARRAY
        ), f"Error: Only two types of jobs accepted: {JobType.SINGLE.value} or {JobType.ARRAY.value}, but this job has type {self._job_type.value}. Check job constructor to ensure proper job instantiation."
        return job_type

    @property
    def array_job_info(self) -> ArrayJobInfo:
        if self._job_type == JobType.SINGLE:
            logger.info(
                f"array_job_info method called on Job of type {JobType.SINGLE.value}. This will necessarily return None"
            )
        return self._array_job_info

    # Methods
    def kill_job(self):
        self._launcher_specific_job_object.cancel()

    def get_status(self) -> RegistryStatuses:
        return convert_slurm_status_into_registry_job_status(
            self._launcher_specific_job_object.state, self.job_id
        )

    def get_job_info_log(self) -> str:
        """
        This function returns a string that can be used to log job info in one line in an organized fashion.
        It logs fields like Job id, status, type, and array job info (if type is array).
        """
        list_of_log_info = [
            f"Job Info - Registry Index: {self._index_in_registry} [id: {self._job_id}",
            f"status: {self.get_status()}",
            f"module: {self._module}",
            f"type: {self._job_type.value}",
        ]

        if self._job_type == JobType.ARRAY:
            list_of_log_info.append(
                f"Extra Array Info: (Parent Job ID: {self._array_job_info.parent_job_id}, Array Index: {self._array_job_info.array_index})"
            )
        else:
            list_of_log_info.append(f"Extra Array Info: None")

        job_info_log = " | ".join(list_of_log_info)
        return job_info_log