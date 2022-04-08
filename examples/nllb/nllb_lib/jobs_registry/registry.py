import logging
import typing as tp

from examples.nllb.nllb_lib.jobs_registry.submitit_slurm_job import (
    RegistryStatuses,
    SubmititJob,
)

logger = logging.getLogger("launcher_job_registry")

################################################################################
#  Registry Exceptions
################################################################################
class JobNotInRegistry(Exception):
    """
    Exception raised when querying for a job id in the registry that doesn't exist
    """

    def __init__(self, job_id: str, message="This Job doesn't exist in registry"):
        self.job_id = job_id
        self.message = f"Job ID is: {job_id}. " + message
        super().__init__(self.message)


################################################################################
#  Registry definition
################################################################################
class JobsRegistry:
    """
    The JobsRegistry is a field in the launcher class that stores all past and present scheduled jobs.
    Implemented as an dictionary (key = job_id, and value = NLLBJob type). The dictionary is ordered (stores jobs chronologically).
    """

    def __init__(self):
        self.registry: tp.OrderedDict[str, SubmititJob] = tp.OrderedDict()

    def get_total_job_count(self) -> int:
        return len(self.registry)

    def get_job(self, job_id: str) -> SubmititJob:
        try:
            job = self.registry[job_id]
            return job
        except KeyError as e:  # job_id doesn't exist in registry
            raise JobNotInRegistry(job_id)

    def register_job(self, nllb_job: SubmititJob):
        """
        Adds job to the registry. If job already exists, logs a warning.
        """
        job_id = nllb_job.job_id
        if job_id in self.registry:
            logger.warning(
                f"Tried to add job with id: {job_id} into registry, but it already exists previously"
            )
        else:
            self.registry[job_id] = nllb_job

    def kill_job(self, job_id: str):
        """
        Kills a job.
        We deal with this based on status. If a job status is not Completed nor Failed, this method kills the job
        """
        job: SubmititJob = self.get_job(job_id)
        job_status = job.get_status()
        if (
            job_status == RegistryStatuses.COMPLETED.value
            or job_status == RegistryStatuses.FAILED.value
        ):
            logger.warning(
                f"Tried to kill a job with id: {job_id}, but this job has already {job_status}"
            )

        elif job_status == RegistryStatuses.UNKNOWN.value:
            logger.warning(
                f"Be careful: About to kill job: {job_id} with Unknown status."
            )
            job.kill_job()

        elif (
            job_status == RegistryStatuses.RUNNING.value
            or job_status == RegistryStatuses.PENDING.value
        ):
            logger.info(f"Killing job: {job_id}")
            job.kill_job()

        else:
            # This case should not be possible as we've tried every possible status
            # However, if new statuses are added in the future (unlikely), and their if cases are not added to this method function then this
            # else case serves as safety - it will raise an exception to ensure we have defined behaviour
            raise NotImplementedError(
                f"Job with id: {job_id} and status: {self.get_job(job_id).get_status()} can't be killed due to unidentifiable status. Please implement kill_job for the status: {job_status}"
            )

    def get_registry_log_as_string(self) -> str:
        """
        This method returns a string that can be used for logging.
        The string logs out each job in the registry.
        """
        # list_of_all_job_logs is an array of strings - each string contains the logs for one job in the registry
        list_of_all_job_logs = ["Job Registry is:\n"]
        for job_id in self.registry:
            current_job: SubmititJob = self.get_job(job_id)
            list_of_all_job_logs.append(current_job.get_job_info_log())

        entire_registry_log: str = "\n".join(list_of_all_job_logs)
        return entire_registry_log