"""
GKE-based code execution client for GSPO training.

This module provides a Kubernetes-native code execution backend that:
- Creates ephemeral jobs for each code execution
- Enforces strict resource limits and security boundaries
- Supports parallel execution across the cluster
- Auto-scales based on workload

Usage:
    from nanochat.gke_executor import GKEExecutor

    executor = GKEExecutor(project_id="my-project", cluster="gspo-cluster")
    result = await executor.execute(code="int main() { return 0; }")
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    compile_time_ms: int
    run_time_ms: int
    timeout: bool
    job_name: Optional[str] = None


class GKEExecutor:
    """
    GKE-based code execution for GSPO training.

    Creates Kubernetes jobs to compile and run untrusted C++ code in
    isolated, resource-limited containers.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        cluster: Optional[str] = None,
        zone: Optional[str] = None,
        namespace: str = "gspo-sandbox",
        image: Optional[str] = None,
        timeout_seconds: int = 30,
        max_concurrent: int = 50,
    ):
        """
        Initialize GKE executor.

        Args:
            project_id: GCP project ID (default: from env GOOGLE_CLOUD_PROJECT)
            cluster: GKE cluster name (default: from env GKE_CLUSTER)
            zone: GKE zone (default: from env GKE_ZONE)
            namespace: Kubernetes namespace for jobs
            image: Container image for sandbox (default: gcr.io/{project}/gspo-sandbox:latest)
            timeout_seconds: Maximum execution time per job
            max_concurrent: Maximum concurrent jobs
        """
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.cluster = cluster or os.environ.get("GKE_CLUSTER", "gspo-cluster")
        self.zone = zone or os.environ.get("GKE_ZONE", "us-central1-a")
        self.namespace = namespace
        self.image = image or f"gcr.io/{self.project_id}/gspo-sandbox:latest"
        self.timeout_seconds = timeout_seconds
        self.max_concurrent = max_concurrent

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._k8s_client = None
        self._batch_api = None
        self._core_api = None

    async def _ensure_client(self):
        """Lazily initialize Kubernetes client."""
        if self._k8s_client is not None:
            return

        try:
            from kubernetes import client, config
            from kubernetes.client.rest import ApiException

            # Try in-cluster config first (when running in GKE)
            try:
                config.load_incluster_config()
                logger.info("Using in-cluster Kubernetes config")
            except config.ConfigException:
                # Fall back to kubeconfig
                config.load_kube_config()
                logger.info("Using kubeconfig")

            self._k8s_client = client
            self._batch_api = client.BatchV1Api()
            self._core_api = client.CoreV1Api()
            self._ApiException = ApiException

        except ImportError:
            raise RuntimeError(
                "kubernetes package not installed. "
                "Install with: pip install kubernetes"
            )

    def _generate_job_name(self, code: str) -> str:
        """Generate unique job name from code hash."""
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
        timestamp = int(time.time() * 1000) % 1000000
        return f"gspo-exec-{code_hash}-{timestamp}"

    def _create_configmap_spec(
        self,
        name: str,
        code: str,
        test_code: Optional[str] = None,
    ) -> dict:
        """Create ConfigMap spec for code content."""
        data = {"code.cpp": code}
        if test_code:
            data["test.cpp"] = test_code

        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": {
                    "app": "gspo-sandbox",
                    "component": "code",
                },
            },
            "data": data,
        }

    def _create_job_spec(
        self,
        job_name: str,
        configmap_name: str,
        compiler: str = "g++",
        cpp_standard: str = "20",
        optimization: str = "-O2",
    ) -> dict:
        """Create Job spec for code execution."""
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "gspo-sandbox",
                    "component": "executor",
                },
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 300,
                "activeDeadlineSeconds": self.timeout_seconds + 30,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "gspo-sandbox",
                            "component": "executor",
                        },
                    },
                    "spec": {
                        "serviceAccountName": "gspo-sandbox-runner",
                        "restartPolicy": "Never",
                        "securityContext": {
                            "runAsUser": 1000,
                            "runAsGroup": 1000,
                            "fsGroup": 1000,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "nodeSelector": {
                            "cloud.google.com/gke-spot": "true",
                        },
                        "tolerations": [
                            {
                                "key": "cloud.google.com/gke-spot",
                                "operator": "Equal",
                                "value": "true",
                                "effect": "NoSchedule",
                            },
                        ],
                        "containers": [
                            {
                                "name": "sandbox",
                                "image": self.image,
                                "imagePullPolicy": "Always",
                                "resources": {
                                    "requests": {"memory": "256Mi", "cpu": "500m"},
                                    "limits": {"memory": "512Mi", "cpu": "1000m"},
                                },
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": False,
                                    "runAsNonRoot": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "env": [
                                    {"name": "COMPILER", "value": compiler},
                                    {"name": "CPP_STANDARD", "value": cpp_standard},
                                    {"name": "TIMEOUT_SECONDS", "value": str(self.timeout_seconds)},
                                    {"name": "OPTIMIZATION_LEVEL", "value": optimization},
                                    {
                                        "name": "CODE_CONTENT",
                                        "valueFrom": {
                                            "configMapKeyRef": {
                                                "name": configmap_name,
                                                "key": "code.cpp",
                                            },
                                        },
                                    },
                                    {
                                        "name": "TEST_CONTENT",
                                        "valueFrom": {
                                            "configMapKeyRef": {
                                                "name": configmap_name,
                                                "key": "test.cpp",
                                                "optional": True,
                                            },
                                        },
                                    },
                                ],
                                "volumeMounts": [
                                    {"name": "tmp", "mountPath": "/tmp"},
                                    {"name": "sandbox-work", "mountPath": "/sandbox"},
                                ],
                            },
                        ],
                        "volumes": [
                            {"name": "tmp", "emptyDir": {"sizeLimit": "100Mi"}},
                            {"name": "sandbox-work", "emptyDir": {"sizeLimit": "50Mi"}},
                        ],
                    },
                },
            },
        }

    async def execute(
        self,
        code: str,
        test_code: Optional[str] = None,
        compiler: str = "g++",
        cpp_standard: str = "20",
        optimization: str = "-O2",
    ) -> ExecutionResult:
        """
        Execute C++ code in GKE sandbox.

        Args:
            code: C++ source code to compile and run
            test_code: Optional test code (GTest) to run against the code
            compiler: Compiler to use (g++, clang++)
            cpp_standard: C++ standard (17, 20, 23)
            optimization: Optimization level (-O0, -O2, -O3)

        Returns:
            ExecutionResult with compilation and execution results
        """
        await self._ensure_client()

        async with self._semaphore:
            job_name = self._generate_job_name(code)
            configmap_name = f"gspo-code-{job_name.split('-')[-2]}-{job_name.split('-')[-1]}"

            try:
                # Create ConfigMap with code
                configmap_spec = self._create_configmap_spec(
                    configmap_name, code, test_code
                )
                await asyncio.to_thread(
                    self._core_api.create_namespaced_config_map,
                    namespace=self.namespace,
                    body=configmap_spec,
                )

                # Create Job
                job_spec = self._create_job_spec(
                    job_name, configmap_name, compiler, cpp_standard, optimization
                )
                await asyncio.to_thread(
                    self._batch_api.create_namespaced_job,
                    namespace=self.namespace,
                    body=job_spec,
                )

                # Wait for job completion
                result = await self._wait_for_job(job_name)
                result.job_name = job_name
                return result

            except self._ApiException as e:
                logger.error(f"Kubernetes API error: {e}")
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Kubernetes error: {e.reason}",
                    exit_code=-1,
                    compile_time_ms=0,
                    run_time_ms=0,
                    timeout=False,
                    job_name=job_name,
                )
            finally:
                # Cleanup (jobs have TTL, but cleanup immediately for cost)
                await self._cleanup(job_name, configmap_name)

    async def _wait_for_job(self, job_name: str) -> ExecutionResult:
        """Wait for job to complete and get results."""
        start_time = time.time()
        max_wait = self.timeout_seconds + 60

        while time.time() - start_time < max_wait:
            job = await asyncio.to_thread(
                self._batch_api.read_namespaced_job,
                name=job_name,
                namespace=self.namespace,
            )

            if job.status.succeeded:
                # Get logs
                logs = await self._get_pod_logs(job_name)
                return self._parse_result(logs)

            if job.status.failed:
                logs = await self._get_pod_logs(job_name)
                if logs:
                    return self._parse_result(logs)
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="Job failed",
                    exit_code=-1,
                    compile_time_ms=0,
                    run_time_ms=0,
                    timeout=False,
                )

            await asyncio.sleep(0.5)

        # Timeout waiting for job
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Job did not complete within {max_wait}s",
            exit_code=-1,
            compile_time_ms=0,
            run_time_ms=0,
            timeout=True,
        )

    async def _get_pod_logs(self, job_name: str) -> Optional[str]:
        """Get logs from job's pod."""
        try:
            pods = await asyncio.to_thread(
                self._core_api.list_namespaced_pod,
                namespace=self.namespace,
                label_selector=f"job-name={job_name}",
            )

            if not pods.items:
                return None

            pod_name = pods.items[0].metadata.name
            logs = await asyncio.to_thread(
                self._core_api.read_namespaced_pod_log,
                name=pod_name,
                namespace=self.namespace,
            )
            return logs

        except self._ApiException as e:
            logger.warning(f"Failed to get pod logs: {e}")
            return None

    def _parse_result(self, logs: Optional[str]) -> ExecutionResult:
        """Parse JSON result from container logs."""
        if not logs:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="No output from container",
                exit_code=-1,
                compile_time_ms=0,
                run_time_ms=0,
                timeout=False,
            )

        try:
            # Find JSON in logs (might have other output)
            for line in logs.strip().split('\n'):
                if line.startswith('{'):
                    data = json.loads(line)
                    return ExecutionResult(
                        success=data.get("success", False),
                        stdout=data.get("stdout", ""),
                        stderr=data.get("stderr", ""),
                        exit_code=data.get("exit_code", -1),
                        compile_time_ms=data.get("compile_time_ms", 0),
                        run_time_ms=data.get("run_time_ms", 0),
                        timeout=data.get("timeout", False),
                    )

            # No JSON found, treat entire output as stdout
            return ExecutionResult(
                success=False,
                stdout=logs,
                stderr="Could not parse container output",
                exit_code=-1,
                compile_time_ms=0,
                run_time_ms=0,
                timeout=False,
            )

        except json.JSONDecodeError as e:
            return ExecutionResult(
                success=False,
                stdout=logs,
                stderr=f"JSON parse error: {e}",
                exit_code=-1,
                compile_time_ms=0,
                run_time_ms=0,
                timeout=False,
            )

    async def _cleanup(self, job_name: str, configmap_name: str):
        """Clean up job and configmap."""
        try:
            # Delete job (propagation deletes pods)
            await asyncio.to_thread(
                self._batch_api.delete_namespaced_job,
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Background",
            )
        except self._ApiException:
            pass

        try:
            await asyncio.to_thread(
                self._core_api.delete_namespaced_config_map,
                name=configmap_name,
                namespace=self.namespace,
            )
        except self._ApiException:
            pass

    async def execute_batch(
        self,
        codes: list[str],
        test_codes: Optional[list[Optional[str]]] = None,
        **kwargs,
    ) -> list[ExecutionResult]:
        """
        Execute multiple code samples in parallel.

        Args:
            codes: List of C++ source codes
            test_codes: Optional list of test codes (same length as codes)
            **kwargs: Additional arguments passed to execute()

        Returns:
            List of ExecutionResults in same order as input
        """
        if test_codes is None:
            test_codes = [None] * len(codes)

        tasks = [
            self.execute(code, test_code, **kwargs)
            for code, test_code in zip(codes, test_codes)
        ]

        return await asyncio.gather(*tasks)


# Fallback to local execution for development/testing
class LocalExecutor:
    """
    Local code execution fallback for development.

    Uses subprocess to compile and run code locally.
    Less secure than GKE - use only for trusted code!
    """

    def __init__(
        self,
        timeout_seconds: int = 10,
        max_concurrent: int = 4,
    ):
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(
        self,
        code: str,
        test_code: Optional[str] = None,
        compiler: str = "g++",
        cpp_standard: str = "20",
        optimization: str = "-O2",
    ) -> ExecutionResult:
        """Execute code locally using subprocess."""
        import tempfile
        import subprocess

        async with self._semaphore:
            with tempfile.TemporaryDirectory() as tmpdir:
                code_file = os.path.join(tmpdir, "code.cpp")
                binary_file = os.path.join(tmpdir, "a.out")

                # Write code
                with open(code_file, "w") as f:
                    f.write(code)

                # Compile
                compile_start = time.time()
                compile_cmd = [
                    compiler,
                    f"-std=c++{cpp_standard}",
                    optimization,
                    "-Wall", "-Wextra",
                    code_file,
                    "-o", binary_file,
                ]

                try:
                    compile_result = await asyncio.to_thread(
                        subprocess.run,
                        compile_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    compile_time = int((time.time() - compile_start) * 1000)

                    if compile_result.returncode != 0:
                        return ExecutionResult(
                            success=False,
                            stdout="",
                            stderr=f"Compilation failed: {compile_result.stderr}",
                            exit_code=compile_result.returncode,
                            compile_time_ms=compile_time,
                            run_time_ms=0,
                            timeout=False,
                        )
                except subprocess.TimeoutExpired:
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr="Compilation timed out",
                        exit_code=-1,
                        compile_time_ms=30000,
                        run_time_ms=0,
                        timeout=True,
                    )

                # Run
                run_start = time.time()
                try:
                    run_result = await asyncio.to_thread(
                        subprocess.run,
                        [binary_file],
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        cwd=tmpdir,
                    )
                    run_time = int((time.time() - run_start) * 1000)

                    return ExecutionResult(
                        success=run_result.returncode == 0,
                        stdout=run_result.stdout[:1048576],  # 1MB limit
                        stderr=run_result.stderr[:1048576],
                        exit_code=run_result.returncode,
                        compile_time_ms=compile_time,
                        run_time_ms=run_time,
                        timeout=False,
                    )
                except subprocess.TimeoutExpired:
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=f"Execution timed out after {self.timeout_seconds}s",
                        exit_code=-1,
                        compile_time_ms=compile_time,
                        run_time_ms=self.timeout_seconds * 1000,
                        timeout=True,
                    )

    async def execute_batch(
        self,
        codes: list[str],
        test_codes: Optional[list[Optional[str]]] = None,
        **kwargs,
    ) -> list[ExecutionResult]:
        """Execute multiple code samples in parallel."""
        if test_codes is None:
            test_codes = [None] * len(codes)

        tasks = [
            self.execute(code, test_code, **kwargs)
            for code, test_code in zip(codes, test_codes)
        ]

        return await asyncio.gather(*tasks)


def get_executor(use_gke: bool = True, **kwargs):
    """
    Get appropriate executor based on environment.

    Args:
        use_gke: Whether to use GKE (default True in production)
        **kwargs: Arguments passed to executor constructor

    Returns:
        GKEExecutor or LocalExecutor
    """
    if use_gke and os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return GKEExecutor(**kwargs)
    else:
        logger.warning("Using LocalExecutor - not suitable for untrusted code!")
        return LocalExecutor(**kwargs)
