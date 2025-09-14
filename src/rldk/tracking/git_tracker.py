"""
Git tracking for capturing repository state and changes.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

from ..config import settings
from ..utils.runtime import with_timeout


class GitTracker:
    """Tracks Git repository state and changes."""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self.tracking_info: Dict[str, Any] = {}
        self._settings = settings
        self._cache_dir = self._settings.get_cache_dir() / "git_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "git_info.json"

    def capture_git_state(
        self,
        capture_commit: bool = True,
        capture_diff: bool = True,
        capture_status: bool = True,
        capture_remote: bool = True
    ) -> Dict[str, Any]:
        """
        Capture comprehensive Git repository state with caching.

        Args:
            capture_commit: Whether to capture commit information
            capture_diff: Whether to capture diff information
            capture_status: Whether to capture status information
            capture_remote: Whether to capture remote information

        Returns:
            Dictionary containing Git state information
        """
        # Check cache first if caching is enabled
        if self._settings.cache_git_info:
            cached_info = self._load_from_cache()
            if cached_info and self._is_cache_valid(cached_info):
                # Update timestamp
                cached_info["timestamp"] = self._get_timestamp()
                self.tracking_info = cached_info
                return cached_info

        git_info = {
            "timestamp": self._get_timestamp(),
            "repo_path": str(self.repo_path.absolute())
        }

        try:
            if capture_commit:
                git_info["commit"] = self._capture_commit_info()

            if capture_diff:
                git_info["diff"] = self._capture_diff_info()

            if capture_status:
                git_info["status"] = self._capture_status_info()

            if capture_remote:
                git_info["remote"] = self._capture_remote_info()

            # Compute Git fingerprint
            git_info["git_checksum"] = self._compute_git_checksum(git_info)

        except Exception as e:
            git_info["error"] = f"Failed to capture git state: {str(e)}"

        # Save to cache
        if self._settings.cache_git_info:
            self._save_to_cache(git_info)

        self.tracking_info = git_info
        return git_info

    @with_timeout(10.0)  # 10 second timeout for commit info capture
    def _capture_commit_info(self) -> Dict[str, Any]:
        """Capture current commit information with timeout."""
        commit_info = {}

        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self._settings.git_timeout
            )
            if result.returncode == 0:
                commit_info["hash"] = result.stdout.strip()
            else:
                commit_info["hash"] = "not available"
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            commit_info["hash"] = "git not available"

        try:
            # Get short commit hash
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                commit_info["short_hash"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            commit_info["short_hash"] = "not available"

        try:
            # Get commit message
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%s"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                commit_info["message"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            commit_info["message"] = "not available"

        try:
            # Get commit author and date
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%an|%ae|%ad", "--date=iso"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split('|')
                if len(parts) >= 3:
                    commit_info["author"] = {
                        "name": parts[0],
                        "email": parts[1],
                        "date": parts[2]
                    }
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            commit_info["author"] = "not available"

        try:
            # Get branch information
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                commit_info["branch"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            commit_info["branch"] = "not available"

        try:
            # Get tag information
            result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                commit_info["tag"] = result.stdout.strip()
            else:
                commit_info["tag"] = None
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            commit_info["tag"] = "not available"

        return commit_info

    def _capture_diff_info(self) -> Dict[str, Any]:
        """Capture diff information."""
        diff_info = {}

        try:
            # Get working directory diff
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                diff_info["modified_files"] = result.stdout.strip().split('\n') if result.stdout.strip() else []
            else:
                diff_info["modified_files"] = []
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            diff_info["modified_files"] = "git not available"

        try:
            # Get staged changes
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                diff_info["staged_files"] = result.stdout.strip().split('\n') if result.stdout.strip() else []
            else:
                diff_info["staged_files"] = []
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            diff_info["staged_files"] = "git not available"

        try:
            # Get untracked files
            result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                diff_info["untracked_files"] = result.stdout.strip().split('\n') if result.stdout.strip() else []
            else:
                diff_info["untracked_files"] = []
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            diff_info["untracked_files"] = "git not available"

        # Compute diff checksum
        if isinstance(diff_info["modified_files"], list):
            diff_info["diff_checksum"] = hashlib.sha256(
                str(sorted(diff_info["modified_files"])).encode()
            ).hexdigest()
        else:
            diff_info["diff_checksum"] = "not available"

        return diff_info

    def _capture_status_info(self) -> Dict[str, Any]:
        """Capture Git status information."""
        status_info = {}

        try:
            # Get detailed status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                status_info["porcelain"] = result.stdout.strip().split('\n') if result.stdout.strip() else []
            else:
                status_info["porcelain"] = []
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            status_info["porcelain"] = "git not available"

        try:
            # Get status summary
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                status_info["short"] = result.stdout.strip()
            else:
                status_info["short"] = "not available"
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            status_info["short"] = "git not available"

        return status_info

    def _capture_remote_info(self) -> Dict[str, Any]:
        """Capture remote repository information."""
        remote_info = {}

        try:
            # Get remote URLs
            result = subprocess.run(
                ["git", "remote", "-v"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                remotes = {}
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[0]
                            url = parts[1]
                            remotes[name] = url
                remote_info["remotes"] = remotes
            else:
                remote_info["remotes"] = {}
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            remote_info["remotes"] = "git not available"

        try:
            # Get current remote tracking branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                remote_info["upstream"] = result.stdout.strip()
            else:
                remote_info["upstream"] = None
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            remote_info["upstream"] = "not available"

        return remote_info

    def _compute_git_checksum(self, git_info: Dict[str, Any]) -> str:
        """Compute checksum of Git information."""
        # Create a simplified version for hashing
        hash_info = {
            "commit_hash": git_info.get("commit", {}).get("hash"),
            "branch": git_info.get("commit", {}).get("branch"),
            "modified_files": git_info.get("diff", {}).get("modified_files", []),
            "staged_files": git_info.get("diff", {}).get("staged_files", []),
            "untracked_files": git_info.get("diff", {}).get("untracked_files", [])
        }

        # Convert to JSON string and hash
        json_str = json.dumps(hash_info, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load git info from cache."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, git_info: Dict[str, Any]) -> None:
        """Save git info to cache."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(git_info, f, indent=2, default=str)
        except Exception:
            pass  # Cache failures shouldn't break the main functionality

    def _is_cache_valid(self, cached_info: Dict[str, Any]) -> bool:
        """Check if cached git info is still valid."""
        # Cache is valid for 30 minutes
        cache_age = time.time() - cached_info.get("cache_timestamp", 0)
        return cache_age < 1800  # 30 minutes

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of Git tracking."""
        return self.tracking_info

    def is_git_repo(self) -> bool:
        """Check if the current path is a Git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_repo_info(self) -> Dict[str, Any]:
        """Get basic repository information."""
        if not self.is_git_repo():
            return {"is_git_repo": False}

        return {
            "is_git_repo": True,
            "repo_path": str(self.repo_path.absolute()),
            "git_info": self.capture_git_state()
        }
