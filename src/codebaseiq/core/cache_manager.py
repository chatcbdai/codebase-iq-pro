#!/usr/bin/env python3
"""
Cache Manager for CodebaseIQ Pro
Handles persistent caching, file change detection, and incremental updates.
"""

import hashlib
import json
import logging
import os
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import subprocess

logger = logging.getLogger(__name__)

@dataclass
class FileChangeInfo:
    """Information about a changed file"""
    path: str
    old_hash: Optional[str]
    new_hash: str
    change_type: str  # 'added', 'modified', 'deleted'
    
@dataclass
class CacheMetadata:
    """Metadata for cached analysis"""
    analysis_timestamp: str
    codebase_path: str
    codebase_name: str
    files_analyzed: int
    file_hashes: Dict[str, str]
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    
class CacheManager:
    """Manages analysis caching and incremental updates"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager"""
        self.cache_dir = cache_dir or (Path.home() / ".codebaseiq" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CacheManager initialized with directory: {self.cache_dir}")
        
    def _get_cache_path(self, codebase_path: Path, analysis_type: str) -> Path:
        """Get cache file path for specific codebase and analysis type"""
        codebase_name = codebase_path.name
        cache_subdir = self.cache_dir / codebase_name
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{analysis_type}_analysis.json"
        
    async def _hash_file(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash {file_path}: {e}")
            return ""
            
    async def hash_files(self, file_paths: Dict[str, Path]) -> Dict[str, str]:
        """Hash multiple files concurrently"""
        tasks = {
            rel_path: self._hash_file(full_path)
            for rel_path, full_path in file_paths.items()
        }
        
        results = {}
        for rel_path, task in tasks.items():
            results[rel_path] = await task
            
        return results
        
    def get_git_info(self, codebase_path: Path) -> Dict[str, str]:
        """Get current git commit and branch info"""
        git_info = {}
        try:
            # Get current commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=codebase_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                git_info['commit'] = result.stdout.strip()
                
            # Get current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=codebase_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
                
            # Get latest commit message
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=format:%s'],
                cwd=codebase_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                git_info['latest_message'] = result.stdout.strip()
                
        except Exception as e:
            logger.warning(f"Failed to get git info: {e}")
            
        return git_info
        
    def get_git_changes_since(self, codebase_path: Path, since_commit: str) -> List[str]:
        """Get list of files changed since a specific commit"""
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', since_commit],
                cwd=codebase_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except Exception as e:
            logger.warning(f"Failed to get git changes: {e}")
            
        return []
        
    async def save_analysis(self, 
                          codebase_path: Path,
                          analysis_type: str,
                          analysis_data: Dict[str, Any],
                          file_hashes: Dict[str, str]) -> None:
        """Save analysis results with metadata"""
        cache_path = self._get_cache_path(codebase_path, analysis_type)
        git_info = self.get_git_info(codebase_path)
        
        metadata = CacheMetadata(
            analysis_timestamp=datetime.now().isoformat(),
            codebase_path=str(codebase_path),
            codebase_name=codebase_path.name,
            files_analyzed=len(file_hashes),
            file_hashes=file_hashes,
            git_commit=git_info.get('commit'),
            git_branch=git_info.get('branch')
        )
        
        cache_data = {
            'metadata': asdict(metadata),
            'analysis': analysis_data,
            'git_info': git_info
        }
        
        async with aiofiles.open(cache_path, 'w') as f:
            await f.write(json.dumps(cache_data, indent=2, default=str))
            
        logger.info(f"Saved {analysis_type} analysis to {cache_path}")
        
    async def load_analysis(self, 
                          codebase_path: Path,
                          analysis_type: str) -> Optional[Dict[str, Any]]:
        """Load cached analysis if it exists"""
        cache_path = self._get_cache_path(codebase_path, analysis_type)
        
        if not cache_path.exists():
            return None
            
        try:
            async with aiofiles.open(cache_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None
            
    async def detect_changes(self,
                           codebase_path: Path,
                           current_files: Dict[str, Path],
                           cached_data: Dict[str, Any]) -> Tuple[List[FileChangeInfo], bool]:
        """Detect which files have changed since last analysis"""
        changes = []
        needs_full_reanalysis = False
        
        cached_metadata = cached_data.get('metadata', {})
        cached_hashes = cached_metadata.get('file_hashes', {})
        
        # Check git commit changes
        git_info = self.get_git_info(codebase_path)
        cached_commit = cached_metadata.get('git_commit')
        
        if cached_commit and git_info.get('commit') != cached_commit:
            # Use git to find changed files
            git_changes = self.get_git_changes_since(codebase_path, cached_commit)
            logger.info(f"Git reports {len(git_changes)} changed files since {cached_commit[:8]}")
        else:
            git_changes = []
            
        # Hash current files
        current_hashes = await self.hash_files(current_files)
        
        # Find added and modified files
        for rel_path, new_hash in current_hashes.items():
            old_hash = cached_hashes.get(rel_path)
            
            if not old_hash:
                changes.append(FileChangeInfo(rel_path, None, new_hash, 'added'))
            elif old_hash != new_hash:
                changes.append(FileChangeInfo(rel_path, old_hash, new_hash, 'modified'))
                
        # Find deleted files
        for rel_path, old_hash in cached_hashes.items():
            if rel_path not in current_hashes:
                changes.append(FileChangeInfo(rel_path, old_hash, '', 'deleted'))
                
        # Check if structural changes require full reanalysis
        if len(changes) > len(current_files) * 0.3:  # More than 30% changed
            logger.warning(f"{len(changes)} files changed - recommending full reanalysis")
            needs_full_reanalysis = True
            
        # Check for critical file changes
        critical_patterns = ['requirements.txt', 'package.json', 'pyproject.toml', 
                           'setup.py', 'tsconfig.json', '.gitignore']
        for change in changes:
            if any(pattern in change.path for pattern in critical_patterns):
                logger.warning(f"Critical file {change.path} changed - recommending full reanalysis")
                needs_full_reanalysis = True
                break
                
        return changes, needs_full_reanalysis
        
    async def get_dependent_files(self,
                                changed_files: List[str],
                                dependency_graph: Dict[str, List[str]]) -> Set[str]:
        """Get all files that depend on changed files"""
        dependent_files = set(changed_files)
        
        # Build reverse dependency map
        reverse_deps = {}
        for file, deps in dependency_graph.items():
            for dep in deps:
                if dep not in reverse_deps:
                    reverse_deps[dep] = []
                reverse_deps[dep].append(file)
                
        # Find all dependent files
        to_check = list(changed_files)
        while to_check:
            current = to_check.pop()
            if current in reverse_deps:
                for dependent in reverse_deps[current]:
                    if dependent not in dependent_files:
                        dependent_files.add(dependent)
                        to_check.append(dependent)
                        
        return dependent_files
        
    def get_latest_changes_summary(self, codebase_path: Path, 
                                 cached_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get summary of latest changes based on git history"""
        git_info = self.get_git_info(codebase_path)
        
        summary = {
            'current_commit': git_info.get('commit', 'unknown')[:8],
            'current_branch': git_info.get('branch', 'unknown'),
            'latest_commit_message': git_info.get('latest_message', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        if cached_data:
            cached_commit = cached_data.get('metadata', {}).get('git_commit')
            if cached_commit and cached_commit != git_info.get('commit'):
                # Get commit count since cache
                try:
                    result = subprocess.run(
                        ['git', 'rev-list', '--count', f"{cached_commit}..HEAD"],
                        cwd=codebase_path,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        summary['commits_since_cache'] = int(result.stdout.strip())
                        
                    # Get recent commit messages
                    result = subprocess.run(
                        ['git', 'log', '--oneline', '-5', f"{cached_commit}..HEAD"],
                        cwd=codebase_path,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        summary['recent_commits'] = result.stdout.strip().split('\n')
                        
                except Exception as e:
                    logger.warning(f"Failed to get commit history: {e}")
                    
        return summary
        
    async def cleanup_old_caches(self, max_age_days: int = 30) -> int:
        """Clean up old cache files"""
        count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        for cache_file in self.cache_dir.rglob("*_analysis.json"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to clean up {cache_file}: {e}")
                
        if count > 0:
            logger.info(f"Cleaned up {count} old cache files")
            
        return count