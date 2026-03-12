"""Kinship matrix computation.

This module provides the main kinship matrix computation function,
implementing GEMMA's centered relatedness matrix algorithm (-gk 1 mode).

The kinship matrix K is computed as:
    K = (1/p) * X_c @ X_c.T

where X_c is the centered genotype matrix with missing values imputed
to per-SNP mean, and p is the number of SNPs.

The standard (non-LOCO) kinship computation and in-memory LOCO kinship
use numpy.matmul exclusively, so JAX is never initialized during kinship
or eigendecomp phases. The streaming LOCO function
(compute_loco_kinship_streaming) uses JAX for GPU-accelerated accumulation.

LOCO (Leave-One-Chromosome-Out) kinship is also supported via the
subtraction approach: K_loco_c = (S_full - S_c) / (p - p_c), where
S_full is the unscaled full kinship numerator and S_c is the contribution
from chromosome c. This avoids redundant computation.
"""

from __future__ import annotations

import gc
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax.numpy as jnp
import psutil
from loguru import logger

from jamma.core.memory import (
    check_memory_available,
    estimate_eigendecomp_memory,
    estimate_streaming_memory,
    log_memory_snapshot,
)
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.io.plink import (
    get_plink_metadata,
    partitions_from_metadata,
    stream_genotype_chunks,
)
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize
from jamma.utils import chr_sort_key

import gzip


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Return arr as float64, copying only when the dtype differs."""
    return arr if arr.dtype == np.float64 else arr.astype(np.float64)


def _accumulate_kinship(K: np.ndarray, X_centered: np.ndarray) -> np.ndarray:
    """Accumulate kinship contribution from centered SNP batch.

    Uses numpy.matmul (backed by MKL/OpenBLAS dgemm) with in-place
    accumulation. The non-LOCO kinship path uses this exclusively so
    that JAX is never initialized during kinship computation.

    Args:
        K: Current kinship matrix accumulator (n_samples, n_samples)
        X_centered: Centered genotype batch (n_samples, batch_snps)

    Returns:
        Updated kinship matrix with batch contribution added.
    """
    K += np.matmul(X_centered, X_centered.T)
    return K


def _filter_snps(
    genotypes: np.ndarray,
    maf_threshold: float,
    miss_threshold: float,
) -> tuple[np.ndarray, int, int]:
    """Filter SNPs by MAF, missing rate, and monomorphism.

    Monomorphic SNPs (variance == 0) are always filtered to match GEMMA.
    Delegates to shared utilities in jamma.core.snp_filter.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
        maf_threshold: Minimum MAF for inclusion (0 to disable MAF filter only).
        miss_threshold: Maximum missing rate for inclusion (1.0 to disable).

    Returns:
        Tuple of (filtered_genotypes, n_filtered, n_original).
    """
    n_samples, n_snps = genotypes.shape

    col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, _allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, miss_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )

    n_filtered = int(np.sum(snp_mask))

    if n_filtered == 0:
        return genotypes[:, :0], 0, n_snps  # Empty array

    return genotypes[:, snp_mask], n_filtered, n_snps


def _compute_kinship_inmemory(
    genotypes: np.ndarray,
    transform_fn: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
    maf_threshold: float,
    miss_threshold: float,
    check_memory: bool,
    label: str,
) -> np.ndarray:
    """Shared implementation for in-memory kinship computation.

    Implements K = (1/p) * sum(transform(X_batch) @ transform(X_batch).T)
    where transform is either centering (gk=1) or centering+standardizing (gk=2).

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
        transform_fn: Per-batch transformation (impute_and_center or
            impute_center_and_standardize).
        batch_size: SNPs per batch.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        check_memory: Check available memory before allocation.
        label: Label for logging (e.g. "Kinship", "Standardized kinship").

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        ValueError: If no SNPs pass filtering.
    """
    n_samples, _ = genotypes.shape

    genotypes_filtered, n_snps, n_original = _filter_snps(
        genotypes, maf_threshold, miss_threshold
    )

    if n_snps == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_original}"
        )

    if n_snps < n_original:
        n_removed = n_original - n_snps
        logger.info(
            f"{label} filtering: {n_snps:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    if check_memory:
        eigendecomp_peak_gb = estimate_eigendecomp_memory(n_samples)
        kinship_peak_gb = n_samples**2 * 8 / 1e9 + n_samples * n_snps * 8 / 1e9
        required_gb = max(eigendecomp_peak_gb, kinship_peak_gb)
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=f"GWAS pipeline (peak: {required_gb:.1f}GB)",
        )

    log_memory_snapshot(f"before_{label.lower().replace(' ', '_')}_{n_samples}samples")

    X = _ensure_float64(genotypes_filtered)
    K = np.zeros((n_samples, n_samples), dtype=np.float64)

    n_batches = (n_snps + batch_size - 1) // batch_size
    logger.info(
        f"{label}: in-memory mode, {n_samples:,} samples x {n_snps:,} SNPs, "
        f"{n_batches} batches of {batch_size:,}"
    )

    batch_starts = list(range(0, n_snps, batch_size))
    if n_batches > 1:
        batch_iter = progress_iterator(
            enumerate(batch_starts), total=n_batches, desc=label
        )
    else:
        batch_iter = enumerate(batch_starts)

    for _, start in batch_iter:
        end = min(start + batch_size, n_snps)
        X_transformed = transform_fn(X[:, start:end])
        K = _accumulate_kinship(K, X_transformed)

    K = K / n_snps

    log_memory_snapshot(f"after_{label.lower().replace(' ', '_')}_{n_samples}samples")

    return K


def compute_centered_kinship(
    genotypes: np.ndarray,
    batch_size: int = 10000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
) -> np.ndarray:
    """Compute centered relatedness matrix (GEMMA -gk 1).

    Implements: K = (1/p) * X_c @ X_c.T
    where X_c is centered with missing values imputed to SNP mean.

    Note: Monomorphic SNPs (constant genotype) are always excluded to match GEMMA.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
            Values are typically 0, 1, or 2 representing minor allele counts.
        batch_size: SNPs per batch (default 10000, matches GEMMA).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation
            and raise MemoryError if insufficient.

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        ValueError: If no SNPs pass filtering.

    Example:
        >>> import numpy as np
        >>> X = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0]], dtype=np.float64)
        >>> K = compute_centered_kinship(X, maf_threshold=0.01)
        >>> K.shape
        (3, 3)
        >>> np.allclose(K, K.T)  # Symmetric
        True
    """
    return _compute_kinship_inmemory(
        genotypes,
        impute_and_center,
        batch_size,
        maf_threshold,
        miss_threshold,
        check_memory,
        "Kinship",
    )


def compute_standardized_kinship(
    genotypes: np.ndarray,
    batch_size: int = 10000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
) -> np.ndarray:
    """Compute standardized relatedness matrix (GEMMA -gk 2).

    Implements K = (1/p) * Z @ Z.T where Z[i,k] = (x[i,k] - mean_k) / sd_k.
    Each SNP is centered and divided by its standard deviation.

    Note: Monomorphic SNPs are excluded by _filter_snps (which removes
    zero-variance SNPs). This matches GEMMA since monomorphic SNPs also
    fail MAF filtering in practice.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
            Values are typically 0, 1, or 2 representing minor allele counts.
        batch_size: SNPs per batch (default 10000, matches GEMMA).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation
            and raise MemoryError if insufficient.

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        ValueError: If no SNPs pass filtering.

    Example:
        >>> import numpy as np
        >>> X = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0]], dtype=np.float64)
        >>> K = compute_standardized_kinship(X, maf_threshold=0.01)
        >>> K.shape
        (3, 3)
        >>> np.allclose(K, K.T)  # Symmetric
        True
    """
    return _compute_kinship_inmemory(
        genotypes,
        impute_center_and_standardize,
        batch_size,
        maf_threshold,
        miss_threshold,
        check_memory,
        "Standardized kinship",
    )


def compute_loco_kinship(
    genotypes: np.ndarray,
    chromosome_for_each_snp: np.ndarray,
    batch_size: int = 10000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
) -> Iterator[tuple[str, np.ndarray]]:
    """Compute LOCO kinship matrices via subtraction approach.

    For each chromosome c, computes K_loco_c = (S_full - S_c) / (p - p_c)
    where S_full is the unscaled full kinship numerator and S_c is the
    contribution from chromosome c's SNPs.

    Global centering is used (not per-chromosome centering) so the
    subtraction identity holds: S_full = sum(S_c) over all chromosomes.

    Yields one (chr_name, K_loco) pair at a time so the caller can process
    and discard each matrix without holding all LOCO matrices in memory.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
        chromosome_for_each_snp: String array of chromosome name per SNP,
            length must equal genotypes.shape[1].
        batch_size: SNPs per batch for kinship accumulation (default 10000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation.

    Yields:
        Tuple of (chr_name, K_loco) where chr_name is the chromosome being
        excluded and K_loco is the LOCO kinship matrix (n_samples, n_samples).

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        ValueError: If no SNPs pass filtering, or if all filtered SNPs are on
            a single chromosome (cannot compute LOCO).
    """
    n_samples, n_snps_original = genotypes.shape

    # Filter SNPs globally (MAF, missingness, monomorphism)
    # Compute mask once, use for both genotype and chromosome filtering
    col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, _allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, miss_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )

    n_filtered = int(np.sum(snp_mask))
    n_original = n_snps_original

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_original}"
        )

    genotypes_filtered = genotypes[:, snp_mask]
    chr_filtered = chromosome_for_each_snp[snp_mask]

    if n_filtered < n_original:
        n_removed = n_original - n_filtered
        logger.info(
            f"LOCO kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    # Memory check: S_full (n^2*8) + X_centered (n*p*8) + one S_c at a time (n^2*8)
    if check_memory:
        required_gb = (
            n_samples**2 * 8 / 1e9  # S_full
            + n_samples * n_filtered * 8 / 1e9  # X_centered (float64)
            + n_samples**2 * 8 / 1e9  # S_c (one at a time)
        )
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=f"LOCO kinship ({n_samples:,} samples, {n_filtered:,} SNPs)",
        )

    X = _ensure_float64(genotypes_filtered)
    X_centered = impute_and_center(X)

    # Accumulate full kinship numerator S_full = X_centered @ X_centered.T (unscaled)
    S_full = np.zeros((n_samples, n_samples), dtype=np.float64)
    n_batches = (n_filtered + batch_size - 1) // batch_size

    logger.info(
        f"LOCO kinship: {n_samples:,} samples x {n_filtered:,} SNPs, "
        f"{n_batches} batches"
    )

    batch_starts = list(range(0, n_filtered, batch_size))
    if n_batches > 1:
        batch_iter = progress_iterator(
            enumerate(batch_starts), total=n_batches, desc="LOCO: full kinship"
        )
    else:
        batch_iter = enumerate(batch_starts)

    for _, start in batch_iter:
        end = min(start + batch_size, n_filtered)
        batch = X_centered[:, start:end]
        S_full += np.matmul(batch, batch.T)

    # Compute per-chromosome LOCO kinship via subtraction
    unique_chrs = sorted(set(chr_filtered))
    logger.info(f"LOCO: computing {len(unique_chrs)} leave-one-out kinship matrices")

    for chr_name in unique_chrs:
        chr_mask = chr_filtered == chr_name
        p_chr = int(np.sum(chr_mask))
        p_loco = n_filtered - p_chr

        if p_loco == 0:
            raise ValueError(
                f"Cannot compute LOCO kinship: all {n_filtered} filtered SNPs "
                f"are on chromosome '{chr_name}'. LOCO requires SNPs on multiple "
                f"chromosomes."
            )

        # Compute chromosome contribution S_c
        X_chr = X_centered[:, chr_mask]
        S_chr = np.matmul(X_chr, X_chr.T)

        # K_loco = (S_full - S_c) / p_loco
        K_loco = (S_full - S_chr) / p_loco

        logger.debug(
            f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs retained"
        )

        yield (chr_name, K_loco)


def _kinship_single_pass(
    bed_path: Path,
    n_samples: int,
    n_snps: int,
    chunk_size: int,
    show_progress: bool,
) -> np.ndarray:
    """Single-pass kinship: compute stats and accumulate in one BED read.

    Only valid when no MAF/missing filters are active (maf_threshold=0.0,
    miss_threshold>=1.0, ksnps_indices=None). Monomorphism filtering
    (variance > 0) is applied per-chunk inline, matching the two-pass result.

    Args:
        bed_path: Path prefix for PLINK files.
        n_samples: Number of samples.
        n_snps: Total number of SNPs.
        chunk_size: Number of SNPs per chunk.
        show_progress: Whether to show progress bar.

    Returns:
        Kinship matrix (n_samples, n_samples).

    Raises:
        ValueError: If no SNPs pass monomorphism filter.
    """
    K = np.zeros((n_samples, n_samples), dtype=np.float64)
    n_filtered = 0

    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        chunk_iter = progress_iterator(
            chunk_iter, total=n_chunks, desc="Computing kinship (single-pass)"
        )

    for chunk, _start, _end in chunk_iter:
        # Per-chunk monomorphism filter: exclude constant genotype columns.
        # Suppress RuntimeWarning for all-NaN columns (no valid samples in chunk).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_vars = np.nanvar(chunk, axis=0)
        poly_mask = col_vars > 0
        n_poly = np.count_nonzero(poly_mask)
        if n_poly == 0:
            continue

        X_chunk = chunk[:, poly_mask]
        X_centered = impute_and_center(X_chunk)
        K = _accumulate_kinship(K, X_centered)
        n_filtered += n_poly
        del chunk, X_chunk, X_centered

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed monomorphism filter. Original SNP count: {n_snps}"
        )

    K /= n_filtered
    return K


def compute_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute centered relatedness matrix from disk-streamed genotypes.

    Implements: K = (1/p) * X_c @ X_c.T
    where X_c is centered with missing values imputed to SNP mean.

    This function reads genotype chunks directly from disk via bed-reader
    windowed reads, avoiding the need to load the full genotype matrix.

    Two-pass approach for filtering:
    1. First pass: compute per-SNP MAF, missing rate, variance for filtering
    2. Second pass: accumulate kinship from filtered SNPs only

    Note: Monomorphic SNPs (constant genotype) are always excluded to match GEMMA.

    Memory behavior:
        O(n^2 + n*chunk_size) vs O(n^2 + n*p) for full-load version.
        Only kinship (n^2) + one chunk (n*chunk_size) in memory at a time.
        Each chunk is freed after accumulation (Python GC).

    Use case:
        When genotypes don't fit in memory. At 200k samples and 95k SNPs,
        full genotypes are 76GB; streaming eliminates this allocation.

    Result equivalence:
        Produces identical kinship to compute_centered_kinship() within
        numerical precision (< 1e-10 relative tolerance).

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation
            and raise MemoryError if insufficient.
        show_progress: If True (default), show progress bar during iteration.
        ksnps_indices: Pre-resolved column indices for -ksnps restriction, or None.

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        FileNotFoundError: If the PLINK .bed file does not exist.
        ValueError: If no SNPs pass filtering.

    Example:
        >>> from pathlib import Path
        >>> K = compute_kinship_streaming(Path("data/my_study"), maf_threshold=0.01)
        >>> K.shape
        (1940, 1940)
    """
    start_time = time.perf_counter()

    # Get dimensions without loading genotypes
    meta = get_plink_metadata(bed_path)
    n_samples = meta["n_samples"]
    n_snps = meta["n_snps"]

    from jamma.core.estimates import estimate_kinship_time

    logger.info("Computing Kinship Matrix")
    logger.info(f"  Individuals: {n_samples:,}")
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chunk size: {chunk_size:,}")
    logger.info(f"  Estimated time: {estimate_kinship_time(n_samples, n_snps)}")

    # Memory check before allocation
    # Check against full pipeline peak (eigendecomp) since it always follows kinship.
    if check_memory:
        est = estimate_streaming_memory(n_samples, chunk_size=chunk_size)
        check_memory_available(
            est.total_peak_gb,
            safety_margin=0.1,
            operation=f"GWAS pipeline (eigendecomp peak: {est.total_peak_gb:.1f}GB)",
        )

    # Single-pass optimization: when no MAF/missing filters are active and
    # no ksnps restriction, monomorphism filtering can be done inline per-chunk.
    # This eliminates the stats-only BED read (pass 1), halving I/O at scale
    # (e.g. ~76 GB saved at 200k samples x 95k SNPs).
    use_single_pass = (
        maf_threshold == 0.0 and miss_threshold >= 1.0 and ksnps_indices is None
    )

    if use_single_pass:
        logger.debug("Kinship: single-pass mode (no MAF/missing filters)")
        K = _kinship_single_pass(bed_path, n_samples, n_snps, chunk_size, show_progress)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Kinship matrix computed in {elapsed:.2f}s")
        return K

    # === PASS 1: Compute per-SNP statistics for filtering ===
    # Always compute stats for monomorphic filtering (GEMMA behavior)
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks, desc="Computing SNP statistics"
        )

    for chunk, start, end in stats_iterator:
        chunk_miss_counts = np.sum(np.isnan(chunk), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chunk_means = np.nanmean(chunk, axis=0)
            chunk_vars = np.nanvar(chunk, axis=0)
        chunk_means = np.nan_to_num(chunk_means, nan=0.0)
        chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

        all_means[start:end] = chunk_means
        all_miss_counts[start:end] = chunk_miss_counts
        all_vars[start:end] = chunk_vars
        del chunk  # Free ~1.6GB per chunk at scale before next iteration

    # Compute filters (free source arrays immediately after deriving values)
    miss_rates = all_miss_counts / n_samples
    del all_miss_counts
    allele_freqs = all_means / 2.0
    del all_means
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)
    is_polymorphic = all_vars > 0
    del all_vars

    # Combined filter: MAF, missing rate, and monomorphism (always applied)
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic

    # Apply kinship SNP list restriction (if -ksnps provided)
    if ksnps_indices is not None:
        from jamma.core.snp_filter import apply_snp_list_mask

        apply_snp_list_mask(snp_mask, ksnps_indices, n_snps, "Kinship SNP list")

    n_filtered = int(np.sum(snp_mask))

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_snps}"
        )

    if n_filtered < n_snps:
        n_removed = n_snps - n_filtered
        logger.info(
            f"Kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )
    else:
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")

    # Get indices of SNPs that passed filtering
    snp_indices = np.where(snp_mask)[0]

    # Initialize kinship accumulator (numpy — no JAX device memory)
    K = np.zeros((n_samples, n_samples), dtype=np.float64)

    # === PASS 2: Accumulate kinship from filtered SNPs ===
    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )

    if show_progress:
        chunk_iter = progress_iterator(
            chunk_iter, total=n_chunks, desc="Computing kinship"
        )

    for chunk, file_start, file_end in chunk_iter:
        # Binary search for filtered SNPs in this chunk: O(log n) vs O(n)
        # snp_indices is sorted (from np.where), so searchsorted is valid
        left = np.searchsorted(snp_indices, file_start, side="left")
        right = np.searchsorted(snp_indices, file_end, side="left")
        chunk_filtered_indices = snp_indices[left:right] - file_start

        if len(chunk_filtered_indices) == 0:
            continue

        # Extract only filtered columns (float64 for numerical accuracy)
        X_chunk = np.asarray(chunk[:, chunk_filtered_indices], dtype=np.float64)

        # Impute and center the chunk
        X_centered = impute_and_center(X_chunk)

        # Accumulate kinship contribution (in-place numpy matmul)
        K = _accumulate_kinship(K, X_centered)

    # Scale by number of filtered SNPs
    K = K / n_filtered

    elapsed = time.perf_counter() - start_time
    logger.info(f"Kinship matrix computed in {elapsed:.2f}s")

    return K


# ---------------------------------------------------------------------------
# paintSparse kinship helper
# ---------------------------------------------------------------------------

def _read_fam_sample_ids(fam_path: Path) -> np.ndarray:
    """Read PLINK .fam file and return array of (FID, IID) strings.

    The returned array has shape (n_samples, 2). Empty lines are skipped.
    This utility is intentionally minimal to avoid importing PLINK readers when
    only the sample order (and count) is required.
    """
    ids: list[tuple[str, str]] = []
    with open(fam_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid .fam line (expected at least 2 columns): '{line.strip()}'"
                )
            ids.append((parts[0], parts[1]))
    if not ids:
        raise ValueError(f".fam file appears empty: {fam_path}")
    return np.array(ids, dtype="<U")


def compute_kinship_from_paint_sparse(
    path_weight_pairs: Sequence[tuple[Path | str, float]],
    fam_path: Path | str,
    paint_sample_ids: np.ndarray | None = None,
    divide_by: float | None = None,
) -> np.ndarray:
    """Construct kinship (GRM) from paintSparse chunk‐length output files.

    PaintSparse produces sparse 3-column files ("chr$i.chunklengths.s.out.gz")
    where each row contains ``IND1 IND2 LENGTH``. IND* are **one-based**
    individual indices corresponding to the order of samples in the input VCF
    provided to PBWTpaint. The third column records the total chunk length
    copied from IND2 by IND1 in units of SNPs (which is typically converted to
    centimorgans in a downstream weighting step).

    JAMMA can interpret these outputs as a kinship matrix by summing the
    reported lengths for each pair of individuals. The returned matrix has
    shape (n_samples, n_samples) and is ordered to match the sample order in a
    PLINK ``.fam`` file (row/column i corresponds to the i-th line of ``fam``).

    This implementation mirrors the behaviour of
    ``combine_chunklength.cpp`` from the SparsePainter repository: each
    input file may be scaled by an optional weight prior to aggregation.  By
    default all files are treated equally (weight=1), and the final matrix may
    still be normalised with ``divide_by``.

    Args:
        path_weight_pairs: Sequence of ``(path, weight)`` tuples describing
            the chunklength files to include and the scalar weight to apply to
            each. ``path`` may be a string or ``Path``; it must name an exact
            existing file – glob patterns or wildcards are disallowed.  A bare
            path string is equivalent to ``(path, 1.0)``.
        fam_path: Path to a PLINK ``.fam`` file describing the same set of
            individuals. The number of lines (samples) determines the size of
            the resulting kinship matrix and provides the canonical ordering.
        paint_sample_ids: Optional array of sample identifiers in the order
            used by paintSparse. Each element may be a tuple ``(FID, IID)`` or
            a single string IID; the array length must equal the number of
            samples. When provided, the function will reorder rows/columns so
            that paintSparse indices are matched to their position in the
            ``.fam`` file. If ``None`` (default) an identity mapping is
            assumed, i.e. paintSparse used the same sample order as ``.fam``.
        divide_by: Optional scalar. If given, the final matrix is divided by
            this value. This is convenient for normalising by the total
            genetic length (for example ~3545.04 cM in the standard map).

    Returns:
        ``np.ndarray`` of shape ``(n_samples, n_samples)`` containing the
        symmetrized kinship matrix (covariance matrix) derived from the
        weighted, aggregated chunk lengths. The matrix is symmetric, and
        diagonal entries reflect self-copying if those values are present in
        the input files.

    Raises:
        ValueError: If the ``.fam`` file is empty, if sample counts mismatch,
            if a paintSparse sample identifier cannot be found in the fam
            ordering, if a path-weight entry is malformed, if a weight is not
            a number, or if an entry contains glob metacharacters.
    """
    # Interpret ``path_weight_pairs`` as an iterable of
    # (path, weight) tuples.  Plain strings/Paths are accepted as shorthand
    # for a weight of 1.0 to preserve convenience.
    entries: list[tuple[Path, float]] = []
    for item in path_weight_pairs:
        if isinstance(item, (str, Path)):
            # bare path -> weight 1.0
            path_str = str(item)
            wt = 1.0
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            path_str, wt = item
            wt = float(wt)
        else:
            raise ValueError(
                "each entry must be (path, weight) or a bare path string"
            )

        # require that the provided path contains no glob metacharacters
        if any(ch in path_str for ch in "*?[]"):
            raise ValueError(
                f"path '{path_str}' contains wildcard characters; full path required"
            )
        entries.append((Path(path_str), wt))

    paths = [p for p, _ in entries]
    weights_list = [w for _, w in entries]

    fam_path = Path(fam_path)
    fam_ids = _read_fam_sample_ids(fam_path)
    n_samples = fam_ids.shape[0]

    # build mapping from paint index -> fam index
    if paint_sample_ids is not None:
        paint_arr = np.asarray(paint_sample_ids)
        if paint_arr.shape[0] != n_samples:
            raise ValueError(
                "paint_sample_ids length does not match number of .fam samples"
            )
        # mapping logic: handle 1D IID list or 2D (FID,IID) list
        if paint_arr.ndim == 1:
            fam_iids = fam_ids[:, 1]
            mapping = np.empty(n_samples, dtype=np.int64)
            for i, pid in enumerate(paint_arr):
                matches = np.where(fam_iids == pid)[0]
                if matches.size == 0:
                    raise ValueError(f"Paint sample '{pid}' not found in fam")
                mapping[i] = matches[0]
        else:
            fam_map = {tuple(row): idx for idx, row in enumerate(fam_ids)}
            mapping = np.empty(n_samples, dtype=np.int64)
            for i, row in enumerate(paint_arr):
                key = tuple(row)
                if key not in fam_map:
                    raise ValueError(f"Paint sample {row} not found in fam")
                mapping[i] = fam_map[key]
    else:
        mapping = np.arange(n_samples, dtype=np.int64)

    # allocate kinship accumulator
    K = np.zeros((n_samples, n_samples), dtype=np.float64)

    # parse each file and add weighted contributions
    for path, wt in zip(paths, weights_list, strict=True):
        if not path.exists():
            raise FileNotFoundError(f"PaintSparse file not found: {path}")
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 3:
                    continue  # ignore malformed lines silently
                try:
                    i = int(parts[0]) - 1
                    j = int(parts[1]) - 1
                    val = float(parts[2]) * wt
                except ValueError:
                    # skip lines that don't parse as numbers
                    continue
                if i < 0 or j < 0 or i >= n_samples or j >= n_samples:
                    raise ValueError(
                        f"Index out of bounds in {path}: {i+1}, {j+1}"
                    )
                pi = mapping[i]
                pj = mapping[j]
                K[pi, pj] += val

    if divide_by is not None:
        K = K / divide_by

    # Symmetrize the matrix to create covariance matrix: (K + K.T) / 2
    # This accounts for directed donor/recipient sharing in paintSparse output
    K = (K + K.T) / 2

    # ensure diagonal reflects the maximum sharing for each individual
    # otherwise the matrix diagaonal is zero and the matrix is not positive semi-definite, 
    # which causes issues for eigendecomposition
    np.fill_diagonal(K, np.max(K, axis=1))

    return K


def _yield_full_kinship_fallback(
    S_full_np: np.ndarray,
    chrs_without_snps: list[str],
    n_filtered: int,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield full kinship for chromosomes with 0 filtered SNPs.

    When a chromosome has no SNPs after filtering, there is nothing to leave
    out, so K_loco equals K_full.

    Divides S_full_np in-place to avoid allocating a separate K_full buffer
    (saves n^2 * 8 bytes — 320GB at 200k samples).  Callers must not use
    S_full_np after this function returns.

    Args:
        S_full_np: Full kinship numerator as numpy array (n_samples, n_samples).
            **Consumed in-place** — contents are overwritten with K_full.
        chrs_without_snps: Chromosomes with 0 filtered SNPs.
        n_filtered: Total number of filtered SNPs.

    Yields:
        (chr_name, K_full) pairs in biological chromosome order.
        Each matrix is an independent allocation (safe to mutate in-place).
    """
    if not chrs_without_snps:
        return
    if n_filtered == 0:
        raise ValueError(
            "Cannot compute fallback kinship: n_filtered is 0 "
            "(no SNPs passed filtering)"
        )
    # In-place division: S_full_np becomes K_full, no extra n^2 allocation.
    S_full_np /= n_filtered
    S_full_np.flags.writeable = False  # Guard against accidental re-mutation
    for chr_name in sorted(chrs_without_snps, key=chr_sort_key):
        logger.debug(f"LOCO chr {chr_name}: 0 SNPs after filtering, using full kinship")
        yield (chr_name, S_full_np.copy())


def _yield_loco_matrices(
    S_full_np: np.ndarray,
    S_chr: dict[str, jnp.ndarray],
    n_chr_filtered: dict[str, int],
    n_filtered: int,
    K_loco_buf: np.ndarray | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Compute and yield LOCO kinship matrices from S_full and per-chr accumulators.

    For each chromosome, computes K_loco = (S_full - S_chr[c]) / (p - p_c),
    freeing S_chr[c] after each yield.

    Args:
        S_full_np: Full kinship numerator as numpy array (n_samples, n_samples).
        S_chr: Per-chromosome kinship contributions (JAX arrays).
        n_chr_filtered: Count of filtered SNPs per chromosome.
        n_filtered: Total number of filtered SNPs.
        K_loco_buf: Pre-allocated workspace (n_samples, n_samples) for K_loco.
            When provided, np.subtract(out=) avoids a temporary, then the
            result is copied before yielding so callers may freely
            materialise the iterator. When None, a new array is allocated
            per chromosome (legacy behavior).

    Yields:
        (chr_name, K_loco) pairs in biological chromosome order.

    Raises:
        ValueError: If all filtered SNPs are on a single chromosome.
    """
    # Safe to del during iteration: sorted() materializes keys into a list.
    for chr_name in sorted(S_chr.keys(), key=chr_sort_key):
        p_chr = n_chr_filtered[chr_name]
        p_loco = n_filtered - p_chr

        if p_loco == 0:
            raise ValueError(
                f"Cannot compute LOCO kinship: all {n_filtered} filtered SNPs "
                f"are on chromosome '{chr_name}'."
            )

        if K_loco_buf is not None:
            # In-place subtraction avoids a temporary array (LOCO-03).
            # Copy before yielding so callers that materialise the full
            # iterator (dict / list) get independent arrays.
            np.subtract(S_full_np, np.asarray(S_chr[chr_name]), out=K_loco_buf)
            K_loco_buf /= p_loco
            K_loco = K_loco_buf.copy()
        else:
            K_loco = (S_full_np - np.asarray(S_chr[chr_name])) / p_loco
        logger.debug(
            f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs retained"
        )
        del S_chr[chr_name]
        yield (chr_name, K_loco)


def _stream_s_full_and_chr(
    bed_path: Path,
    n_samples: int,
    n_snps: int,
    snp_indices: np.ndarray,
    chromosomes: np.ndarray,
    chr_subset: list[str],
    chunk_size: int,
    show_progress: bool,
    desc: str,
    S_full_accum: bool = True,
) -> tuple[jnp.ndarray | None, dict[str, jnp.ndarray]]:
    """Stream genotypes and accumulate S_full and/or per-chromosome S_chr.

    Args:
        bed_path: PLINK file prefix.
        n_samples: Number of samples.
        n_snps: Total SNPs in the BED file (for chunk iteration).
        snp_indices: Global indices of filtered SNPs (sorted).
        chromosomes: Chromosome label for every SNP in the BED file.
        chr_subset: Chromosomes to accumulate S_chr for in this pass.
        chunk_size: SNPs per disk chunk.
        show_progress: Show progress bar.
        desc: Progress bar description.
        S_full_accum: If True, also accumulate S_full. Set False for
            multi-pass batches after S_full is already computed.

    Returns:
        (S_full or None, dict of chr_name -> S_chr)
    """
    import jax.numpy as jnp

    S_full = (
        jnp.zeros((n_samples, n_samples), dtype=jnp.float64) if S_full_accum else None
    )
    chr_set = set(chr_subset)
    S_chr: dict[str, jnp.ndarray] = {
        c: jnp.zeros((n_samples, n_samples), dtype=jnp.float64) for c in chr_subset
    }

    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )
    if show_progress:
        chunk_iter = progress_iterator(chunk_iter, total=n_chunks, desc=desc)

    for chunk, file_start, file_end in chunk_iter:
        left = np.searchsorted(snp_indices, file_start, side="left")
        right = np.searchsorted(snp_indices, file_end, side="left")
        chunk_snp_global_indices = snp_indices[left:right]
        chunk_filtered_local = chunk_snp_global_indices - file_start

        if len(chunk_filtered_local) == 0:
            continue

        # Check which target chromosomes are in this chunk before allocating
        chunk_chrs = chromosomes[chunk_snp_global_indices]
        target_chrs_in_chunk = set(chunk_chrs) & chr_set

        # Skip centering when S_full isn't needed and no target chromosomes present
        if S_full is None and not target_chrs_in_chunk:
            continue

        X_chunk = jnp.array(chunk[:, chunk_filtered_local])
        X_centered = impute_and_center(X_chunk)

        if S_full is not None:
            S_full = S_full + jnp.matmul(X_centered, X_centered.T)
            S_full.block_until_ready()

        for chr_name in target_chrs_in_chunk:
            X_chr_part = X_centered[:, chunk_chrs == chr_name]
            S_chr[chr_name] = S_chr[chr_name] + jnp.matmul(X_chr_part, X_chr_part.T)
            S_chr[chr_name].block_until_ready()

    return S_full, S_chr


def compute_loco_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Compute LOCO kinship matrices from disk-streamed genotypes.

    Two-pass streaming approach that accumulates both S_full and per-chromosome
    S_chr matrices, then derives LOCO kinship via subtraction:
    K_loco_c = (S_full - S_chr[c]) / (p - p_c).

    Pass 1: Compute per-SNP statistics for filtering (MAF, missingness, variance).
    Pass 2: Stream filtered SNPs, accumulate S_full and S_chr.

    When all S_chr fit in memory, a single second pass accumulates everything.
    When memory is insufficient (e.g. 100k+ samples), chromosomes are batched
    across multiple passes — S_full is computed once in the first pass, then
    each subsequent pass accumulates S_chr for a batch of chromosomes.

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation.
        show_progress: If True (default), show progress bar during iteration.
        ksnps_indices: Pre-resolved column indices for -ksnps restriction, or None.

    Yields:
        Tuple of (chr_name, K_loco) where chr_name is the chromosome being
        excluded and K_loco is the LOCO kinship matrix (n_samples, n_samples).

    Raises:
        MemoryError: If check_memory=True and insufficient memory for even
            S_full + one S_chr.
        FileNotFoundError: If the PLINK .bed file does not exist.
        ValueError: If no SNPs pass filtering, or if all filtered SNPs are on
            a single chromosome.
    """
    from jamma.core import ensure_jax_configured

    ensure_jax_configured()

    start_time = time.perf_counter()

    # Get dimensions and chromosome metadata
    meta = get_plink_metadata(bed_path)
    n_samples = meta["n_samples"]
    n_snps = meta["n_snps"]
    chromosomes = meta["chromosome"]

    # Derive partitions from already-loaded metadata — avoids re-opening BED (LOCO-04)
    partitions = partitions_from_metadata(meta)
    unique_chrs = sorted(partitions.keys(), key=chr_sort_key)

    logger.info("Computing LOCO Kinship (streaming)")
    logger.info(f"  Individuals: {n_samples:,}")
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chromosomes: {len(unique_chrs)}")
    logger.info(f"  Chunk size: {chunk_size:,}")

    # === PASS 1: Compute per-SNP statistics for filtering ===
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks, desc="LOCO: SNP statistics"
        )

    for chunk, start, end in stats_iterator:
        chunk_miss_counts = np.sum(np.isnan(chunk), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chunk_means = np.nanmean(chunk, axis=0)
            chunk_vars = np.nanvar(chunk, axis=0)
        chunk_means = np.nan_to_num(chunk_means, nan=0.0)
        chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

        all_means[start:end] = chunk_means
        all_miss_counts[start:end] = chunk_miss_counts
        all_vars[start:end] = chunk_vars
        del chunk  # Free ~1.6GB per chunk at scale before next iteration

    # Compute filters
    miss_rates = all_miss_counts / n_samples
    del all_miss_counts
    allele_freqs = all_means / 2.0
    del all_means
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)
    is_polymorphic = all_vars > 0
    del all_vars
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic

    # Apply kinship SNP list restriction (if -ksnps provided)
    if ksnps_indices is not None:
        from jamma.core.snp_filter import apply_snp_list_mask

        apply_snp_list_mask(snp_mask, ksnps_indices, n_snps, "Kinship SNP list")

    n_filtered = int(np.sum(snp_mask))

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_snps}"
        )

    if n_filtered < n_snps:
        n_removed = n_snps - n_filtered
        logger.info(
            f"LOCO kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    # Build SNP-to-chromosome mapping for filtered SNPs
    snp_indices = np.where(snp_mask)[0]

    # Map each filtered SNP index to its chromosome
    chr_for_filtered = chromosomes[snp_indices]

    # Count filtered SNPs per chromosome
    n_chr_filtered: dict[str, int] = {
        chr_name: int(np.sum(chr_for_filtered == chr_name)) for chr_name in unique_chrs
    }
    chrs_with_snps = [c for c in unique_chrs if n_chr_filtered.get(c, 0) > 0]
    chrs_without_snps = [c for c in unique_chrs if n_chr_filtered.get(c, 0) == 0]
    if chrs_without_snps:
        logger.warning(
            f"{len(chrs_without_snps)} chromosome(s) have 0 ksnps after filtering: "
            f"{chrs_without_snps}. LOCO will use full kinship for these "
            f"(nothing to leave out)."
        )
    n_chr_with_snps = len(chrs_with_snps)

    # Determine memory strategy: single-pass vs multi-pass batching
    matrix_gb = n_samples**2 * 8 / 1e9
    chunk_buffer_gb = n_samples * chunk_size * 8 / 1e9
    single_pass_gb = matrix_gb * (1 + n_chr_with_snps) + chunk_buffer_gb
    available_gb = psutil.virtual_memory().available / 1e9
    # S_full (numpy, persistent) + batch of S_chr (JAX) + chunk buffer
    min_required_gb = matrix_gb * 2 + chunk_buffer_gb  # S_full + 1 S_chr + buffer

    if check_memory and min_required_gb > available_gb * 0.9:
        raise MemoryError(
            f"Insufficient memory for LOCO kinship: need at least "
            f"{min_required_gb:.1f}GB for S_full + one S_chr, "
            f"available {available_gb:.1f}GB"
        )

    single_pass = single_pass_gb <= available_gb * 0.9

    if single_pass:
        # === SINGLE-PASS: accumulate S_full and all S_chr together ===
        if single_pass_gb > 10:
            logger.info(
                f"LOCO streaming: single-pass ({single_pass_gb:.1f}GB for "
                f"{n_chr_with_snps} chromosomes)"
            )

        S_full_jax, S_chr = _stream_s_full_and_chr(
            bed_path,
            n_samples,
            n_snps,
            snp_indices,
            chromosomes,
            chrs_with_snps,
            chunk_size,
            show_progress,
            desc="LOCO: kinship accumulation",
        )

        S_full_np = np.array(S_full_jax)
        del S_full_jax
        gc.collect()

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"LOCO streaming accumulation complete in {elapsed:.2f}s, "
            f"computing {len(S_chr)} LOCO matrices"
        )

        K_loco_buf = np.empty_like(S_full_np)
        yield from _yield_loco_matrices(
            S_full_np, S_chr, n_chr_filtered, n_filtered, K_loco_buf
        )
        yield from _yield_full_kinship_fallback(
            S_full_np, chrs_without_snps, n_filtered
        )
    else:
        # === MULTI-PASS: batch chromosomes across disk passes ===
        # First pass holds JAX S_full + batch S_chr + chunk buffer; after
        # conversion, numpy S_full replaces JAX S_full (briefly both exist).
        # Reserve 2x matrix_gb for that transition.
        usable_gb = available_gb * 0.9 - 2 * matrix_gb - chunk_buffer_gb
        batch_size = max(1, int(usable_gb / matrix_gb))

        n_batches = (n_chr_with_snps + batch_size - 1) // batch_size
        logger.warning(
            f"LOCO streaming: multi-pass mode ({n_batches} passes, "
            f"{batch_size} chromosomes/pass). Single-pass would need "
            f"{single_pass_gb:.1f}GB, available {available_gb:.1f}GB."
        )

        # First batch: compute S_full + first batch of S_chr
        first_batch = chrs_with_snps[:batch_size]
        S_full_jax, S_chr = _stream_s_full_and_chr(
            bed_path,
            n_samples,
            n_snps,
            snp_indices,
            chromosomes,
            first_batch,
            chunk_size,
            show_progress,
            desc=f"LOCO: pass 1/{n_batches} (S_full + {len(first_batch)} chr)",
        )

        S_full_np = np.array(S_full_jax)
        del S_full_jax
        gc.collect()

        K_loco_buf = np.empty_like(S_full_np)
        yield from _yield_loco_matrices(
            S_full_np, S_chr, n_chr_filtered, n_filtered, K_loco_buf
        )
        del S_chr
        gc.collect()

        # Subsequent batches: only accumulate S_chr (S_full already computed)
        for batch_idx in range(1, n_batches):
            batch_start = batch_idx * batch_size
            batch_chrs = chrs_with_snps[batch_start : batch_start + batch_size]

            _, S_chr = _stream_s_full_and_chr(
                bed_path,
                n_samples,
                n_snps,
                snp_indices,
                chromosomes,
                batch_chrs,
                chunk_size,
                show_progress,
                desc=(
                    f"LOCO: pass {batch_idx + 1}/{n_batches} ({len(batch_chrs)} chr)"
                ),
                S_full_accum=False,
            )

            yield from _yield_loco_matrices(
                S_full_np, S_chr, n_chr_filtered, n_filtered, K_loco_buf
            )
            del S_chr
            gc.collect()

        yield from _yield_full_kinship_fallback(
            S_full_np, chrs_without_snps, n_filtered
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"LOCO multi-pass complete in {elapsed:.2f}s, "
            f"{n_batches} passes over {n_chr_with_snps} chromosomes"
        )
