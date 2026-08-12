"""Lossless global-index partitions for protocol-specific population splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class IndexPartition:
    """One ordered subset of a population's stable global sheet indices."""

    name: str
    global_indices: tuple[int, ...]
    population_size: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("partition name cannot be empty")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if len(set(self.global_indices)) != len(self.global_indices):
            raise ValueError("partition global indices must be unique")
        if any(index < 0 or index >= self.population_size for index in self.global_indices):
            raise ValueError("partition index outside population")

    @property
    def global_to_local(self) -> np.ndarray:
        result = np.full(self.population_size, -1, dtype=int)
        result[np.asarray(self.global_indices, dtype=int)] = np.arange(len(self.global_indices))
        return result


@dataclass(frozen=True, slots=True)
class PartitionedEdges:
    source_partition: str
    target_partition: str
    source_local: np.ndarray
    target_local: np.ndarray
    spatial_factor: np.ndarray
    source_global: np.ndarray
    target_global: np.ndarray


@dataclass(slots=True)
class PartitionedProjection:
    """Global-index facade over multiple Brian2 synapse partition blocks."""

    blocks: tuple[Any, ...]
    source_global: tuple[np.ndarray, ...]
    target_global: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("partitioned projection requires at least one block")
        if not (len(self.blocks) == len(self.source_global) == len(self.target_global)):
            raise ValueError("projection block metadata lengths must match")
        for block, source, target in zip(
            self.blocks, self.source_global, self.target_global, strict=True
        ):
            if len(block) != len(source) or len(block) != len(target):
                raise ValueError("global edge metadata must match each block length")

    def __len__(self) -> int:
        return sum(len(block) for block in self.blocks)

    @property
    def i(self) -> np.ndarray:
        return np.concatenate(self.source_global)

    @property
    def j(self) -> np.ndarray:
        return np.concatenate(self.target_global)

    def read(self, variable: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(block, variable)[:]) for block in self.blocks]
        )

    def write(self, variable: str, values: np.ndarray | float) -> None:
        if np.isscalar(values):
            for block in self.blocks:
                setattr(block, variable, values)
            return
        array = np.asarray(values)
        if array.shape != (len(self),):
            raise ValueError(f"expected {len(self)} values for {variable}, got {array.shape}")
        offset = 0
        for block in self.blocks:
            count = len(block)
            setattr(block, variable, array[offset : offset + count])
            offset += count


@dataclass(slots=True)
class PartitionedPopulation:
    """Global-index facade over disjoint physical neuron groups."""

    parts: tuple[tuple[IndexPartition, Any], ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("partitioned population requires at least one part")
        sizes = {partition.population_size for partition, _ in self.parts}
        covered = [index for partition, _ in self.parts for index in partition.global_indices]
        if len(sizes) != 1 or sorted(covered) != list(range(next(iter(sizes)))):
            raise ValueError("population partitions must form an exact global cover")

    @property
    def population_size(self) -> int:
        return self.parts[0][0].population_size

    @property
    def compartments(self) -> tuple[str, ...]:
        return self.parts[0][1].compartments

    def set_external_input(self, record_id: str, channel: str, value: float, indices=slice(None)) -> None:
        selected = None if isinstance(indices, slice) else {int(index) for index in indices}
        for partition, population in self.parts:
            if selected is None:
                local = slice(None)
            else:
                local = [i for i, global_index in enumerate(partition.global_indices) if global_index in selected]
                if not local:
                    continue
            population.set_external_input(record_id, channel, value, indices=local)

    def set_convergent_external_input(self, record_id: str, channel: str, source_values, indices=slice(None)) -> None:
        for _, population in self.parts:
            population.set_convergent_external_input(record_id, channel, source_values, indices=indices)


def population_parts(population: Any) -> tuple[tuple[IndexPartition, Any], ...]:
    """Return physical groups and stable global-index maps for a population."""

    if isinstance(population, PartitionedPopulation):
        return population.parts
    size = int(population.group.N)
    return ((IndexPartition("all", tuple(range(size)), size), population),)


def complementary_partitions(
    *,
    population_size: int,
    selected_indices: tuple[int, ...],
    selected_name: str = "selected",
    remainder_name: str = "remainder",
) -> tuple[IndexPartition, IndexPartition]:
    selected = IndexPartition(selected_name, selected_indices, population_size)
    selected_set = set(selected_indices)
    remainder = IndexPartition(
        remainder_name,
        tuple(index for index in range(population_size) if index not in selected_set),
        population_size,
    )
    return selected, remainder


def partition_edges(
    source_global: np.ndarray,
    target_global: np.ndarray,
    spatial_factor: np.ndarray,
    *,
    source_partitions: tuple[IndexPartition, ...],
    target_partitions: tuple[IndexPartition, ...],
) -> tuple[PartitionedEdges, ...]:
    """Split an edge list into lossless local-index partition blocks."""

    source = np.asarray(source_global, dtype=int)
    target = np.asarray(target_global, dtype=int)
    factor = np.asarray(spatial_factor, dtype=float)
    if source.shape != target.shape or source.shape != factor.shape:
        raise ValueError("source, target, and factor arrays must have the same shape")
    if not source_partitions or not target_partitions:
        raise ValueError("source and target partitions cannot be empty")
    if len({partition.population_size for partition in source_partitions}) != 1:
        raise ValueError("source partitions must share one population size")
    if len({partition.population_size for partition in target_partitions}) != 1:
        raise ValueError("target partitions must share one population size")

    assigned = np.zeros(source.shape, dtype=int)
    blocks: list[PartitionedEdges] = []
    for source_partition in source_partitions:
        source_map = source_partition.global_to_local
        source_local = source_map[source]
        for target_partition in target_partitions:
            target_map = target_partition.global_to_local
            target_local = target_map[target]
            mask = (source_local >= 0) & (target_local >= 0)
            if not np.any(mask):
                continue
            assigned[mask] += 1
            blocks.append(
                PartitionedEdges(
                    source_partition=source_partition.name,
                    target_partition=target_partition.name,
                    source_local=source_local[mask],
                    target_local=target_local[mask],
                    spatial_factor=factor[mask],
                    source_global=source[mask],
                    target_global=target[mask],
                )
            )
    if not np.all(assigned == 1):
        missing = int(np.count_nonzero(assigned == 0))
        duplicated = int(np.count_nonzero(assigned > 1))
        raise ValueError(
            f"partitions must cover every edge exactly once; missing={missing}, duplicated={duplicated}"
        )
    return tuple(blocks)
