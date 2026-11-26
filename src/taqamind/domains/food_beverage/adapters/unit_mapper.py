"""Thin unit-conversion wrapper around Shared Core DeterministicColumnMapper (P1.3)."""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pandas as pd

from taqamind.core.contracts import DatasetContract
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    ColumnMapper,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetState,
    MappingProposal,
    PipelineContext,
    StageResult,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.domains.food_beverage.config import UNIT_CONVERSION_FACTORS


class UnitAwareColumnMapperAdapter(ColumnMapper):
    """Wraps Core DeterministicColumnMapper to apply physical unit conversion (AUD-001).

    Delegates mapping proposal and canonical column renaming to Core DeterministicColumnMapper,
    then normalizes any approved source units differing from canonical units.
    Candidate for migration into ``taqamind.core.mapping``.
    """

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        core_mapper: DeterministicColumnMapper | None = None,
    ) -> None:
        self._storage = storage
        self._core = core_mapper or DeterministicColumnMapper(storage)

    def propose_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        explicit_source_units: dict[str, str] | None = None,
    ) -> MappingProposal:
        proposal = self._core.propose_mapping(
            context=context,
            artifact=artifact,
            contract=contract,
        )
        if not explicit_source_units:
            return proposal

        updated_entries = []
        for entry in proposal.entries:
            if entry.source_column in explicit_source_units:
                updated_entries.append(
                    entry.model_copy(
                        update={"source_unit": explicit_source_units[entry.source_column]}
                    )
                )
            else:
                updated_entries.append(entry)
        return proposal.model_copy(update={"entries": tuple(updated_entries)})

    def apply_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        mapping: ApprovedMapping,
    ) -> StageResult:
        # Validate unit compatibility first before delegating
        conversions: dict[str, float] = {}
        for entry in mapping.proposal.entries:
            if entry.canonical_unit is not None and entry.source_unit != entry.canonical_unit:
                factor = UNIT_CONVERSION_FACTORS.get(
                    (entry.source_unit or "", entry.canonical_unit)
                )
                if factor is None:
                    raise ValueError(
                        f"INCOMPATIBLE_UNIT: Cannot convert '{entry.source_column}' "
                        f"from '{entry.source_unit}' to '{entry.canonical_unit}'."
                    )
                conversions[entry.canonical_column] = factor

        # Reset source_unit to canonical_unit for Core apply_mapping, then apply numeric factor
        if not conversions:
            return self._core.apply_mapping(
                context=context,
                artifact=artifact,
                contract=contract,
                mapping=mapping,
            )

        norm_entries = tuple(
            e.model_copy(update={"source_unit": e.canonical_unit})
            if e.canonical_column in conversions
            else e
            for e in mapping.proposal.entries
        )
        norm_mapping = mapping.model_copy(
            update={"proposal": mapping.proposal.model_copy(update={"entries": norm_entries})}
        )
        stage_res = self._core.apply_mapping(
            context=context,
            artifact=artifact,
            contract=contract,
            mapping=norm_mapping,
        )
        if not stage_res.is_successful or stage_res.output_artifact is None:
            return stage_res

        out_art = stage_res.output_artifact
        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=out_art.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True)

        for col, factor in conversions.items():
            if col in df.columns:
                df[col] = df[col].astype(float) * factor

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        new_sha = calculate_bytes_sha256(csv_bytes)
        new_art_id = uuid4()
        new_ref = ArtifactReference(
            artifact_id=new_art_id,
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=new_sha,
            size_bytes=len(csv_bytes),
        )
        self._storage.put(
            factory_id=context.factory_id,
            reference=new_ref,
            stream=BytesIO(csv_bytes),
        )
        col_schemas = tuple(
            DatasetColumnSchema(name=str(c), physical_dtype=str(df[c].dtype))
            for c in df.columns
        )
        new_meta = out_art.metadata.model_copy(
            update={
                "revision_id": out_art.metadata.revision_id,
                "state": DatasetState.MAPPED,
                "columns": col_schemas,
            }
        )
        return stage_res.model_copy(
            update={"output_artifact": DatasetArtifact(metadata=new_meta, reference=new_ref)}
        )
