import blosum
import itertools
import math
import numpy as np
from pandas import DataFrame
from typing import Iterable, List, Optional, Set

from edit_penalty.edit_penalty_collection import EditPenaltyCollection
from edit_penalty.edit import (
    Position,
    Residue,
    JunctionEdit,
)
from edit_penalty.edit_penalty import EditPenalty


class EditPenaltyCollectionAnalyser:
    def __init__(
        self, tcr_edit_record_collection: EditPenaltyCollection
    ) -> None:
        self.edit_record_collection = tcr_edit_record_collection

    def get_summary_df(self) -> DataFrame:
        insertion_distances = self._get_insertion_distances_over_positions()
        deletion_distances = self._get_deletion_distances_over_positions()
        substitution_distances = self._get_substitution_distances_over_positions()

        df = DataFrame(index = [position.name for position in Position])

        df[["ins", "ins_std"]] = insertion_distances
        df[["del", "del_std"]] = deletion_distances
        df[["sub", "sub_std"]] = substitution_distances

        return df

    def get_average_distance_over_central_edits(self) -> float:
        central_insertions = [edit for edit in self._get_all_junction_aa_insertions() if edit.is_central]
        central_deletions = [edit for edit in self._get_all_junction_aa_deletions() if edit.is_central]
        central_substitutions = [edit for edit in self._get_all_junction_aa_substitutions() if edit.is_central]
        
        ins_dist, _ = self._get_mean_std_distance_from_specified_edits(central_insertions)
        del_dist, _ = self._get_mean_std_distance_from_specified_edits(central_deletions)
        sub_dist, _ = self._get_mean_std_distance_from_specified_edits(central_substitutions)

        return (ins_dist + del_dist + sub_dist) / 3

    def _get_insertion_distances_over_positions(self) -> List[List[float]]:
        all_insertions = self._get_all_junction_aa_insertions()
        insertions_over_positions = [
            edits.intersection(all_insertions)
            for edits in self._get_all_junction_edits_over_positions()
        ]
        distances_over_positions = [
            self._get_mean_std_distance_from_specified_edits(edits)
            for edits in insertions_over_positions
        ]

        return distances_over_positions

    def _get_deletion_distances_over_positions(self) -> List[List[float]]:
        all_deletions = self._get_all_junction_aa_deletions()
        deletions_over_positions = [
            edits.intersection(all_deletions)
            for edits in self._get_all_junction_edits_over_positions()
        ]
        distances_over_positions = [
            self._get_mean_std_distance_from_specified_edits(edits)
            for edits in deletions_over_positions
        ]

        return distances_over_positions

    def _get_substitution_distances_over_positions(self) -> List[List[float]]:
        all_substitutions = self._get_all_junction_aa_substitutions()
        substitutions_over_positions = [
            edits.intersection(all_substitutions)
            for edits in self._get_all_junction_edits_over_positions()
        ]
        distances_over_positions = [
            self._get_mean_std_distance_from_specified_edits(edits)
            for edits in substitutions_over_positions
        ]

        return distances_over_positions

    def _get_all_junction_aa_insertions(self) -> Set[JunctionEdit]:
        return {
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if edit.is_from(Residue.null)
        }

    def _get_all_junction_aa_deletions(self) -> Set[JunctionEdit]:
        return {
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if edit.is_to(Residue.null)
        }

    def _get_all_junction_aa_substitutions(self) -> Set[JunctionEdit]:
        return {
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if not (edit.is_from(Residue.null) or edit.is_to(Residue.null))
        }

    def _get_all_junction_edits_over_positions(self) -> List[Set[JunctionEdit]]:
        return [self._get_all_edits_at_position(position) for position in Position]

    def _get_all_edits_at_position(self, position: Position) -> Set[JunctionEdit]:
        return {
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if edit.is_at_position(position)
        }

    def _get_mean_std_distance_from_specified_edits(
        self, edits: Iterable[JunctionEdit]
    ) -> tuple:
        edit_records = [
            self.edit_record_collection.edit_record_dictionary[edit] for edit in edits
        ]
        mean_distances = [edit_record.average_distance for edit_record in edit_records]
        var_distances = [edit_record.var_distance for edit_record in edit_records]
        num_samples = [
            edit_record.num_distances_sampled for edit_record in edit_records
        ]
        weights = [num / sum(num_samples) for num in num_samples]

        mean_distance = sum(
            [
                weight * mean_distance
                for (weight, mean_distance) in zip(weights, mean_distances)
            ]
        )
        var_distance = sum(
            [weight * var for (weight, var) in zip(weights, var_distances)]
        )

        return (mean_distance, math.sqrt(var_distance))

    def _get_sample_weights(
        self, edit_records: Iterable[EditPenalty]
    ) -> List[float]:
        num_distances_sampled_per_edit_record = [
            edit_record.num_distances_sampled for edit_record in edit_records
        ]
        total_num_distances_sampled = sum(num_distances_sampled_per_edit_record)
        return [
            num_distances_sampled / total_num_distances_sampled
            for num_distances_sampled in num_distances_sampled_per_edit_record
        ]

    def _get_average_distance_for_central_substitution(
        self, from_residue: Residue, to_residue: Residue
    ) -> Optional[float]:
        relevant_edits = [
            edit
            for edit in self.edit_record_collection.edit_record_dictionary
            if edit.is_from(from_residue) and edit.is_to(to_residue) and edit.is_central
        ]

        edit_records_per_position = [
            self.edit_record_collection.edit_record_dictionary[edit]
            for edit in relevant_edits
        ]
        edit_records_with_data = [
            edit_record
            for edit_record in edit_records_per_position
            if edit_record.num_distances_sampled > 0
        ]
        distance_per_available_position = [
            edit_record.average_distance for edit_record in edit_records_with_data
        ]

        if len(distance_per_available_position) == 0:
            return None

        return sum(distance_per_available_position) / len(
            distance_per_available_position
        )
