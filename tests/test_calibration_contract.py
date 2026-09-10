import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from smart_robustness.models.currents import biexponential_peak_time_ms
from smart_robustness.validation.calibration import (
    TrnStageAResult,
    load_calibration_contract,
    runtime_conventions_for_candidate,
)

ROOT = Path(__file__).parents[1]
HISTORICAL_FIGURE10_RUNTIME_SHA256 = (
    "ebdf48f0138803ab50b1dec2ef87817b4c537e4da490a088b9df6bdcb9d119a1"
)
CONTRACT_PATH = ROOT / "configs/calibration/classic_uncertainty_space.yaml"
TRN_SURVIVOR_PATH = ROOT / "configs/calibration/trn_stage_a_survivor_v1.yaml"
NETWORK_CALIBRATION_PATH = (
    ROOT / "docs/validation-results/calibration-network-trn-survivor-121.yaml"
)
GAUSSIAN_VARIANCE_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_stage_a_survivor_gaussian_variance_v1.yaml"
)
FIGURE8_VOLTAGE_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure8-voltage-observable-audit-124.yaml"
)
FIGURE6_RELAY_BALANCE_PATH = ROOT / "docs/validation-results/figure6-relay-current-balance-125.yaml"
FIGURE6_RELAY_SCREEN_PATH = ROOT / "docs/validation-results/figure6-relay-survivor-screen-126.yaml"
FIGURE6_RELAY_EQUILIBRATION_PATH = (
    ROOT / "docs/validation-results/figure6-relay-equilibration-127.yaml"
)
FIGURE6_SOURCE_HYBRID_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_relay_source_hybrid_v1.yaml"
)
FIGURE6_SOURCE_HYBRID_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-relay-source-hybrid-128.yaml"
)
FIGURE6_RELAY_AXIAL_HYBRID_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_relay_axial_source_hybrid_v1.yaml"
)
FIGURE6_RELAY_AXIAL_HYBRID_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-relay-axial-source-hybrid-129.yaml"
)
FIGURE6_RELAY_AXIAL_DECOMPOSITION_PATH = (
    ROOT / "docs/validation-results/figure6-relay-axial-map-decomposition-130.yaml"
)
FIGURE6_LEADING_ALTERNATIVES_PATH = (
    ROOT / "docs/validation-results/figure6-leading-source-alternatives-131.yaml"
)
FIGURE6_LEARNING_PHASE_PATH = (
    ROOT / "docs/validation-results/figure6-top-down-learning-phase-132.yaml"
)
FIGURE6_PROJECTION_BOUNDS_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_projection_level_learning_bounds_v1.yaml"
)
FIGURE6_PROJECTION_BOUNDS_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-projection-level-learning-bounds-133.yaml"
)
FIGURE6_PROJECTION_D_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_projection_depression_scale_v1.yaml"
)
FIGURE6_PROJECTION_D_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-projection-depression-scale-134.yaml"
)
FIGURE6_VOLLEY_DECOMPOSITION_PATH = (
    ROOT / "docs/validation-results/figure6-teaching-volley-decomposition-135.yaml"
)
FIGURE6_POPULATION_AXIAL_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_population_resolved_axial_v1.yaml"
)
FIGURE6_POPULATION_AXIAL_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-population-resolved-axial-136.yaml"
)
FIGURE6_POPULATION_AXIAL_AMPLITUDE_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure6-population-axial-amplitude-audit-140.yaml"
)
FIGURE6_POPULATION_AXIAL_LEARNING_PHASE_PATH = (
    ROOT / "docs/validation-results/figure6-population-axial-learning-phase-141.yaml"
)
FIGURE6_LEARNING_THRESHOLD_ASSESSMENT_PATH = (
    ROOT / "docs/validation-results/figure6-learning-threshold-coordinate-assessment-145.yaml"
)
FIGURE6_LEARNING_RULE_ASSESSMENT_PATH = (
    ROOT / "docs/validation-results/figure6-learning-rule-coordinate-assessment-148.yaml"
)
FIGURE6_DUAL_AND_LEAK_PHASE_PATH = (
    ROOT / "docs/validation-results/figure6-dual-and-leak-learning-phase-149.yaml"
)
FIGURE6_RELAY_WAVEFORM_PATH = (
    ROOT / "docs/validation-results/figure6-relay-waveform-nak-audit-150.yaml"
)
FIGURE6_UPWARD_TIMESTAMP_PATH = (
    ROOT / "docs/validation-results/figure6-learning-timestamp-upward-151.yaml"
)
FIGURE6_LEAK_PLUS30_PATH = (
    ROOT / "docs/validation-results/figure6-learning-coordinate-leak-plus30-152.yaml"
)
FIGURE7_POPULATION_AXIAL_RESULT_PATH = (
    ROOT / "docs/validation-results/calibration-figure6-figure7-population-axial-137.yaml"
)
FIGURE7_THALAMOCORTICAL_AXIAL_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_thalamocortical_axial_v1.yaml"
)
FIGURE6_THALAMOCORTICAL_AXIAL_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-thalamocortical-axial-138.yaml"
)
FIGURE7_POPULATION_TRN_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_population_resolved_trn_v1.yaml"
)
FIGURE6_POPULATION_TRN_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-population-resolved-trn-139.yaml"
)
FIGURE7_TRN_POTASSIUM_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure7-trn-potassium-source-screen-166.yaml"
)
FIGURE7_TRN_CALCIUM_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure7-trn-calcium-source-screen-167.yaml"
)
FIGURE7_TRN_CALCIUM_PROFILE_PATH = ROOT / "configs/calibration/figure7_trn_calcium_reversal_v1.yaml"
FIGURE7_TRN_CALCIUM_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-trn-calcium-reversal-simultaneous-pair-168.yaml"
)
FIGURE7_TRN_DENDRITIC_CALCIUM_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure7-trn-dendritic-calcium-source-screen-169.yaml"
)
FIGURE7_TRN_DENSITY_GRID_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_dendritic_density_behavior_grid_v1.yaml"
)
FIGURE7_TRN_DENSITY_CUE_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-trn-density-cue-grid-170.yaml"
)
FIGURE7_TRN_DENSITY_MATCH_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-trn-density-match-grid-171.yaml"
)
FIGURE7_TRN_SOURCE_TOPOLOGY_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure7-trn-source-topology-audit-172.yaml"
)
FIGURE7_TRN_AXIAL_GRID_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_soma_proximal_axial_behavior_grid_v1.yaml"
)
FIGURE7_TRN_AXIAL_CUE_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-trn-axial-cue-grid-173.yaml"
)
FIGURE7_TRN_AXIAL_MATCH_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-trn-axial-match-grid-174.yaml"
)
FIGURE7_TRN_EVENT_OFFSET_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_event_offset_behavior_grid_v1.yaml"
)
FIGURE7_TRN_EVENT_OFFSET_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-offset-cue-grid-175.yaml"
)
FIGURE7_TRN_EVENT_OFFSET_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-offset-match-grid-176.yaml"
)
FIGURE7_TRN_DENSITY_EVENT_OFFSET_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_density_event_offset_cross_v1.yaml"
)
FIGURE7_TRN_DENSITY_EVENT_OFFSET_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-trn-density-event-offset-cue-grid-177.yaml"
)
FIGURE7_TRN_EVENT_BLEND_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_soma_proximal_event_blend_v1.yaml"
)
FIGURE7_TRN_EVENT_BLEND_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-blend-cue-grid-178.yaml"
)
FIGURE7_TRN_EVENT_BLEND_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-blend-match-grid-179.yaml"
)
FIGURE7_TRN_EVENT_BLEND_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-blend-pair-180.yaml"
)
FIGURE7_PROTOCOL_GATE_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure7-protocol-gate-audit-181.yaml"
)
FIGURE7_EVENT_BLEND_CURRENT_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_event_blend_top_down_grid_v1.yaml"
)
FIGURE7_EVENT_BLEND_CURRENT_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-event-blend-current-cue-grid-182.yaml"
)
FIGURE7_EVENT_BLEND_CURRENT_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-event-blend-current-match-grid-183.yaml"
)
FIGURE7_EVENT_BLEND_CURRENT_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-event-blend-current-pair-300ms-184.yaml"
)
FIGURE7_NONSPECIFIC_EVENT_BLEND_PROFILE_PATH = (
    ROOT / "configs/calibration/nonspecific_soma_proximal_event_blend_v1.yaml"
)
FIGURE7_NONSPECIFIC_EVENT_BLEND_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-nonspecific-blend-cue-grid-185.yaml"
)
FIGURE7_NONSPECIFIC_EVENT_BLEND_MISMATCH_PATH = (
    ROOT / "docs/validation-results/figure7-nonspecific-blend-mismatch-grid-186.yaml"
)
FIGURE7_NONSPECIFIC_EVENT_BLEND_COMPARISON_PATH = (
    ROOT / "docs/validation-results/figure7-nonspecific-blend-match-comparison-187.yaml"
)
FIGURE7_FEEDBACK_ARRIVAL_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure7-feedback-arrival-order-audit-188.yaml"
)
FIGURE7_FEEDBACK_ARRIVAL_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-feedback-arrival-alignment-grid-189.yaml"
)
FIGURE7_TRN_BLEND_ARRIVAL_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-blend-arrival-match-grid-190.yaml"
)
FIGURE7_TRN_BLEND_ARRIVAL_MISMATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-blend-arrival-mismatch-191.yaml"
)
FIGURE7_INHIBITORY_ARRIVAL_PATH = (
    ROOT / "docs/validation-results/figure7-inhibitory-arrival-alignment-192.yaml"
)
FIGURE6_SOURCE_STRENGTH_REASSESSMENT_PATH = (
    ROOT / "docs/validation-results/figure6-source-strength-reassessment-193.yaml"
)
FIGURE7_TRN_DETECTOR_CYCLE_PATH = (
    ROOT / "docs/validation-results/figure7-trn-detector-cycle-diagnostic-194.yaml"
)
FIGURE7_SAME_NETWORK_DETECTOR_PATH = (
    ROOT / "docs/validation-results/figure7-same-network-detector-diagnostic-195.yaml"
)
FIGURE6_RELAY_DETECTOR_CYCLE_PATH = (
    ROOT / "docs/validation-results/figure6-relay-detector-cycle-audit-196.yaml"
)
FIGURE7_TRN_SOMA_POTASSIUM_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_soma_potassium_behavior_grid_v1.yaml"
)
FIGURE7_TRN_SOMA_POTASSIUM_STAGE1_PATH = (
    ROOT / "docs/validation-results/figure7-trn-soma-potassium-stage1-197.yaml"
)
LEGACY_SOURCE_RECOVERY_FOLLOWUP_PATH = (
    ROOT / "docs/validation-results/legacy-source-recovery-followup-198.yaml"
)
FIGURE7_TRN_SOMA_SODIUM_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_soma_sodium_behavior_grid_v1.yaml"
)
FIGURE7_TRN_SOMA_SODIUM_STAGE1_PATH = (
    ROOT / "docs/validation-results/figure7-trn-soma-sodium-stage1-199.yaml"
)
FIGURE7_TRN_DETECTOR_HYSTERESIS_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_detector_hysteresis_behavior_grid_v1.yaml"
)
FIGURE7_TRN_DETECTOR_HYSTERESIS_STAGE1_PATH = (
    ROOT / "docs/validation-results/figure7-trn-detector-hysteresis-stage1-200.yaml"
)
FIGURE6_TRN_DETECTOR_HYSTERESIS_PREREQUISITE_PATH = (
    ROOT / "docs/validation-results/figure6-trn-detector-hysteresis-prerequisite-201.yaml"
)
FIGURE6_TRN_GABA_TRANSFER_GRID_PATH = (
    ROOT / "docs/validation-results/figure6-trn-gaba-transfer-grid-202.yaml"
)
FIGURE7_TRN_GABA_TRANSFER_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-gaba-transfer-match-203.yaml"
)
FIGURE7_TRN_GABA_TRANSFER_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-trn-gaba-transfer-pair-204.yaml"
)
FIGURE6_TRN_GABA_COMPARTMENT_SOURCE_PATH = (
    ROOT / "docs/validation-results/figure6-trn-gaba-compartment-source-endpoint-205.yaml"
)
FIGURE6_TRN_GABA_COMPARTMENT_INTERMEDIATE_PATH = (
    ROOT / "docs/validation-results/figure6-trn-gaba-compartment-intermediate-grid-206.yaml"
)
FIGURE7_TRN_GABA_COMPARTMENT_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-gaba-compartment-match-207.yaml"
)
FIGURE7_TRN_GABA_COMPARTMENT_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-trn-gaba-compartment-pair-208.yaml"
)
FIGURE6_PROJECTION022_SOURCE_RESOLUTION_PATH = (
    ROOT / "docs/validation-results/figure6-projection022-source-resolution-209.yaml"
)
FIGURE7_PROJECTION022_SOURCE_RESOLUTION_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-projection022-source-resolution-match-210.yaml"
)
FIGURE7_PROJECTION022_SOURCE_RESOLUTION_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-projection022-source-resolution-pair-211.yaml"
)
FIGURE6_PROJECTION022_DISTAL002_PATH = (
    ROOT / "docs/validation-results/figure6-projection022-distal002-212.yaml"
)
FIGURE7_PROJECTION022_DISTAL002_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-projection022-distal002-match-213.yaml"
)
FIGURE7_PROJECTION022_DISTAL002_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-projection022-distal002-pair-214.yaml"
)
FIGURE7_PROJECTION022_DISTAL002_FRESH_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-projection022-distal002-fresh-match-215.yaml"
)
FIGURE7_PROJECTION022_DISTAL002_FRESH_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-projection022-distal002-fresh-pair-216.yaml"
)
FIGURE6_PROJECTION022_DISTAL003_PATH = (
    ROOT / "docs/validation-results/figure6-projection022-distal003-217.yaml"
)
FIGURE7_PROJECTION022_DISTAL003_FRESH_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-projection022-distal003-fresh-match-218.yaml"
)
FIGURE7_PROJECTION022_DISTAL003_FRESH_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-projection022-distal003-fresh-pair-219.yaml"
)
KINNESS_RING_SOURCE_RECOVERY_PATH = (
    ROOT / "docs/validation-results/kinness-ring-source-recovery-220.yaml"
)
RING_KERNEL_SENSITIVITY_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/ring-kernel-sensitivity-registration-221.yaml"
)
RING_KERNEL_RADIAL_ANNULUS_PROFILE_PATH = (
    ROOT / "configs/calibration/ring_kernel_radial_annulus_figure6_v1.yaml"
)
FIGURE6_RING_KERNEL_RADIAL_ANNULUS_PATH = (
    ROOT / "docs/validation-results/figure6-ring-kernel-radial-annulus-222.yaml"
)
FIGURE7_RENDERED_TARGET_CORRECTION_PATH = (
    ROOT / "docs/validation-results/figure7-rendered-target-correction-223.yaml"
)
FIGURE7_TOP_DOWN_CURRENT_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top-down-current-reopen-registration-224.yaml"
)
FIGURE7_TOP_DOWN_CURRENT_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top_down_current_match_reopen_v1.yaml"
)
FIGURE7_TOP_DOWN_CURRENT_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top-down-current-match-reopen-225.yaml"
)
FIGURE7_TOP_DOWN_CURRENT_MISMATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top-down-current-800-mismatch-registration-226.yaml"
)
FIGURE7_TOP_DOWN_CURRENT_PAIR_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top_down_current_800_fresh_pair_v1.yaml"
)
FIGURE7_TOP_DOWN_CURRENT_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top-down-current-800-fresh-pair-227.yaml"
)
FIGURE7_CURRENT_TERMINATION_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-current-termination-registration-228.yaml"
)
FIGURE7_ONE_EVENT_CURRENT_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_one_event_current_match_v1.yaml"
)
FIGURE7_ONE_EVENT_CURRENT_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-one-event-current-match-229.yaml"
)
FIGURE7_ONE_EVENT_CURRENT_MISMATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-one-event-current-mismatch-registration-230.yaml"
)
FIGURE7_ONE_EVENT_CURRENT_PAIR_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_one_event_current_fresh_pair_v1.yaml"
)
FIGURE7_ONE_EVENT_CURRENT_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-one-event-current-fresh-pair-231.yaml"
)
FIGURE7_SELECTED_CATEGORY_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-selected-category-routing-registration-232.yaml"
)
FIGURE7_SELECTED_CATEGORY_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_selected_category_routing_match_v1.yaml"
)
FIGURE7_SELECTED_CATEGORY_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-selected-category-routing-match-233.yaml"
)
FIGURE7_SELECTED_CATEGORY_MISMATCH_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-selected-category-routing-mismatch-registration-234.yaml"
)
FIGURE7_SELECTED_CATEGORY_PAIR_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_selected_category_routing_fresh_pair_v1.yaml"
)
FIGURE7_SELECTED_CATEGORY_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-selected-category-routing-fresh-pair-235.yaml"
)
SPIKE_EVENT_EQUATION_VISUAL_AUDIT_PATH = (
    ROOT / "docs/validation-results/spike-event-equation-visual-audit-236.yaml"
)
FIGURE7_RECEPTOR_ALIGNMENT_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_arrival_aligned_match_v1.yaml"
)
FIGURE7_RECEPTOR_ALIGNMENT_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-arrival-alignment-registration-237.yaml"
)
FIGURE7_RECEPTOR_ALIGNMENT_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-arrival-aligned-match-238.yaml"
)
FIGURE7_ALIGNED_HEADROOM_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_aligned_on_center_headroom_v1.yaml"
)
FIGURE7_ALIGNED_HEADROOM_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-on-center-headroom-registration-239.yaml"
)
FIGURE7_ALIGNED_HEADROOM_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-on-center-headroom-screen-240.yaml"
)
FIGURE7_RECEPTOR_PEAK_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_peak_aligned_headroom_match_v1.yaml"
)
FIGURE7_RECEPTOR_PEAK_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-aligned-headroom-registration-296.yaml"
)
FIGURE7_RECEPTOR_PEAK_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-aligned-headroom-match-297.yaml"
)
FIGURE7_RECEPTOR_PEAK_VERIFICATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_peak_aligned_headroom_verification_v1.yaml"
)
FIGURE7_RECEPTOR_PEAK_VERIFICATION_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-receptor-peak-aligned-verification-registration-298.yaml"
)
FIGURE7_RECEPTOR_PEAK_VERIFICATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-aligned-verification-299.yaml"
)
FIGURE7_RECEPTOR_PEAK_MISMATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_peak_aligned_mismatch_v1.yaml"
)
FIGURE7_RECEPTOR_PEAK_MISMATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-aligned-mismatch-registration-300.yaml"
)
FIGURE7_RECEPTOR_PEAK_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-aligned-pair-301.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_peak_gaba_gain_match_v1.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-gaba-gain-registration-302.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-gaba-gain-match-303.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_VERIFICATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_peak_gaba_gain_verification_v1.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_VERIFICATION_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-receptor-peak-gaba-gain-verification-registration-304.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_VERIFICATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-gaba-gain-verification-305.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_MISMATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_peak_gaba_gain_mismatch_v1.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_MISMATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-gaba-gain-mismatch-registration-306.yaml"
)
FIGURE7_RECEPTOR_PEAK_GABA_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-gaba-gain-pair-307.yaml"
)
FIGURE7_RECEPTOR_PEAK_ANNULUS_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_peak_radial_annulus_match_v1.yaml"
)
FIGURE7_RECEPTOR_PEAK_ANNULUS_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-radial-annulus-registration-308.yaml"
)
FIGURE7_RECEPTOR_PEAK_ANNULUS_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-radial-annulus-match-309.yaml"
)
FIGURE7_RECEPTOR_PEAK_ADJACENT_ANNULUS_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_receptor_peak_adjacent_annulus_match_v1.yaml"
)
FIGURE7_RECEPTOR_PEAK_ADJACENT_ANNULUS_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-adjacent-annulus-registration-310.yaml"
)
FIGURE7_RECEPTOR_PEAK_ADJACENT_ANNULUS_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-receptor-peak-adjacent-annulus-match-311.yaml"
)
FIGURE7_LEARNED_COMPARATOR_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_learned_comparator_floor_match_v1.yaml"
)
FIGURE7_LEARNED_COMPARATOR_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-learned-comparator-floor-registration-312.yaml"
)
FIGURE7_LEARNED_COMPARATOR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-learned-comparator-floor-match-313.yaml"
)
FIGURE7_HALF_MAX_COMPARATOR_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_half_max_comparator_match_v1.yaml"
)
FIGURE7_HALF_MAX_COMPARATOR_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-half-max-comparator-registration-314.yaml"
)
FIGURE7_HALF_MAX_COMPARATOR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-half-max-comparator-match-315.yaml"
)
FIGURE7_TOP5_COMPARATOR_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_comparator_match_v1.yaml"
)
FIGURE7_TOP5_COMPARATOR_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-comparator-registration-316.yaml"
)
FIGURE7_TOP5_COMPARATOR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-comparator-match-317.yaml"
)
FIGURE7_TOP5_COMPARATOR_VERIFICATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_comparator_verification_v1.yaml"
)
FIGURE7_TOP5_COMPARATOR_VERIFICATION_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-comparator-verification-registration-318.yaml"
)
FIGURE7_TOP5_COMPARATOR_VERIFICATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-comparator-verification-319.yaml"
)
FIGURE7_TOP5_COMPARATOR_MISMATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_comparator_mismatch_v1.yaml"
)
FIGURE7_TOP5_COMPARATOR_MISMATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-comparator-mismatch-registration-320.yaml"
)
FIGURE7_TOP5_COMPARATOR_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-comparator-pair-321.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_BLEND_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_blend_match_v1.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_BLEND_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-blend-registration-322.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_BLEND_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-blend-match-323.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_SPLIT_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_split_detector_match_v1.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_SPLIT_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-split-registration-324.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_SPLIT_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-split-match-325.yaml"
)
FIGURE6_NONSPECIFIC_DISTAL_GABA_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_nonspecific_distal_gaba_supplement_v1.yaml"
)
FIGURE6_NONSPECIFIC_DISTAL_GABA_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure6-nonspecific-distal-gaba-source-registration-326.yaml"
)
FIGURE6_NONSPECIFIC_DISTAL_GABA_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-nonspecific-distal-gaba-source-327.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_gaba_transfer_match_v1.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-gaba-transfer-registration-328.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_SUPERSEDED_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-gaba-transfer-match-329.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-gaba-transfer-match-330.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_VERIFICATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_gaba_transfer_verification_v1.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_VERIFICATION_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-gaba-transfer-verification-registration-331.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_VERIFICATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-gaba-transfer-verification-332.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_MISMATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_gaba_transfer_mismatch_v1.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_MISMATCH_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-gaba-transfer-mismatch-registration-333.yaml"
)
FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-gaba-transfer-pair-334.yaml"
)
FIGURE6_METHODS_GLOBAL_CALCIUM_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_methods_global_calcium_v1.yaml"
)
FIGURE6_METHODS_GLOBAL_CALCIUM_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure6-methods-global-calcium-registration-335.yaml"
)
FIGURE6_METHODS_GLOBAL_CALCIUM_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-methods-global-calcium-336.yaml"
)
FIGURE7_NONSPECIFIC_VOLTAGE_AUDIT_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_voltage_peak_audit_v1.yaml"
)
FIGURE7_NONSPECIFIC_VOLTAGE_AUDIT_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-voltage-peak-audit-registration-337.yaml"
)
FIGURE7_NONSPECIFIC_VOLTAGE_AUDIT_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-voltage-peak-audit-338.yaml"
)
FIGURE7_NONSPECIFIC_PEAK_CURRENT_AUDIT_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_peak_current_audit_v1.yaml"
)
FIGURE7_NONSPECIFIC_PEAK_CURRENT_AUDIT_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-peak-current-audit-registration-365.yaml"
)
FIGURE7_NONSPECIFIC_PEAK_CURRENT_AUDIT_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-peak-current-audit-366.yaml"
)
FIGURE7_TOP5_TRN_VOLLEY_AUDIT_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_trn_volley_audit_v1.yaml"
)
FIGURE7_TOP5_TRN_VOLLEY_AUDIT_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-trn-volley-audit-registration-367.yaml"
)
FIGURE7_TOP5_TRN_VOLLEY_AUDIT_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-trn-volley-audit-368.yaml"
)
FIGURE7_TOP5_ARRIVAL_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_arrival_aligned_match_v1.yaml"
)
FIGURE7_TOP5_ARRIVAL_MATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-arrival-aligned-registration-339.yaml"
)
FIGURE7_TOP5_ARRIVAL_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-arrival-aligned-match-340.yaml"
)
FIGURE7_TOP5_ARRIVAL_VERIFICATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_arrival_aligned_verification_v1.yaml"
)
FIGURE7_TOP5_ARRIVAL_VERIFICATION_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-arrival-aligned-verification-registration-341.yaml"
)
FIGURE7_TOP5_ARRIVAL_VERIFICATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-arrival-aligned-verification-342.yaml"
)
FIGURE7_TOP5_ARRIVAL_MISMATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_arrival_aligned_mismatch_v1.yaml"
)
FIGURE7_TOP5_ARRIVAL_MISMATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-arrival-aligned-mismatch-registration-343.yaml"
)
FIGURE7_TOP5_ARRIVAL_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-arrival-aligned-pair-344.yaml"
)
FIGURE7_CORTICOINTRALAMINAR_ABLATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_corticointralaminar_ablation_v1.yaml"
)
FIGURE7_CORTICOINTRALAMINAR_ABLATION_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-corticointralaminar-ablation-registration-345.yaml"
)
FIGURE7_CORTICOINTRALAMINAR_ABLATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-corticointralaminar-ablation-346.yaml"
)
FIGURE7_NONSPECIFIC_GABA_COMPARTMENT_ABLATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_gaba_compartment_ablation_v1.yaml"
)
FIGURE7_NONSPECIFIC_GABA_COMPARTMENT_ABLATION_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-gaba-compartment-ablation-registration-347.yaml"
)
FIGURE7_NONSPECIFIC_GABA_COMPARTMENT_ABLATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-gaba-compartment-ablation-348.yaml"
)
FIGURE7_NONSPECIFIC_PAPER_TTYPE_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_paper_ttype_match_v1.yaml"
)
FIGURE7_NONSPECIFIC_PAPER_TTYPE_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-paper-ttype-registration-349.yaml"
)
FIGURE7_NONSPECIFIC_PAPER_TTYPE_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-paper-ttype-match-350.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_somatic_gaba_match_v1.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-somatic-gaba-registration-351.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-somatic-gaba-match-352.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_VERIFICATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_somatic_gaba_verification_v1.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_VERIFICATION_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-somatic-gaba-verification-registration-353.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_VERIFICATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-somatic-gaba-verification-354.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_MISMATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_somatic_gaba_mismatch_v1.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_MISMATCH_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-somatic-gaba-mismatch-registration-355.yaml"
)
FIGURE7_NONSPECIFIC_SOMATIC_GABA_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-somatic-gaba-pair-356.yaml"
)
FIGURE7_NONSPECIFIC_KINNESS_AXIAL_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_kinness_axial_match_v1.yaml"
)
FIGURE7_NONSPECIFIC_KINNESS_AXIAL_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-kinness-axial-registration-357.yaml"
)
FIGURE7_NONSPECIFIC_KINNESS_AXIAL_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-kinness-axial-match-358.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_INTRINSIC_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_modeldb_intrinsic_match_v1.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_INTRINSIC_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-modeldb-intrinsic-registration-359.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_INTRINSIC_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-modeldb-intrinsic-match-360.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_KINNESS_DETECTOR_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_modeldb_kinness_detector_match_v1.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_KINNESS_DETECTOR_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-modeldb-kinness-detector-registration-361.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_KINNESS_DETECTOR_RESULT_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-modeldb-kinness-detector-match-362.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_KINNESS_HYSTERESIS_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_modeldb_kinness_hysteresis_match_v1.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_KINNESS_HYSTERESIS_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-modeldb-kinness-hysteresis-registration-363.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_KINNESS_HYSTERESIS_RESULT_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-modeldb-kinness-hysteresis-match-364.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_UNIFORM_HANDLER_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_modeldb_uniform_handler_match_v1.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_UNIFORM_HANDLER_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-modeldb-uniform-handler-registration-369.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_UNIFORM_HANDLER_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-modeldb-uniform-handler-match-370.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_nonspecific_modeldb_calibrated20_match_v1.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_MATCH_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-modeldb-calibrated20-registration-371.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-modeldb-calibrated20-match-372.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_VERIFICATION_RESULT_PATH = (
    ROOT
    / "docs/validation-results/figure7-top5-nonspecific-modeldb-calibrated20-verification-374.yaml"
)
FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-nonspecific-modeldb-calibrated20-pair-376.yaml"
)
ISOLATED_NONSPECIFIC_REST_AUDIT_PROFILE_PATH = (
    ROOT / "configs/calibration/isolated_nonspecific_rest_audit_v1.yaml"
)
ISOLATED_NONSPECIFIC_REST_AUDIT_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/isolated-nonspecific-rest-audit-registration-377.yaml"
)
ISOLATED_NONSPECIFIC_REST_AUDIT_RESULT_PATH = (
    ROOT / "docs/validation-results/isolated-nonspecific-rest-audit-378.yaml"
)
ISOLATED_NONSPECIFIC_FIXED_POINT_DERIVATION_PATH = (
    ROOT / "docs/validation-results/isolated-nonspecific-fixed-point-derivation-379.yaml"
)
ISOLATED_NONSPECIFIC_FIXED_POINT_VERIFICATION_PATH = (
    ROOT / "docs/validation-results/isolated-nonspecific-fixed-point-verification-381.yaml"
)
ISOLATED_NONSPECIFIC_FIXED_POINT_PERTURBATION_PROFILE_PATH = (
    ROOT / "configs/calibration/isolated_nonspecific_fixed_point_soma_perturbation_v1.yaml"
)
ISOLATED_NONSPECIFIC_FIXED_POINT_PERTURBATION_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/isolated-nonspecific-fixed-point-soma-perturbation-registration-382.yaml"
)
ISOLATED_NONSPECIFIC_FIXED_POINT_PERTURBATION_RESULT_PATH = (
    ROOT / "docs/validation-results/isolated-nonspecific-fixed-point-soma-perturbation-383.yaml"
)
FIGURE7_FULL_GRID_CONNECTFROMALL_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_top5_full_grid_connectfromall_match_v1.yaml"
)
FIGURE7_FULL_GRID_CONNECTFROMALL_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-top5-full-grid-connectfromall-registration-384.yaml"
)
FIGURE7_FULL_GRID_CONNECTFROMALL_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-top5-full-grid-connectfromall-match-385.yaml"
)
FIGURE7_ALIGNED_VERIFICATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_aligned_on_center_verification_v1.yaml"
)
FIGURE7_ALIGNED_VERIFICATION_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-on-center-verification-registration-241.yaml"
)
FIGURE7_ALIGNED_VERIFICATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-on-center-verification-242.yaml"
)
FIGURE7_ALIGNED_MISMATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_aligned_on_center_mismatch_v1.yaml"
)
FIGURE7_ALIGNED_MISMATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-on-center-mismatch-registration-243.yaml"
)
FIGURE7_ALIGNED_PAIR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-on-center-pair-244.yaml"
)
FIGURE7_ALIGNED_SUSTAINED_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_aligned_sustained_match_v1.yaml"
)
FIGURE7_ALIGNED_SUSTAINED_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-sustained-registration-245.yaml"
)
FIGURE7_ALIGNED_SUSTAINED_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-sustained-match-246.yaml"
)
FIGURE7_EVENT_CURRENT_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-mismatch-event-current-registration-247.yaml"
)
FIGURE7_EVENT_CURRENT_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure7-mismatch-event-current-audit-248.yaml"
)
FIGURE7_PRE_EVENT_TRACE_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-mismatch-pre-event-trace-registration-249.yaml"
)
FIGURE7_PRE_EVENT_TRACE_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure7-mismatch-pre-event-trace-audit-250.yaml"
)
FIGURE7_GABA_CAPACITY_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_mismatch_gaba_capacity_v1.yaml"
)
FIGURE7_GABA_CAPACITY_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-mismatch-gaba-capacity-registration-251.yaml"
)
FIGURE7_GABA_CAPACITY_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-mismatch-gaba-capacity-252.yaml"
)
RADIAL_ANNULUS_GABA_GAIN_PROFILE_PATH = (
    ROOT / "configs/calibration/radial_annulus_gaba_gain_figure6_v1.yaml"
)
RADIAL_ANNULUS_GABA_GAIN_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/radial-annulus-gaba-gain-registration-253.yaml"
)
RADIAL_ANNULUS_GABA_GAIN_RESULT_PATH = (
    ROOT / "docs/validation-results/radial-annulus-gaba-gain-figure6-254.yaml"
)
FIGURE7_TWO_EVENT_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_aligned_two_event_match_v1.yaml"
)
FIGURE7_TWO_EVENT_MATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-two-event-registration-255.yaml"
)
FIGURE7_TWO_EVENT_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-aligned-two-event-match-256.yaml"
)
FIGURE7_TRN_ARRIVAL_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_trn_arrival_aligned_match_v1.yaml"
)
FIGURE7_TRN_ARRIVAL_MATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-trn-arrival-aligned-registration-257.yaml"
)
FIGURE7_TRN_ARRIVAL_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-trn-arrival-aligned-match-258.yaml"
)
FIGURE7_CORTICORETICULAR_GAIN_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_corticoreticular_gain_match_v1.yaml"
)
FIGURE7_CORTICORETICULAR_GAIN_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-corticoreticular-gain-registration-259.yaml"
)
FIGURE7_CORTICORETICULAR_GAIN_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-corticoreticular-gain-match-260.yaml"
)
FIGURE7_CORTICORETICULAR_VERIFICATION_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_corticoreticular_gain_verification_v1.yaml"
)
FIGURE7_CORTICORETICULAR_VERIFICATION_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-corticoreticular-verification-registration-261.yaml"
)
FIGURE7_CORTICORETICULAR_VERIFICATION_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-corticoreticular-gain-verification-262.yaml"
)
FIGURE7_CORTICORETICULAR_MISMATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_corticoreticular_gain_mismatch_v1.yaml"
)
FIGURE7_CORTICORETICULAR_MISMATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-corticoreticular-mismatch-registration-263.yaml"
)
FIGURE7_CORTICORETICULAR_MISMATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-corticoreticular-gain-mismatch-264.yaml"
)
FIGURE7_CORTICORETICULAR_MARGIN_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-corticoreticular-mismatch-margin-registration-265.yaml"
)
FIGURE7_CORTICORETICULAR_MARGIN_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-corticoreticular-mismatch-margin-audit-266.yaml"
)
FIGURE7_TARGETED_ANNULAR_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_targeted_annular_corticoreticular_match_v1.yaml"
)
FIGURE7_TARGETED_ANNULAR_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-targeted-annular-corticoreticular-registration-267.yaml"
)
FIGURE7_TARGETED_ANNULAR_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-targeted-annular-corticoreticular-match-268.yaml"
)
FIGURE7_GAIN8_AMPA_ARRIVAL_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_gain8_ampa_arrival_match_v1.yaml"
)
FIGURE7_GAIN8_AMPA_ARRIVAL_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-gain8-ampa-arrival-registration-269.yaml"
)
FIGURE7_GAIN8_AMPA_ARRIVAL_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-gain8-ampa-arrival-match-270.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_corticoreticular_ampa_delay2_v1.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure6-corticoreticular-ampa-delay-registration-271.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-corticoreticular-ampa-delay2-272.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY3_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_corticoreticular_ampa_delay3_v1.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY3_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure6-corticoreticular-ampa-delay3-registration-273.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY3_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-corticoreticular-ampa-delay3-274.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY2_CORRECTED_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_corticoreticular_ampa_delay2_corrected_v1.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY2_CORRECTED_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure6-corticoreticular-ampa-delay2-corrected-registration-275.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY2_CORRECTED_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-corticoreticular-ampa-delay2-corrected-276.yaml"
)
FIGURE7_DELAY2_GAIN8_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_delay2_gain8_match_v1.yaml"
)
FIGURE7_DELAY2_GAIN8_MATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-delay2-gain8-match-registration-277.yaml"
)
FIGURE7_DELAY2_GAIN8_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-delay2-gain8-match-278.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY3_CORRECTED_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_corticoreticular_ampa_delay3_corrected_v1.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY3_CORRECTED_REGISTRATION_PATH = (
    ROOT
    / "docs/validation-results/figure6-corticoreticular-ampa-delay3-corrected-registration-279.yaml"
)
FIGURE6_CORTICORETICULAR_AMPA_DELAY3_CORRECTED_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-corticoreticular-ampa-delay3-corrected-280.yaml"
)
FIGURE7_DELAY3_GAIN8_MATCH_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_delay3_gain8_match_v1.yaml"
)
FIGURE7_DELAY3_GAIN8_MATCH_REGISTRATION_PATH = (
    ROOT / "docs/validation-results/figure7-delay3-gain8-match-registration-281.yaml"
)
FIGURE7_DELAY3_GAIN8_MATCH_RESULT_PATH = (
    ROOT / "docs/validation-results/figure7-delay3-gain8-match-282.yaml"
)


def test_classic_calibration_contract_separates_training_and_holdout() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    assert contract.base_tag == "classic-smart-source-constrained-v0.1.0"
    assert set(contract.training_targets).isdisjoint(contract.holdout_targets)
    assert len(contract.dimensions) == 10
    assert contract.fingerprint == load_calibration_contract(CONTRACT_PATH).fingerprint


def test_candidate_fingerprint_requires_complete_admissible_values() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    candidate = {
        dimension.name: (
            dimension.values[0] if dimension.kind == "categorical" else dimension.grid[0]
        )
        for dimension in contract.dimensions
    }
    assert contract.candidate_fingerprint(candidate) == contract.candidate_fingerprint(candidate)
    with pytest.raises(ValueError, match="missing"):
        contract.candidate_fingerprint({})
    candidate["top_down_current_pA"] = 2000
    with pytest.raises(ValueError, match="outside declared bounds"):
        contract.candidate_fingerprint(candidate)


def test_contract_enumerates_complete_and_projected_spaces_deterministically() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    complete = list(contract.iter_candidates())
    assert len(complete) == 9216
    assert complete == list(contract.iter_candidates())
    assert len({contract.candidate_fingerprint(item) for item in complete}) == 9216

    stage_a_dimensions = (
        "intrinsic_cell_convention",
        "calcium_kinetics_convention",
        "calcium_density_convention",
        "nak_rate_convention",
        "axial_convention",
        "membrane_initialization_convention",
        "spike_event_rule",
    )
    projected = list(contract.iter_candidates(stage_a_dimensions))
    assert len(projected) == 768
    assert {item["gaussian_spread_convention"] for item in projected} == {"standard_deviation"}
    assert {item["relay_input_interpretation"] for item in projected} == {
        "archived_finite_conductance"
    }
    assert {item["top_down_current_pA"] for item in projected} == {600.0}


def test_candidate_maps_source_coupled_calcium_conventions() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    modeldb = next(contract.iter_candidates(()))
    modeldb_runtime = runtime_conventions_for_candidate(modeldb)
    assert modeldb_runtime.calcium_kinetics_convention == "modeldb_112923"
    assert modeldb_runtime.calcium_gate_convention == "modeldb_112923"
    assert modeldb_runtime.calcium_density_convention == "table3"
    assert modeldb_runtime.nak_rate_convention == "standard_traub_miles"

    paper = dict(modeldb, calcium_kinetics_convention="paper_2008")
    paper_runtime = runtime_conventions_for_candidate(paper)
    assert paper_runtime.calcium_kinetics_convention == "paper_2008"
    assert paper_runtime.calcium_gate_convention == "reciprocal"
    assert paper_runtime.fingerprint != modeldb_runtime.fingerprint

    printed_nak = dict(modeldb, nak_rate_convention="printed_smart")
    printed_nak_runtime = runtime_conventions_for_candidate(printed_nak)
    assert printed_nak_runtime.nak_rate_convention == "printed_smart"
    assert printed_nak_runtime.fingerprint != modeldb_runtime.fingerprint

    global_calcium = dict(modeldb, calcium_density_convention="methods_global_250")
    global_calcium_runtime = runtime_conventions_for_candidate(global_calcium)
    assert global_calcium_runtime.calcium_density_convention == "methods_global_250"
    assert global_calcium_runtime.fingerprint != modeldb_runtime.fingerprint


def test_projected_enumeration_rejects_unknown_or_duplicate_dimensions() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    with pytest.raises(ValueError, match="unknown active dimensions"):
        list(contract.iter_candidates(("unknown",)))
    with pytest.raises(ValueError, match="must be unique"):
        list(contract.iter_candidates(("axial_convention", "axial_convention")))


def test_trn_stage_a_requires_quiescence_and_recruitment() -> None:
    base = {
        "candidate_fingerprint": "a" * 64,
        "runtime_fingerprint": "b" * 64,
        "control_finite": True,
        "driven_finite": True,
        "control_post_drive_spikes": 0,
        "driven_post_drive_spikes": 1,
        "control_soma_range_mV": (-80.0, -60.0),
        "driven_soma_range_mV": (-80.0, 35.0),
    }
    assert TrnStageAResult(**base).promoted
    assert not TrnStageAResult(**dict(base, control_post_drive_spikes=1)).promoted
    assert not TrnStageAResult(**dict(base, driven_post_drive_spikes=0)).promoted
    assert not TrnStageAResult(**dict(base, driven_finite=False)).promoted


def test_frozen_trn_survivor_matches_current_contract_and_runtime() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    survivor = yaml.safe_load(TRN_SURVIVOR_PATH.read_text())
    candidate = survivor["candidate"]
    assert survivor["contract_fingerprint"] == contract.fingerprint
    assert survivor["candidate_fingerprint"] == contract.candidate_fingerprint(candidate)
    assert (
        survivor["runtime_fingerprint"] == runtime_conventions_for_candidate(candidate).fingerprint
    )
    assert survivor["result"]["control_post_drive_spikes"] == 0
    assert survivor["result"]["driven_post_drive_spikes"] > 0


def test_network_calibration_artifact_is_bound_to_frozen_trn_survivor() -> None:
    survivor = yaml.safe_load(TRN_SURVIVOR_PATH.read_text())
    artifact = yaml.safe_load(NETWORK_CALIBRATION_PATH.read_text())
    assert artifact["candidate_fingerprint"] == survivor["candidate_fingerprint"]
    assert artifact["runtime_fingerprint"] == survivor["runtime_fingerprint"]
    assert artifact["holdouts_consulted"] is False
    assert artifact["status"] == "failed-figure6-training-target"
    assert artifact["figure6"]["feedforward_chain_complete"] is False
    assert artifact["assessment"]["figure6_reproduced"] is False
    assert artifact["assessment"]["figure7_eligible"] is False


def test_gaussian_variance_discriminator_is_a_registered_complete_candidate() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    profile = yaml.safe_load(GAUSSIAN_VARIANCE_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    assert candidate["gaussian_spread_convention"] == "variance"
    assert profile["contract_fingerprint"] == contract.fingerprint
    assert profile["candidate_fingerprint"] == contract.candidate_fingerprint(candidate)
    assert (
        profile["runtime_fingerprint"] == runtime_conventions_for_candidate(candidate).fingerprint
    )


def test_figure8_voltage_audit_rejects_release_events_as_the_observable() -> None:
    artifact = yaml.safe_load(FIGURE8_VOLTAGE_AUDIT_PATH.read_text())
    assert artifact["observable_correction"]["old"] == ("SMART Equation-8 axonal release events")
    assert (
        artifact["literal_250_result"]["tonic_voltage_peaks"]
        > artifact["literal_250_result"]["tonic_release_events"]
    )
    assert artifact["calcium_unit_grid_mS_cm2"]["best_tonic_peak_times_ms"] == [
        48.13,
        112.96,
        195.31,
        288.55,
    ]
    assert artifact["assessment"]["calcium_unit_rescue"] is False
    assert artifact["assessment"]["holdouts_consulted"] is False


def test_figure6_relay_balance_causally_localizes_trn_feedback() -> None:
    survivor = yaml.safe_load(TRN_SURVIVOR_PATH.read_text())
    artifact = yaml.safe_load(FIGURE6_RELAY_BALANCE_PATH.read_text())
    assert artifact["candidate_fingerprint"] == survivor["candidate_fingerprint"]
    assert artifact["holdouts_consulted"] is False
    assert len(artifact["connected_result"]["relay_event_times_ms"]) == 1
    assert len(artifact["intrinsic_only_result"]["relay_event_times_ms"]) == 19
    trn_control = artifact["without_trn_to_relay_result"]
    assert trn_control["disabled_projection_ids"] == [
        "modeldb112923.projection.000",
        "modeldb112923.projection.001",
        "modeldb112923.projection.004",
    ]
    assert len(trn_control["relay_event_times_ms"]) == 19
    assert artifact["assessment"]["trn_inhibition_explains_repetition_failure"]


def test_all_registered_trn_survivors_fail_intact_relay_repetition() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_SCREEN_PATH.read_text())
    assert artifact["holdouts_consulted"] is False
    assert artifact["candidate_count"] == 6
    assert artifact["relay_repetition_survivor_count"] == 0
    assert {len(item["result"]["relay_event_times_ms"]) for item in artifact["outcomes"]} == {1}
    assert {item["result"]["relay_event_times_ms"][0] for item in artifact["outcomes"]} == {
        1.8900000000000001
    }


def test_registered_trn_predrive_does_not_rescue_figure6_relay() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_EQUILIBRATION_PATH.read_text())
    assert artifact["holdouts_consulted"] is False
    assert artifact["protocol"]["warmup_ms"] == 5.0
    assert artifact["result"]["relay_event_times_ms"] == []
    assert artifact["assessment"]["equilibration_rejected"]


def test_figure6_source_hybrid_is_reproducible_but_not_promoted() -> None:
    profile = yaml.safe_load(FIGURE6_SOURCE_HYBRID_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = yaml.safe_load(FIGURE6_SOURCE_HYBRID_RESULT_PATH.read_text())

    assert profile["candidate_fingerprint"] == fingerprint
    assert (
        profile["runtime_fingerprint"] == runtime_conventions_for_candidate(candidate).fingerprint
    )
    assert artifact["candidate_fingerprint"] == fingerprint
    assert artifact["population_spikes"]["thalamic_relay"] == 5
    assert artifact["population_spikes"]["layer4_excitatory_v1"] == 5
    assert artifact["maps"]["bottom_up_oriented"]
    assert not artifact["maps"]["top_down_oriented"]
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_leading_figure6_hybrid_keeps_the_top_down_gate_fixed() -> None:
    profile = yaml.safe_load(FIGURE6_RELAY_AXIAL_HYBRID_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = yaml.safe_load(FIGURE6_RELAY_AXIAL_HYBRID_RESULT_PATH.read_text())

    assert profile["candidate_fingerprint"] == fingerprint
    assert (
        profile["runtime_fingerprint"] == runtime_conventions_for_candidate(candidate).fingerprint
    )
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert artifact["maps"]["minimum_top_down_horizontal_contrast"] == 0.01
    assert artifact["maps"]["top_down_horizontal_orientation_contrast"] < 0.01
    assert artifact["assessment"]["only_failed_gate"] == (
        "top_down_horizontal_orientation_contrast"
    )
    assert not artifact["assessment"]["promoted"]


def test_leading_figure6_map_shortfall_is_wide_field_phase_specific() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_AXIAL_DECOMPOSITION_PATH.read_text())
    maps = artifact["maps"]
    assert (
        maps["top_down_narrow"]["horizontal_orientation_contrast"]
        > maps["top_down_wide"]["horizontal_orientation_contrast"]
    )
    assert maps["top_down_wide"]["horizontal_arm_delta"][1] < 0
    assert maps["top_down_wide"]["horizontal_arm_delta"][0] > 0
    assert maps["top_down_wide"]["vertical_arm_delta"] == [0.0] * 4
    assert artifact["active_cell_times_ms"]["category_40"] == [
        7.88,
        14.93,
        39.27,
        62.02,
        82.92,
    ]


def test_registered_intrinsic_alternatives_do_not_clear_figure6c() -> None:
    artifact = yaml.safe_load(FIGURE6_LEADING_ALTERNATIVES_PATH.read_text())
    assert artifact["holdouts_consulted"] is False
    assert artifact["assessment"]["base_profile_remains_leading"]
    assert all(
        artifact["assessment"][name] is False
        for name in (
            "event_rule_rescue",
            "initialization_rescue",
            "calcium_kinetics_rescue",
            "nak_rate_rescue",
            "archived_category_cell_rescue",
            "serialized_weight_initialization_rescue",
        )
    )
    assert not artifact["assessment"]["figure6_reproduced"]


def test_figure6_learning_phase_exactly_localizes_wide_field_depression() -> None:
    profile = yaml.safe_load(FIGURE6_RELAY_AXIAL_HYBRID_PROFILE_PATH.read_text())
    artifact = yaml.safe_load(FIGURE6_LEARNING_PHASE_PATH.read_text())
    assert artifact["candidate_fingerprint"] == profile["candidate_fingerprint"]
    assert artifact["runtime_fingerprint"] == profile["runtime_fingerprint"]
    assert artifact["holdouts_consulted"] is False
    assert artifact["assessment"]["integration_consistent"]
    assert artifact["assessment"]["maximum_delta_reconstruction_error"] < 1e-12

    by_projection_target = {
        (connection["projection_id"], connection["target_index"]): connection
        for connection in artifact["result"]["connections"]
    }
    wide_near = by_projection_target[("modeldb112923.projection.005", 39)]
    wide_far = by_projection_target[("modeldb112923.projection.005", 38)]
    narrow_near = by_projection_target[("modeldb112923.projection.007", 39)]
    assert wide_near["measured_delta"] < 0 < wide_far["measured_delta"]
    assert wide_near["negative_correlation_delta"] < -wide_near["positive_correlation_delta"]
    assert narrow_near["measured_delta"] > 0
    assert artifact["result"]["relay_event_times_ms"][31] == []


def test_projection_level_learning_bounds_are_source_literal_but_fail_figure6c() -> None:
    profile = yaml.safe_load(FIGURE6_PROJECTION_BOUNDS_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    conventions = runtime_conventions_for_candidate(candidate)
    artifact = yaml.safe_load(FIGURE6_PROJECTION_BOUNDS_RESULT_PATH.read_text())
    assert profile["candidate_fingerprint"] == fingerprint
    assert profile["runtime_fingerprint"] == conventions.fingerprint
    assert conventions.gaussian_learning_bounds_convention == "projection_level"
    assert artifact["candidate_fingerprint"] == fingerprint
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert artifact["maps"]["top_down_horizontal_orientation_contrast"] < 0
    assert any(delta > 0 for delta in artifact["maps"]["top_down_wide"]["vertical_arm_delta"])
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_projection_level_depression_scale_is_incompatible_with_local_baseline() -> None:
    profile = yaml.safe_load(FIGURE6_PROJECTION_D_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    conventions = runtime_conventions_for_candidate(candidate)
    artifact = yaml.safe_load(FIGURE6_PROJECTION_D_RESULT_PATH.read_text())
    assert profile["candidate_fingerprint"] == fingerprint
    assert profile["runtime_fingerprint"] == conventions.fingerprint
    assert conventions.postsynaptic_depression_scale_convention == ("serialized_projection_bounds")
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert artifact["maps"]["top_down_horizontal_orientation_contrast"] < 0
    assert min(artifact["maps"]["top_down_narrow"]["horizontal_arm_after"]) < 0
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_figure6_teaching_volley_decomposition_closes_each_connection() -> None:
    artifact = yaml.safe_load(FIGURE6_VOLLEY_DECOMPOSITION_PATH.read_text())
    assert artifact["assessment"]["integration_consistent"]
    assert artifact["assessment"]["maximum_delta_reconstruction_error"] < 1e-12
    by_projection_target = {
        (connection["projection_id"], connection["target_index"]): connection
        for connection in artifact["result"]["connections"]
    }
    wide_near = by_projection_target[("modeldb112923.projection.005", 39)]
    assert sum(window["measured_delta"] for window in wide_near["windows"]) == (
        pytest.approx(wide_near["measured_delta"])
    )
    onset_window = wide_near["windows"][0]
    assert onset_window["start_ms"] == pytest.approx(9.88)
    assert onset_window["end_ms"] == pytest.approx(16.93)
    assert onset_window["positive_correlation_delta"] == 0
    assert onset_window["negative_correlation_delta"] < 0


def test_population_resolved_axial_profile_is_retracted_by_amplitude_audit() -> None:
    profile = yaml.safe_load(FIGURE6_POPULATION_AXIAL_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    conventions = runtime_conventions_for_candidate(candidate)
    historical = yaml.safe_load(FIGURE6_POPULATION_AXIAL_RESULT_PATH.read_text())
    artifact = yaml.safe_load(FIGURE6_POPULATION_AXIAL_AMPLITUDE_AUDIT_PATH.read_text())
    assert profile["candidate_fingerprint"] == fingerprint
    assert profile["runtime_fingerprint"] == conventions.fingerprint
    assert profile["status"] == "retracted-shape-pass-amplitude-fail"
    assert artifact["candidate_fingerprint"] == fingerprint
    assert artifact["status"] == "partial-figure6b-pass-figure6c-fail"
    assert artifact["population_spikes"]["thalamic_relay"] == 20
    assert all(
        len(times) == 4 for times in artifact["active_cell_times_ms"]["relay_horizontal"].values()
    )
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert artifact["maps"]["top_down_horizontal_orientation_contrast"] >= 0.01
    assert artifact["maps"]["top_down_combined"]["maximum_after"] < 0.5
    assert not artifact["maps"]["top_down_oriented"]
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]
    # Artifact 136 remains the immutable record of the superseded shape-only gate.
    assert historical["status"] == "complete-figure6-pass"
    assert historical["maps"]["top_down_oriented"]


def test_shape_only_figure6_profile_fails_first_genuine_figure7_holdout() -> None:
    artifact = yaml.safe_load(FIGURE7_POPULATION_AXIAL_RESULT_PATH.read_text())
    assert artifact["holdouts_consulted"] is True
    assert artifact["status"] == "passing-figure6-failed-figure7-holdout"
    assert artifact["assessment"]["figure6_reproduced"]
    profile = yaml.safe_load(FIGURE6_POPULATION_AXIAL_PROFILE_PATH.read_text())
    assert profile["status"] == "retracted-shape-pass-amplitude-fail"
    assert artifact["figure7"]["conditions"]["match"]["nonspecific_spike_times_ms"] == [
        pytest.approx(0.74)
    ]
    assert artifact["figure7"]["conditions"]["mismatch"]["nonspecific_spike_times_ms"] == [
        pytest.approx(0.74)
    ]
    assessment = artifact["figure7"]["assessment"]
    assert assessment["arousal"]["match_rate_hz"] == 10.0
    assert assessment["arousal"]["mismatch_rate_hz"] == 10.0
    assert assessment["pathway"]["match_trn_spikes"] == 81
    assert assessment["pathway"]["mismatch_trn_spikes"] == 81
    assert not artifact["figure7"]["reproduced"]


def test_population_axial_learning_phase_exactly_closes_amplitude_deficit() -> None:
    artifact = yaml.safe_load(FIGURE6_POPULATION_AXIAL_LEARNING_PHASE_PATH.read_text())
    assert artifact["assessment"]["integration_consistent"]
    assert artifact["assessment"]["maximum_delta_reconstruction_error"] < 1e-12
    result = artifact["result"]
    assert len(result["category_event_times_ms"]) == 5
    active_relays = {38, 39, 41, 42}
    assert all(len(result["relay_event_times_ms"][index]) == 4 for index in active_relays)
    selected = {
        (connection["projection_id"], connection["target_index"]): connection
        for connection in result["connections"]
    }
    wide = selected[("modeldb112923.projection.005", 39)]
    narrow = selected[("modeldb112923.projection.007", 39)]
    assert wide["measured_delta"] == pytest.approx(0.013663, abs=1e-6)
    assert narrow["measured_delta"] == pytest.approx(0.026395, abs=1e-6)
    assert wide["postsynaptic_positive_overlap_ms"] < 0.5
    assert narrow["postsynaptic_positive_overlap_ms"] < 0.5


def test_learning_threshold_and_coordinate_candidates_are_all_rejected() -> None:
    artifact = yaml.safe_load(FIGURE6_LEARNING_THRESHOLD_ASSESSMENT_PATH.read_text())
    assert artifact["registered_gates"]["minimum_combined_peak"] == 2.0
    assert artifact["registered_gates"]["events_per_relay_in_100_ms"] == 4
    assert len(artifact["candidates"]) == 4
    assert not any(candidate["figure6_reproduced"] for candidate in artifact["candidates"])
    leak_relative = artifact["candidates"][-1]
    assert leak_relative["combined_peak"] == pytest.approx(0.527413146)
    assert leak_relative["relay_spikes"] == 58
    assert not leak_relative["relay_recruitment_confined"]
    assert artifact["assessment"]["promoted_profile"] is None


def test_methods_dual_and_candidates_fail_amplitude_or_spatial_gate() -> None:
    artifact = yaml.safe_load(FIGURE6_LEARNING_RULE_ASSESSMENT_PATH.read_text())
    assert len(artifact["candidates"]) == 6
    absolute = artifact["candidates"][-2]
    interaction = artifact["candidates"][-1]
    assert absolute["combined_peak"] == pytest.approx(0.193188217)
    assert absolute["relay_recruitment_confined"]
    assert interaction["combined_peak"] == pytest.approx(0.968761508)
    assert interaction["relay_spikes"] == 58
    assert not interaction["relay_recruitment_confined"]
    assert not artifact["assessment"]["learning_rule_explanation_sufficient"]
    assert artifact["assessment"]["promoted_profile"] is None


def test_dual_and_leak_phase_exposes_subthreshold_surround_potentiation() -> None:
    artifact = yaml.safe_load(FIGURE6_DUAL_AND_LEAK_PHASE_PATH.read_text())
    assert artifact["assessment"]["integration_consistent"]
    assert artifact["assessment"]["maximum_delta_reconstruction_error"] < 1e-12
    selected = {
        (connection["projection_id"], connection["target_index"]): connection
        for connection in artifact["result"]["connections"]
    }
    horizontal = selected[("modeldb112923.projection.005", 39)]
    vertical = selected[("modeldb112923.projection.005", 31)]
    assert horizontal["postsynaptic_positive_overlap_ms"] > 5.0
    assert vertical["postsynaptic_positive_overlap_ms"] > 2.0
    assert artifact["result"]["relay_event_times_ms"][31] == []
    assert vertical["final_weight"] > 0.2


def test_registered_nak_families_do_not_extend_relay_positive_phase() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_WAVEFORM_PATH.read_text())
    results = {item["nak_rate_convention"]: item["result"] for item in artifact["results"]}
    standard = results["standard_traub_miles"]
    hybrid = results["archived_activation_printed_inactivation"]
    assert len(standard["relay_event_times_ms"]) == 1
    assert len(hybrid["relay_event_times_ms"]) == 1
    assert standard["soma_time_above_30_mV"] == pytest.approx(0.18)
    assert hybrid["soma_time_above_30_mV"] == pytest.approx(0.17)
    for family in ("printed_activation_archived_inactivation", "printed_smart"):
        assert results[family]["relay_event_times_ms"] == []
        assert results[family]["target_layer4_event_times_ms"] == []
        assert results[family]["soma_voltage_peak_mV"] < 0
    assert not artifact["assessment"]["nak_family_explanation_sufficient"]
    assert artifact["assessment"]["waveform_survivor"] is None


def test_upward_learning_timestamp_preserves_spikes_but_reduces_peak() -> None:
    artifact = yaml.safe_load(FIGURE6_UPWARD_TIMESTAMP_PATH.read_text())
    assert artifact["population_spikes"]["thalamic_relay"] == 20
    assert artifact["relay_recruitment"]["confined_to_horizontal_bar_at_40_hz"]
    assert artifact["maps"]["top_down_combined"]["maximum_after"] == pytest.approx(0.1074981615)
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_leak_relative_plus30_preserves_confinement_but_misses_peak() -> None:
    artifact = yaml.safe_load(FIGURE6_LEAK_PLUS30_PATH.read_text())
    assert artifact["population_spikes"]["thalamic_relay"] == 20
    assert artifact["relay_recruitment"]["confined_to_horizontal_bar_at_40_hz"]
    combined = artifact["maps"]["top_down_combined"]
    assert combined["maximum_after"] == pytest.approx(0.1860603038)
    assert combined["horizontal_orientation_contrast"] > 0.01
    assert not artifact["assessment"]["figure6_reproduced"]


@pytest.mark.parametrize(
    ("profile_path", "result_path", "relay_spikes", "trn_spikes"),
    (
        (
            FIGURE7_THALAMOCORTICAL_AXIAL_PROFILE_PATH,
            FIGURE6_THALAMOCORTICAL_AXIAL_RESULT_PATH,
            5,
            319,
        ),
        (
            FIGURE7_POPULATION_TRN_PROFILE_PATH,
            FIGURE6_POPULATION_TRN_RESULT_PATH,
            0,
            810,
        ),
    ),
)
def test_post_holdout_trn_source_candidates_fail_figure6_prerequisite(
    profile_path: Path,
    result_path: Path,
    relay_spikes: int,
    trn_spikes: int,
) -> None:
    profile = yaml.safe_load(profile_path.read_text())
    candidate = profile["candidate"]
    artifact = yaml.safe_load(result_path.read_text())
    assert (
        profile["runtime_fingerprint"] == runtime_conventions_for_candidate(candidate).fingerprint
    )
    assert artifact["population_spikes"]["thalamic_relay"] == relay_spikes
    assert artifact["population_spikes"]["trn"] == trn_spikes
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_trn_potassium_source_matrix_has_no_connected_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_POTASSIUM_SCREEN_PATH.read_text())
    assert artifact["status"] == "no-connected-causal-survivor"
    assert artifact["assessment"]["connected_causal_survivors"] == 0
    assert all(item["post_bottom_up_trn_events"] == 0 for item in artifact["outcomes"])
    assert artifact["assessment"]["best_sampled_soma_peak_mV"] == pytest.approx(-12.1097056947)
    assert artifact["assessment"]["gap_to_published_arm_threshold_mV"] > 42.0


def test_trn_soma_potassium_behavior_grid_is_registered_outside_source_range() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_SOMA_POTASSIUM_PROFILE_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["dimension"]["source_status"] == (
        "behavior_calibration_outside_published_80_to_100_range"
    )
    assert profile["dimension"]["grid"] == [
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
    ]
    assert profile["fixed_choices"]["trn_spike_event_proximal_blend_fraction"] is None


def test_lower_trn_soma_potassium_has_no_isolated_recruitment_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_SOMA_POTASSIUM_STAGE1_PATH.read_text())
    assert artifact["status"] == "no-stage-1-survivor"
    assert artifact["stage_1_survivor_densities_mS_cm2"] == []
    assert artifact["assessment"]["all_controls_post_drive_quiet"]
    assert artifact["assessment"]["all_driven_trials_finite"]
    assert not artifact["assessment"]["lower_potassium_sufficient_for_recruitment"]
    assert not artifact["assessment"]["advance_to_connected_match"]
    assert artifact["assessment"]["best_driven_soma_peak_mV"] < -22.0
    assert all(not item["stage_1_pass"] for item in artifact["outcomes"])
    assert all(item["driven"]["post_drive_spike_times_ms"] == [] for item in artifact["outcomes"])


def test_source_recovery_followup_does_not_promote_compiler_log_to_source() -> None:
    artifact = yaml.safe_load(LEGACY_SOURCE_RECOVERY_FOLLOWUP_PATH.read_text())
    assert artifact["status"] == "source-body-not-recovered"
    assert artifact["new_public_trace"]["evidence_class"] == "third-party-build-log"
    assert not artifact["search_outcome"]["package_archive_recovered"]
    assert not artifact["search_outcome"]["detector_implementation_recovered"]
    assert not artifact["search_outcome"]["source_correction_authorized"]


def test_trn_soma_sodium_grid_registers_nonmonotonic_shunting_drive() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_SOMA_SODIUM_PROFILE_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["dimension"]["source_value_mS_cm2"] == 100.0
    assert profile["dimension"]["grid"] == [
        100.0,
        125.0,
        150.0,
        200.0,
        300.0,
        400.0,
    ]
    assert profile["stage_1_protocol"]["drive_multipliers"] == [
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
        1.0,
    ]


def test_increased_trn_soma_sodium_has_no_detector_cycle_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_SOMA_SODIUM_STAGE1_PATH.read_text())
    assert artifact["status"] == "no-stage-1-survivor"
    assert artifact["stage_1_survivor_densities_mS_cm2"] == []
    assessment = artifact["assessment"]
    assert assessment["all_controls_post_drive_quiet"]
    assert assessment["finite_driven_trial_count"] == 36
    assert assessment["total_driven_trial_count"] == 36
    assert assessment["best_finite_driven_soma_peak_mV"] == pytest.approx(14.6526694610)
    assert not assessment["sodium_density_sufficient_in_registered_assay"]
    assert not assessment["advance_to_connected_match"]
    assert all(not item["stage_1_pass"] for item in artifact["outcomes"])


def test_trn_detector_hysteresis_grid_separates_source_and_calibration_pairs() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_DETECTOR_HYSTERESIS_PROFILE_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["dimension"]["source_pair"] == {
        "label": "paper_30_to_0",
        "arm_mV": 30.0,
        "release_mV": 0.0,
        "status": "published",
    }
    assert profile["dimension"]["source_status"] == ("behavior_calibration_except_paper_30_to_0")
    assert profile["stage_1_protocol"]["drive_multipliers"] == [
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
        1.0,
    ]


def test_trn_detector_hysteresis_has_four_isolated_fresh_cycle_survivors() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_DETECTOR_HYSTERESIS_STAGE1_PATH.read_text())
    assert artifact["status"] == "stage-1-survivors-found"
    assert artifact["stage_1_survivor_labels"] == [
        "arm_n35_release_n45",
        "arm_n30_release_n40",
        "arm_n25_release_n35",
        "arm_n20_release_n30",
    ]
    assessment = artifact["assessment"]
    assert assessment["stage_1_survivor_count"] == 4
    assert not assessment["source_pair_survives"]
    assert assessment["all_controls_pass"]
    assert not assessment["startup_latched_release_counts_as_pass"]
    assert assessment["advance_to_connected_prerequisite"]

    outcomes = {item["pair"]["label"]: item for item in artifact["outcomes"]}
    assert not outcomes["paper_30_to_0"]["stage_1_pass"]
    assert all(
        not item["fresh_detector_cycle_pass"]
        for item in outcomes["paper_30_to_0"]["driven_outcomes"]
    )
    sparse = outcomes["arm_n20_release_n30"]
    passing = [item for item in sparse["driven_outcomes"] if item["fresh_detector_cycle_pass"]]
    assert len(passing) == 1
    assert passing[0]["drive_multiplier"] == 0.4
    result = passing[0]["result"]
    assert len(result["post_stimulus_spike_times_ms"]) == 1
    assert result["threshold_upcrossings"] == 1
    assert result["arm_transitions"] == 1
    assert result["release_transitions"] == 1
    assert all(
        not outcomes[label]["stage_1_pass"]
        for label in (
            "arm_n10_release_n20",
            "arm_0_release_n10",
            "arm_10_release_0",
            "arm_20_release_10",
            "arm_30_release_20",
        )
    )


def test_trn_detector_hysteresis_has_no_figure6_survivor() -> None:
    artifact = yaml.safe_load(FIGURE6_TRN_DETECTOR_HYSTERESIS_PREREQUISITE_PATH.read_text())
    assert artifact["status"] == "figure6-prerequisite-complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["figure6_survivor_labels"] == []
    assessment = artifact["assessment"]
    assert assessment["completed_candidate_count"] == 4
    assert assessment["registered_candidate_count"] == 4
    assert assessment["figure6_survivor_count"] == 0
    assert not assessment["advance_to_same_network_match"]
    assert [item["population_spikes"]["trn"] for item in artifact["outcomes"]] == [
        506,
        525,
        489,
        511,
    ]
    for item in artifact["outcomes"]:
        assert item["population_spikes"]["thalamic_relay"] == 5
        assert item["relay_spike_indices"] == [38, 39, 40, 41, 42]
        assert set(item["relay_event_counts_by_index"].values()) == {1}
        assert item["gates"]["relay_active_indices"]
        assert item["gates"]["feedforward_chain_complete"]
        assert item["gates"]["bottom_up_horizontal_orientation"]
        assert item["gates"]["top_down_horizontal_contrast"]
        assert not item["gates"]["relay_event_count"]
        assert not item["gates"]["relay_events_per_active_index"]
        assert not item["gates"]["causal_pair_in_learning_window"]
        assert not item["figure6_pass"]


def test_trn_gaba_transfer_grid_has_one_calibrated_figure6_survivor() -> None:
    artifact = yaml.safe_load(FIGURE6_TRN_GABA_TRANSFER_GRID_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_1_survivor_scales"] == [0.001, 0.003, 0.01]
    assert artifact["stage_2_survivor_scales"] == [0.01]
    assessment = artifact["assessment"]
    assert assessment["stage_1_completed_count"] == 8
    assert assessment["registered_scale_count"] == 8
    assert assessment["stage_2_completed_count"] == 3
    assert assessment["figure6_survivor_count"] == 1
    assert assessment["advance_to_same_network_match"]
    outcomes = {item["scale"]: item for item in artifact["stage_2_outcomes"]}
    assert outcomes[0.001]["population_spikes"]["thalamic_relay"] == 35
    assert outcomes[0.003]["population_spikes"]["thalamic_relay"] == 30
    survivor = outcomes[0.01]
    assert survivor["population_spikes"]["thalamic_relay"] == 20
    assert survivor["population_spikes"]["trn"] == 728
    assert set(survivor["relay_event_counts_by_index"].values()) == {4}
    assert set(survivor["relay_detector_threshold_upcrossings_by_index"].values()) == {4}
    assert set(survivor["relay_detector_arm_transitions_by_index"].values()) == {4}
    assert set(survivor["relay_detector_release_transitions_by_index"].values()) == {4}
    assert all(survivor["gates"].values())
    assert survivor["pass"]


def test_calibrated_trn_transfer_passes_same_network_match_gate() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_GABA_TRANSFER_MATCH_PATH.read_text())
    assert artifact["status"] == "match-pass"
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    result = artifact["result"]
    assert result["learned_state_provenance"] == "same-network-figure6-episode"
    assert len(result["relay_spike_indices"]) == 25
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 538
    assert len(result["nonspecific_spike_times_ms"]) == 6
    assert set(artifact["relay_event_counts_by_index"].values()) == {5}
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_threshold_upcrossings_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_arm_transitions_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_release_transitions_by_index"]
    )
    assert all(artifact["gates"].values())
    assert artifact["assessment"] == {
        "same_network_match_pass": True,
        "advance_to_mismatch": True,
    }


def test_calibrated_trn_transfer_pair_fails_official_directional_pathway() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_GABA_TRANSFER_PAIR_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["holdouts_consulted"] == ["figure7_match", "figure7_mismatch"]
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 60.0,
        "mismatch_rate_hz": 70.0,
    }
    assert assessment["pathway"]["match_active_relay_cells"] == 5
    assert assessment["pathway"]["mismatch_active_relay_cells"] == 5
    assert assessment["pathway"]["match_trn_spikes"] == 538
    assert assessment["pathway"]["mismatch_trn_spikes"] == 589
    assert set(assessment["pathway"]["mismatch_active_relay_indices"]) == {
        22,
        31,
        40,
        49,
        58,
    }
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": False,
        "mismatch_more_nonspecific_events": True,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_each_compartment_source_endpoint_fails_figure6_repeat_gate() -> None:
    artifact = yaml.safe_load(FIGURE6_TRN_GABA_COMPARTMENT_SOURCE_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_1_survivor_labels"] == []
    assert artifact["stage_2_outcomes"] == []
    assert artifact["stage_2_survivor_labels"] == []
    assert artifact["assessment"] == {
        "registered_profile_count": 3,
        "stage_1_completed_count": 3,
        "stage_2_completed_count": 0,
        "figure6_survivor_count": 0,
        "advance_to_figure7": False,
    }
    assert [item["label"] for item in artifact["stage_1_outcomes"]] == [
        "restore_soma_source",
        "restore_proximal_source",
        "restore_distal_source",
    ]
    for item in artifact["stage_1_outcomes"]:
        assert item["population_spikes"]["thalamic_relay"] == 5
        assert item["population_spikes"]["trn"] == 392
        assert set(item["relay_event_counts_by_index"].values()) == {1}
        assert not item["pass"]


def test_distal_intermediate_grid_has_one_figure6_survivor() -> None:
    artifact = yaml.safe_load(FIGURE6_TRN_GABA_COMPARTMENT_INTERMEDIATE_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_1_survivor_labels"] == [
        "distal_0_015",
        "distal_0_02",
    ]
    assert artifact["stage_2_survivor_labels"] == ["distal_0_015"]
    assert artifact["assessment"] == {
        "registered_profile_count": 9,
        "stage_1_completed_count": 9,
        "stage_2_completed_count": 2,
        "figure6_survivor_count": 1,
        "advance_to_figure7": True,
    }

    outcomes = {item["label"]: item for item in artifact["stage_2_outcomes"]}
    survivor = outcomes["distal_0_015"]
    assert survivor["scales"] == {
        "modeldb112923.projection.000": 0.01,
        "modeldb112923.projection.001": 0.01,
        "modeldb112923.projection.004": 0.015,
    }
    assert survivor["population_spikes"]["thalamic_relay"] == 20
    assert survivor["population_spikes"]["trn"] == 724
    assert set(survivor["relay_event_counts_by_index"].values()) == {4}
    assert all(survivor["gates"].values())
    assert survivor["pass"]

    rejected = outcomes["distal_0_02"]
    assert rejected["population_spikes"]["thalamic_relay"] == 19
    assert rejected["population_spikes"]["trn"] == 714
    assert not rejected["gates"]["relay_event_count"]
    assert not rejected["gates"]["relay_events_per_active_index"]
    assert not rejected["pass"]


def test_distal_transfer_survivor_passes_same_network_match() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_GABA_COMPARTMENT_MATCH_PATH.read_text())
    assert artifact["status"] == "match-pass"
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    assert artifact["projection_weight_scales"] == {
        "modeldb112923.projection.000": 0.01,
        "modeldb112923.projection.001": 0.01,
        "modeldb112923.projection.004": 0.015,
    }
    result = artifact["result"]
    assert len(result["relay_spike_indices"]) == 22
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 558
    assert len(result["nonspecific_spike_times_ms"]) == 9
    assert artifact["relay_event_counts_by_index"] == {
        "38": 5,
        "39": 4,
        "40": 4,
        "41": 4,
        "42": 5,
    }
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_threshold_upcrossings_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_arm_transitions_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_release_transitions_by_index"]
    )
    assert all(artifact["gates"].values())
    assert artifact["assessment"] == {
        "same_network_match_pass": True,
        "advance_to_mismatch": True,
    }


def test_distal_transfer_pair_fails_official_figure7_directions() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_GABA_COMPARTMENT_PAIR_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["holdouts_consulted"] == [
        "figure7_match",
        "figure7_mismatch",
    ]
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 90.0,
        "mismatch_rate_hz": 80.0,
    }
    assert assessment["pathway"]["match_active_relay_cells"] == 5
    assert assessment["pathway"]["mismatch_active_relay_cells"] == 5
    assert assessment["pathway"]["match_trn_spikes"] == 558
    assert assessment["pathway"]["mismatch_trn_spikes"] == 566
    assert set(assessment["pathway"]["mismatch_active_relay_indices"]) == {
        22,
        31,
        40,
        49,
        58,
    }
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": False,
        "mismatch_more_nonspecific_events": False,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_projection022_source_resolution_preserves_complete_figure6() -> None:
    artifact = yaml.safe_load(FIGURE6_PROJECTION022_SOURCE_RESOLUTION_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_1_survivor_labels"] == ["paper_supplement_projection022"]
    assert artifact["stage_2_survivor_labels"] == ["paper_supplement_projection022"]
    assert artifact["assessment"] == {
        "registered_profile_count": 1,
        "stage_1_completed_count": 1,
        "stage_2_completed_count": 1,
        "figure6_survivor_count": 1,
        "advance_to_figure7": True,
    }
    outcome = artifact["stage_2_outcomes"][0]
    assert outcome["population_spikes"]["thalamic_relay"] == 20
    assert outcome["population_spikes"]["trn"] == 541
    assert set(outcome["relay_event_counts_by_index"].values()) == {4}
    assert outcome["top_down_combined_horizontal_orientation_contrast"] == pytest.approx(
        0.48046899168012513
    )
    assert all(outcome["gates"].values())
    assert outcome["pass"]


def test_projection022_source_resolution_passes_same_network_match() -> None:
    artifact = yaml.safe_load(FIGURE7_PROJECTION022_SOURCE_RESOLUTION_MATCH_PATH.read_text())
    assert artifact["status"] == "match-pass"
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    result = artifact["result"]
    assert len(result["relay_spike_indices"]) == 30
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 547
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert set(artifact["relay_event_counts_by_index"].values()) == {6}
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_threshold_upcrossings_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_arm_transitions_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_release_transitions_by_index"]
    )
    assert all(artifact["gates"].values())
    assert artifact["assessment"]["advance_to_mismatch"]


def test_projection022_source_resolution_fixes_only_trn_direction() -> None:
    artifact = yaml.safe_load(FIGURE7_PROJECTION022_SOURCE_RESOLUTION_PAIR_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 50.0,
        "mismatch_rate_hz": 50.0,
    }
    pathway = assessment["pathway"]
    assert pathway["match_active_relay_cells"] == 5
    assert pathway["mismatch_active_relay_cells"] == 9
    assert pathway["match_trn_spikes"] == 547
    assert pathway["mismatch_trn_spikes"] == 514
    assert set(pathway["mismatch_active_relay_indices"]) == {
        22,
        31,
        38,
        39,
        40,
        41,
        42,
        49,
        58,
    }
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": True,
        "mismatch_more_nonspecific_events": False,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_source_resolved_distal002_preserves_complete_figure6() -> None:
    artifact = yaml.safe_load(FIGURE6_PROJECTION022_DISTAL002_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_2_survivor_labels"] == ["paper_supplement_projection022_distal_0_02"]
    assert artifact["assessment"]["advance_to_figure7"]
    outcome = artifact["stage_2_outcomes"][0]
    assert outcome["scales"]["modeldb112923.projection.004"] == 0.02
    assert outcome["population_spikes"]["thalamic_relay"] == 20
    assert outcome["population_spikes"]["trn"] == 544
    assert set(outcome["relay_event_counts_by_index"].values()) == {4}
    assert outcome["top_down_combined_horizontal_orientation_contrast"] == pytest.approx(
        0.5687421760725476
    )
    assert all(outcome["gates"].values())
    assert outcome["pass"]


def test_source_resolved_distal002_passes_same_network_match() -> None:
    artifact = yaml.safe_load(FIGURE7_PROJECTION022_DISTAL002_MATCH_PATH.read_text())
    assert artifact["status"] == "match-pass"
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    result = artifact["result"]
    assert len(result["relay_spike_indices"]) == 30
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 522
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert set(artifact["relay_event_counts_by_index"].values()) == {6}
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_threshold_upcrossings_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_arm_transitions_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_release_transitions_by_index"]
    )
    assert all(artifact["gates"].values())
    assert artifact["assessment"]["advance_to_mismatch"]


def test_source_resolved_distal002_pair_remains_nonselective() -> None:
    artifact = yaml.safe_load(FIGURE7_PROJECTION022_DISTAL002_PAIR_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 50.0,
        "mismatch_rate_hz": 50.0,
    }
    pathway = assessment["pathway"]
    assert pathway["match_active_relay_cells"] == 5
    assert pathway["mismatch_active_relay_cells"] == 9
    assert pathway["match_trn_spikes"] == 522
    assert pathway["mismatch_trn_spikes"] == 520
    assert set(pathway["mismatch_active_relay_indices"]) == {
        22,
        31,
        38,
        39,
        40,
        41,
        42,
        49,
        58,
    }
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": True,
        "mismatch_more_nonspecific_events": False,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_fresh_network_learned_weight_handoff_passes_match() -> None:
    artifact = yaml.safe_load(FIGURE7_PROJECTION022_DISTAL002_FRESH_MATCH_PATH.read_text())
    assert artifact["status"] == "match-pass"
    assert artifact["learned_state_handoff"] == ("fresh_network_from_figure6_weights")
    assert artifact["handoff_figure6_population_spikes"]["thalamic_relay"] == 20
    result = artifact["result"]
    assert result["learned_state_provenance"] == "simulated-learned-weight-snapshot"
    assert len(result["relay_spike_indices"]) == 20
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 559
    assert len(result["nonspecific_spike_times_ms"]) == 7
    assert set(artifact["relay_event_counts_by_index"].values()) == {4}
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_threshold_upcrossings_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_arm_transitions_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_release_transitions_by_index"]
    )
    assert all(artifact["gates"].values())
    assert artifact["assessment"]["advance_to_mismatch"]


def test_fresh_network_handoff_localizes_bottom_up_only_failure() -> None:
    artifact = yaml.safe_load(FIGURE7_PROJECTION022_DISTAL002_FRESH_PAIR_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["learned_state_handoff"] == ("fresh_network_from_figure6_weights")
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 70.0,
        "mismatch_rate_hz": 60.0,
    }
    pathway = assessment["pathway"]
    assert pathway["match_active_relay_cells"] == 5
    assert pathway["mismatch_active_relay_cells"] == 5
    assert pathway["match_trn_spikes"] == 559
    assert pathway["mismatch_trn_spikes"] == 567
    assert set(pathway["mismatch_active_relay_indices"]) == {
        22,
        31,
        40,
        49,
        58,
    }
    mismatch = artifact["mismatch_result"]
    assert {
        index: mismatch["relay_spike_indices"].count(index) for index in (22, 31, 40, 49, 58)
    } == {22: 4, 31: 4, 40: 4, 49: 4, 58: 4}
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": False,
        "mismatch_more_nonspecific_events": False,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_final_preregistered_distal_endpoint_preserves_figure6() -> None:
    artifact = yaml.safe_load(FIGURE6_PROJECTION022_DISTAL003_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_2_survivor_labels"] == ["paper_supplement_projection022_distal_0_03"]
    outcome = artifact["stage_2_outcomes"][0]
    assert outcome["population_spikes"]["thalamic_relay"] == 20
    assert outcome["population_spikes"]["trn"] == 554
    assert set(outcome["relay_event_counts_by_index"].values()) == {4}
    assert outcome["top_down_combined_horizontal_orientation_contrast"] == pytest.approx(
        0.6201263250511576
    )
    assert all(outcome["gates"].values())
    assert outcome["pass"]
    assert artifact["assessment"]["advance_to_figure7"]


def test_final_distal_endpoint_passes_clean_match() -> None:
    artifact = yaml.safe_load(FIGURE7_PROJECTION022_DISTAL003_FRESH_MATCH_PATH.read_text())
    assert artifact["status"] == "match-pass"
    assert artifact["learned_state_handoff"] == ("fresh_network_from_figure6_weights")
    result = artifact["result"]
    assert len(result["relay_spike_indices"]) == 20
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 558
    assert len(result["nonspecific_spike_times_ms"]) == 6
    assert set(artifact["relay_event_counts_by_index"].values()) == {4}
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_threshold_upcrossings_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_arm_transitions_by_index"]
    )
    assert (
        artifact["sampled_trn_event_counts_by_index"]
        == artifact["sampled_trn_release_transitions_by_index"]
    )
    assert all(artifact["gates"].values())
    assert artifact["assessment"]["advance_to_mismatch"]


def test_final_distal_endpoint_fails_complete_clean_figure7() -> None:
    artifact = yaml.safe_load(FIGURE7_PROJECTION022_DISTAL003_FRESH_PAIR_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["learned_state_handoff"] == ("fresh_network_from_figure6_weights")
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 60.0,
        "mismatch_rate_hz": 50.0,
    }
    pathway = assessment["pathway"]
    assert pathway["match_active_relay_cells"] == 5
    assert pathway["mismatch_active_relay_cells"] == 5
    assert pathway["match_trn_spikes"] == 558
    assert pathway["mismatch_trn_spikes"] == 534
    assert set(pathway["mismatch_active_relay_indices"]) == {
        22,
        31,
        40,
        49,
        58,
    }
    mismatch = artifact["mismatch_result"]
    assert {
        index: mismatch["relay_spike_indices"].count(index) for index in (22, 31, 40, 49, 58)
    } == {22: 4, 31: 4, 40: 4, 49: 4, 58: 4}
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": True,
        "mismatch_more_nonspecific_events": False,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_ring_source_recovery_preserves_identifiability_boundary() -> None:
    artifact = yaml.safe_load(KINNESS_RING_SOURCE_RECOVERY_PATH.read_text())
    assert artifact["status"] == "source-not-recovered-ring-not-identifiable"
    assert artifact["primary_release"]["payload_recovered"] is False
    assert artifact["recovered_predecessor_source"]["sha256"] == (
        "ee6f0700280ea3df8ab1be0f2b45a03a351842ba4390c275df0ec45f64f466a3"
    )
    assert artifact["recovered_kinness_example"]["sha256"] == (
        "6c3047d281f4fe432c5144748171a05b2e2ef8bcc4cdd6361c3d7612962f352a"
    )
    assert not artifact["assessment"]["exact_legacy_ring_geometry_identifiable"]
    assert not artifact["assessment"]["center_excluded_gaussian_is_officially_verified"]
    assert not artifact["assessment"]["radial_annulus_is_officially_verified"]


def test_ring_sensitivity_is_bounded_and_registered_before_figure7() -> None:
    registration = yaml.safe_load(RING_KERNEL_SENSITIVITY_REGISTRATION_PATH.read_text())
    profile = yaml.safe_load(RING_KERNEL_RADIAL_ANNULUS_PROFILE_PATH.read_text())
    assert registration["status"] == "registered-before-execution"
    assert registration["identifiability"] == "source-unresolved"
    assert set(registration["ring_family"]) == {
        "historical_control",
        "sole_new_candidate",
    }
    assert registration["ring_family"]["sole_new_candidate"]["free_parameters"] == 0
    assert profile["runtime_overrides"]["ring_kernel_convention"] == "radial_annulus"
    assert profile["locked_holdouts"][:2] == ["figure7_match", "figure7_mismatch"]


def test_radial_annulus_is_rejected_before_figure7() -> None:
    artifact = yaml.safe_load(FIGURE6_RING_KERNEL_RADIAL_ANNULUS_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["runtime_fingerprint"] == (
        "0b636a062deb0df8502bde5a941350b17e55d3dece22212163f3113c8cb2637e"
    )
    assert artifact["holdouts_consulted"] is False
    assert artifact["stage_1_survivor_labels"] == ["radial_annulus_projection022_distal_0_03"]
    assert artifact["stage_2_survivor_labels"] == []
    outcome = artifact["stage_2_outcomes"][0]
    assert outcome["population_spikes"]["thalamic_relay"] == 25
    assert outcome["population_spikes"]["trn"] == 447
    assert set(outcome["relay_event_counts_by_index"].values()) == {5}
    assert outcome["gates"] == {
        "relay_event_count": False,
        "relay_active_indices": True,
        "relay_events_per_active_index": False,
        "relay_fresh_detector_cycles_per_active_index": False,
        "feedforward_chain_complete": True,
        "causal_pair_in_learning_window": True,
        "bottom_up_horizontal_orientation": True,
        "top_down_horizontal_contrast": True,
    }
    assert not outcome["pass"]
    assert not artifact["assessment"]["advance_to_figure7"]


def test_rendered_figure7_restores_exact_numeric_target() -> None:
    artifact = yaml.safe_load(FIGURE7_RENDERED_TARGET_CORRECTION_PATH.read_text())
    assert artifact["status"] == "official-numeric-target-restored"
    assert artifact["visual_audit"]["observations"] == {
        "x_axis_ms": [0.0, 100.0],
        "match_label_hz": 40.0,
        "mismatch_label_hz": 70.0,
        "visible_match_spikes": 4,
        "visible_mismatch_spikes": 7,
    }
    assert artifact["correction"]["active_numeric_targets_hz"] == {
        "match": 40.0,
        "mismatch": 70.0,
    }
    assert artifact["correction"]["tolerance_hz"] == 0.0
    assert not artifact["rescored_leading_candidate"]["numeric_target_pass"]
    assert artifact["assessment"]["figure7_numeric_target_identifiable"]


def test_top_down_current_reopen_preserves_original_grid_and_locks_mismatch() -> None:
    registration = yaml.safe_load(FIGURE7_TOP_DOWN_CURRENT_REGISTRATION_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_TOP_DOWN_CURRENT_MATCH_PROFILE_PATH.read_text())
    assert registration["status"] == "registered-before-execution"
    grid = registration["current_grid_pA"]
    assert grid["historical_control"]["value"] == 600.0
    assert grid["unopened_candidates"] == [800.0, 1000.0]
    assert profile["protocol"]["top_down_currents_pA"] == [800.0, 1000.0]
    assert profile["match_gate"]["nonspecific_events"] == 4
    assert profile["match_gate"]["nonspecific_rate_hz"] == 40.0
    assert profile["locked_holdouts"][0] == "figure7_mismatch"


def test_top_down_current_match_screen_selects_only_800_pa() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP_DOWN_CURRENT_MATCH_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["mismatch_consulted"] is False
    assert artifact["survivor_currents_pA"] == [800.0]
    outcomes = {item["top_down_current_pA"]: item for item in artifact["outcomes"]}
    assert len(outcomes[800.0]["result"]["nonspecific_spike_times_ms"]) == 4
    assert outcomes[800.0]["pass"]
    assert len(outcomes[1000.0]["result"]["nonspecific_spike_times_ms"]) == 6
    assert not outcomes[1000.0]["pass"]
    assert artifact["assessment"]["advance_to_mismatch"]


def test_800_pa_mismatch_is_registered_with_every_pair_gate_locked() -> None:
    registration = yaml.safe_load(FIGURE7_TOP_DOWN_CURRENT_MISMATCH_REGISTRATION_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_TOP_DOWN_CURRENT_PAIR_PROFILE_PATH.read_text())
    assert registration["status"] == "registered-before-execution"
    assert registration["selected_current_pA"] == 800.0
    assert profile["selected_current_pA"] == 800.0
    assert registration["official_pair_gate"]["match_nonspecific_events"] == 4
    assert registration["official_pair_gate"]["mismatch_nonspecific_events"] == 7
    assert registration["official_pair_gate"]["nonspecific_rates_hz"] == {
        "match": 40.0,
        "mismatch": 70.0,
    }


def test_800_pa_pair_is_rejected_without_reopening_current_grid() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP_DOWN_CURRENT_PAIR_RESULT_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["selected_current_pA"] == 800.0
    assert artifact["official_assessment"]["arousal"] == {
        "match_rate_hz": 40.0,
        "mismatch_rate_hz": 50.0,
        "duration_ms": 100.0,
    }
    assert artifact["official_assessment"]["pathway"]["match_trn_spikes"] == 559
    assert artifact["official_assessment"]["pathway"]["mismatch_trn_spikes"] == 559
    assert set(artifact["official_assessment"]["pathway"]["mismatch_active_relay_indices"]) == {
        22,
        31,
        40,
        49,
        58,
    }
    assert not artifact["gates"]["mismatch_relay_overlap_only"]
    assert not artifact["gates"]["match_more_trn_events"]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    assert artifact["gates"]["sampled_mismatch_trn_events_have_fresh_cycles"]
    assert not artifact["reproduced"]


def test_one_event_current_sensitivity_is_bounded_before_match() -> None:
    registration = yaml.safe_load(FIGURE7_CURRENT_TERMINATION_REGISTRATION_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_ONE_EVENT_CURRENT_MATCH_PROFILE_PATH.read_text())
    assert registration["status"] == "registered-before-execution"
    assert registration["sole_new_candidate"] == {
        "top_down_current_mode": "until_cued_cell_first_event",
        "free_numeric_parameters": 0,
    }
    assert profile["protocol"]["top_down_currents_pA"] == [800.0]
    assert profile["protocol"]["top_down_current_mode"] == "until_cued_cell_first_event"
    assert profile["protocol"]["top_down_cue_lead_ms"] == 0.0
    assert registration["locked_holdouts"][0].startswith("figure7_mismatch")


def test_one_event_current_match_passes_and_unlocks_only_fixed_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_ONE_EVENT_CURRENT_MATCH_RESULT_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_ONE_EVENT_CURRENT_MISMATCH_REGISTRATION_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_ONE_EVENT_CURRENT_PAIR_PROFILE_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert artifact["status"] == "complete"
    assert artifact["survivor_currents_pA"] == [800.0]
    assert result["top_down_current_mode"] == "until_cued_cell_first_event"
    assert result["top_down_current_termination_time_ms"] == 5.85
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_times_ms"]) == 570
    assert len(result["nonspecific_spike_times_ms"]) == 4
    assert outcome["gates"]["top_down_current_protocol"]
    assert outcome["pass"]
    assert registration["status"] == "registered-before-execution"
    assert registration["selected_candidate"] == {
        "top_down_current_pA": 800.0,
        "top_down_current_mode": "until_cued_cell_first_event",
    }
    assert profile["selected_current_pA"] == 800.0
    assert profile["protocol"]["top_down_current_mode"] == "until_cued_cell_first_event"


def test_one_event_current_pair_is_rejected_on_spatial_and_arousal_gates() -> None:
    artifact = yaml.safe_load(FIGURE7_ONE_EVENT_CURRENT_PAIR_RESULT_PATH.read_text())
    mismatch = artifact["mismatch_result"]
    assert artifact["status"] == "figure7-failed"
    assert mismatch["top_down_current_mode"] == "until_cued_cell_first_event"
    assert mismatch["top_down_current_termination_time_ms"] == 5.85
    assert set(mismatch["relay_spike_indices"]) == {22, 31, 40, 49, 58}
    assert artifact["official_assessment"]["pathway"]["match_trn_spikes"] == 570
    assert artifact["official_assessment"]["pathway"]["mismatch_trn_spikes"] == 557
    assert artifact["official_assessment"]["arousal"]["match_rate_hz"] == 40.0
    assert artifact["official_assessment"]["arousal"]["mismatch_rate_hz"] == 40.0
    assert artifact["gates"]["mismatch_top_down_current_protocol"]
    assert artifact["gates"]["match_more_trn_events"]
    assert not artifact["gates"]["mismatch_relay_overlap_only"]
    assert not artifact["gates"]["mismatch_more_nonspecific_events"]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    assert not artifact["reproduced"]


def test_selected_category_routing_is_preregistered_as_diagnostic_only() -> None:
    registration = yaml.safe_load(FIGURE7_SELECTED_CATEGORY_REGISTRATION_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_SELECTED_CATEGORY_MATCH_PROFILE_PATH.read_text())
    assert registration["status"] == "registered-before-execution-diagnostic"
    assert registration["intervention"]["retained_source_indices"] == [40]
    assert registration["intervention"]["affected_projection_ids"] == [
        "modeldb112923.projection.003",
        "modeldb112923.projection.005",
        "modeldb112923.projection.006",
        "modeldb112923.projection.007",
    ]
    assert profile["protocol"]["top_down_relay_source_indices"] == [40]
    assert "cannot itself unlock official" in profile["next_gate"]


def test_selected_category_match_passes_and_unlocks_only_diagnostic_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_SELECTED_CATEGORY_MATCH_RESULT_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_SELECTED_CATEGORY_MISMATCH_REGISTRATION_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_SELECTED_CATEGORY_PAIR_PROFILE_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert artifact["survivor_currents_pA"] == [800.0]
    assert result["top_down_relay_source_indices"] == [40]
    assert result["top_down_current_termination_time_ms"] == 5.85
    assert len(result["relay_spike_times_ms"]) == 25
    assert len(result["trn_spike_times_ms"]) == 570
    assert len(result["nonspecific_spike_times_ms"]) == 4
    assert outcome["pass"]
    assert registration["status"] == "registered-before-execution-diagnostic"
    assert registration["interpretation_boundary"].startswith("This is a causal intervention")
    assert profile["diagnostic_only"] is True
    assert profile["protocol"]["top_down_relay_source_indices"] == [40]
    assert profile["locked_holdouts"][0] == "figure10_reset"


def test_selected_category_routing_diagnostic_is_rejected() -> None:
    artifact = yaml.safe_load(FIGURE7_SELECTED_CATEGORY_PAIR_RESULT_PATH.read_text())
    mismatch = artifact["mismatch_result"]
    assessment = artifact["official_assessment"]
    assert artifact["status"] == "figure7-failed"
    assert artifact["diagnostic_only"] is True
    assert artifact["phenotype_reproduced"] is False
    assert artifact["reproduced"] is False
    assert mismatch["top_down_relay_source_indices"] == [40]
    assert mismatch["top_down_current_termination_time_ms"] == 5.85
    assert set(mismatch["relay_spike_indices"]) == {22, 31, 40, 49, 58}
    assert assessment["pathway"]["match_trn_spikes"] == 570
    assert assessment["pathway"]["mismatch_trn_spikes"] == 555
    assert assessment["arousal"]["match_rate_hz"] == 40.0
    assert assessment["arousal"]["mismatch_rate_hz"] == 40.0
    assert artifact["gates"]["match_more_trn_events"]
    assert not artifact["gates"]["mismatch_relay_overlap_only"]
    assert not artifact["gates"]["match_more_active_relay_cells"]
    assert not artifact["gates"]["mismatch_more_nonspecific_events"]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]


def test_trn_calcium_reversal_restores_wrong_cue_lead_mechanism() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_CALCIUM_SCREEN_PATH.read_text())
    outcomes = {item["trn_calcium_source_convention"]: item for item in artifact["outcomes"]}
    reversal = outcomes["modeldb_reversal"]
    assert reversal["post_bottom_up_trn_events"] == 112
    assert len(reversal["connected_match"]["cue_lead_trn_spike_times_ms"]) == 81
    assert reversal["post_bottom_up_relay_events"] == 0
    assert not reversal["connected_causal_recruitment_pass"]
    assert artifact["assessment"]["connected_causal_survivors"] == 0


def test_trn_calcium_reversal_simultaneous_pair_is_condition_invariant() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_CALCIUM_PROFILE_PATH.read_text())
    assert (
        profile["runtime_fingerprint"]
        == runtime_conventions_for_candidate(profile["candidate"]).fingerprint
    )
    artifact = yaml.safe_load(FIGURE7_TRN_CALCIUM_PAIR_PATH.read_text())
    assert artifact["protocol"]["top_down_cue_lead_ms"] == 0.0
    assert not artifact["reproduced"]
    match = artifact["conditions"]["match"]
    mismatch = artifact["conditions"]["mismatch"]
    for condition in (match, mismatch):
        assert condition["relay_spike_indices"] == []
        assert len(condition["trn_spike_times_ms"]) == 229
        assert len(set(condition["trn_spike_indices"])) == 81
        assert len(condition["nonspecific_spike_times_ms"]) == 3
    assert match["trn_spike_times_ms"] == mismatch["trn_spike_times_ms"]
    assert artifact["assessment"]["arousal"]["match_rate_hz"] == 30.0
    assert artifact["assessment"]["arousal"]["mismatch_rate_hz"] == 30.0


def test_archived_trn_dendritic_calcium_density_closes_source_cube() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_DENDRITIC_CALCIUM_SCREEN_PATH.read_text())
    assert artifact["protocol"]["trn_dendritic_calcium_density_convention"] == ("modeldb_100")
    assert artifact["assessment"]["connected_causal_survivors"] == 0
    assert all(item["post_bottom_up_trn_events"] == 0 for item in artifact["outcomes"])
    assert all(item["sampled_trn_proximal_peak_mV"] > 85.0 for item in artifact["outcomes"])
    assert max(item["sampled_trn_soma_peak_mV"] for item in artifact["outcomes"]) < -23.0


def test_behavior_density_grid_rejects_low_endpoint_and_promotes_cue_safe_values() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_DENSITY_GRID_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_TRN_DENSITY_CUE_GRID_PATH.read_text())
    assert profile["dimension"]["grid"] == [
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        60.0,
        80.0,
        100.0,
    ]
    outcomes = {item["trn_dendritic_calcium_density_mS_cm2"]: item for item in cue["outcomes"]}
    assert outcomes[10.0]["cue_lead_trn_events"] == 81
    assert not outcomes[10.0]["stage_1_pass"]
    assert cue["stage_1_survivor_densities_mS_cm2"] == profile["dimension"]["grid"][1:]


def test_behavior_density_grid_has_no_simultaneous_match_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_DENSITY_MATCH_GRID_PATH.read_text())
    assert artifact["status"] == "no-stage-2a-survivor"
    assert artifact["stage_2a_survivor_densities_mS_cm2"] == []
    for outcome in artifact["outcomes"]:
        assert outcome["active_relay_indices"] == [38, 39, 40, 41, 42]
        assert outcome["trn_events"] == 0
        assert outcome["nonspecific_events"] == 0
        assert not outcome["stage_2a_pass"]


def test_trn_source_topology_is_linear_somatic_output_and_not_a_search_dimension() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_SOURCE_TOPOLOGY_AUDIT_PATH.read_text())
    source = artifact["sources"]["smart_nml"]
    assert source["structure_class"] == "linear"
    assert source["serialized_compartment_order"] == ["Soma", "Dendrite 0", "Dendrite 1"]
    assert source["spike_monitoring"] == {
        "Soma": True,
        "Dendrite 0": False,
        "Dendrite 1": False,
    }
    assert artifact["implementation"]["compiled_edges"] == [
        ["soma", "proximal_dendrite"],
        ["proximal_dendrite", "distal_dendrite"],
    ]
    assert artifact["implementation"]["chemical_output_compartment"] == "soma"
    assert not artifact["assessment"]["star_topology_admissible"]
    assert not artifact["assessment"]["dendritic_event_output_admissible"]
    assert not artifact["assessment"]["topology_search_authorized"]


def test_local_trn_axial_grid_is_registered_and_entirely_cue_safe() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_AXIAL_GRID_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_TRN_AXIAL_CUE_GRID_PATH.read_text())
    assert profile["dimension"]["grid"] == [
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
    ]
    assert cue["stage_1_survivor_scales"] == profile["dimension"]["grid"]
    assert all(item["stage_1_pass"] for item in cue["outcomes"])
    assert all(item["cue_lead_trn_events"] == 0 for item in cue["outcomes"])
    assert all(item["cue_lead_relay_events"] == 0 for item in cue["outcomes"])


def test_local_trn_axial_grid_loses_relay_selectivity_before_recruiting_trn() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_AXIAL_MATCH_GRID_PATH.read_text())
    assert artifact["status"] == "no-stage-2a-survivor"
    assert artifact["stage_2a_survivor_scales"] == []
    for outcome in artifact["outcomes"]:
        scale = outcome["trn_soma_proximal_axial_conductance_scale"]
        expected_relay = list(range(81)) if scale >= 4.0 else [38, 39, 40, 41, 42]
        assert outcome["active_relay_indices"] == expected_relay
        assert outcome["trn_events"] == 0
        assert not outcome["stage_2a_pass"]
    assert max(item["sampled_trn_soma_peak_mV"] for item in artifact["outcomes"]) < -23.7


def test_trn_event_offset_grid_has_a_nonmonotonic_cue_safety_boundary() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_EVENT_OFFSET_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_TRN_EVENT_OFFSET_CUE_PATH.read_text())
    assert profile["dimension"]["grid"] == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        67.0,
        69.0,
    ]
    assert cue["stage_1_survivor_offsets_mV"] == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        60.0,
        67.0,
        69.0,
    ]
    outcomes = {item["trn_spike_event_voltage_offset_mV"]: item for item in cue["outcomes"]}
    for offset in (0.0, 10.0, 20.0, 30.0, 40.0):
        assert len(outcomes[offset]["result"]["equilibration_trn_spike_times_ms"]) == 81
        assert outcomes[offset]["equilibration_tail_output_events"] == 0
    assert len(outcomes[50.0]["result"]["equilibration_trn_spike_times_ms"]) == 405
    assert outcomes[50.0]["equilibration_tail_output_events"] == 162
    assert outcomes[50.0]["cue_lead_trn_events"] == 162
    for offset in (60.0, 67.0, 69.0):
        assert outcomes[offset]["result"]["equilibration_trn_spike_times_ms"] == []


def test_trn_event_offset_grid_has_no_selective_match_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_EVENT_OFFSET_MATCH_PATH.read_text())
    assert artifact["status"] == "no-stage-2a-survivor"
    assert artifact["stage_2a_survivor_offsets_mV"] == []
    for outcome in artifact["outcomes"]:
        offset = outcome["trn_spike_event_voltage_offset_mV"]
        expected_relay = list(range(81)) if offset >= 60.0 else [38, 39, 40, 41, 42]
        assert outcome["active_relay_indices"] == expected_relay
        assert outcome["trn_events"] == 0
        assert not outcome["stage_2a_pass"]


def test_trn_density_event_offset_cross_has_no_cue_safe_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_DENSITY_EVENT_OFFSET_PROFILE_PATH.read_text())
    artifact = yaml.safe_load(FIGURE7_TRN_DENSITY_EVENT_OFFSET_CUE_PATH.read_text())
    assert profile["fixed_choices"]["trn_spike_event_voltage_offset_mV"] == 50.0
    assert profile["dimension"]["grid"] == [
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        60.0,
        80.0,
        100.0,
    ]
    assert artifact["status"] == "no-stage-1-survivor"
    assert artifact["stage_1_survivor_densities_mS_cm2"] == []
    assert [item["equilibration_trn_events"] for item in artifact["outcomes"]] == [
        162,
        162,
        162,
        243,
        243,
        324,
        324,
        405,
    ]
    assert all(item["cue_lead_trn_events"] >= 81 for item in artifact["outcomes"])
    assert all(not item["stage_1_pass"] for item in artifact["outcomes"])


def test_trn_event_blend_grid_is_cue_safe_and_has_one_match_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_EVENT_BLEND_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_TRN_EVENT_BLEND_CUE_PATH.read_text())
    match = yaml.safe_load(FIGURE7_TRN_EVENT_BLEND_MATCH_PATH.read_text())
    expected_grid = [index / 10 for index in range(11)]
    assert profile["dimension"]["grid"] == expected_grid
    assert cue["stage_1_survivor_blend_fractions"] == expected_grid
    assert match["stage_2a_survivor_blend_fractions"] == [0.5]
    survivor = next(
        item for item in match["outcomes"] if item["trn_spike_event_proximal_blend_fraction"] == 0.5
    )
    assert survivor["active_relay_indices"] == [38, 39, 40, 41, 42]
    assert survivor["relay_events"] == 5
    assert survivor["trn_events"] == 81
    assert survivor["stage_2a_pass"]


def test_trn_event_blend_short_pair_fails_mismatch_gates() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_EVENT_BLEND_PAIR_PATH.read_text())
    assert artifact["status"] == "no-stage-2b-survivor"
    assert artifact["stage_2b_survivor_blend_fractions"] == []
    outcome = artifact["outcomes"][0]
    assert outcome["trn_spike_event_proximal_blend_fraction"] == 0.5
    assert outcome["gates"] == {
        "match_relay_subset": True,
        "mismatch_relay_suppressed": False,
        "trn_match_greater_than_mismatch": False,
        "nonspecific_mismatch_greater_than_match": False,
    }
    match = outcome["conditions"]["match"]
    mismatch = outcome["conditions"]["mismatch"]
    assert match["relay_spike_indices"] == [38, 39, 40, 41, 42]
    assert mismatch["relay_spike_indices"] == [22, 31, 40, 49, 58]
    assert match["trn_spike_times_ms"] == mismatch["trn_spike_times_ms"]


def test_figure7_protocol_audit_reopens_cue_trn_and_extends_final_pair() -> None:
    audit = yaml.safe_load(FIGURE7_PROTOCOL_GATE_AUDIT_PATH.read_text())
    assert audit["status"] == "validation-gate-correction-required"
    assert audit["reopened_registered_dimension"]["grid"] == [600.0, 800.0, 1000.0]
    replacements = audit["scoring_correction"]["replace"]
    assert any("TRN events are permitted" in item["new"] for item in replacements)
    assert any("300 ms" in item["new"] for item in replacements)


def test_event_blend_top_down_current_grid_is_cue_safe_and_matches_early() -> None:
    profile = yaml.safe_load(FIGURE7_EVENT_BLEND_CURRENT_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_EVENT_BLEND_CURRENT_CUE_PATH.read_text())
    match = yaml.safe_load(FIGURE7_EVENT_BLEND_CURRENT_MATCH_PATH.read_text())
    assert profile["dimension"]["grid"] == [600.0, 800.0, 1000.0]
    assert cue["stage_1_survivor_currents_pA"] == profile["dimension"]["grid"]
    assert match["stage_2a_survivor_currents_pA"] == profile["dimension"]["grid"]
    assert [
        item["result"]["cue_lead_category_spike_times_ms"][0] for item in cue["outcomes"]
    ] == pytest.approx([8.89, 5.83, 4.47])
    assert all(item["cue_lead_trn_events"] == 0 for item in cue["outcomes"])
    assert all(item["cue_lead_relay_events"] == 0 for item in cue["outcomes"])
    for item in match["outcomes"]:
        assert item["active_relay_indices"] == [38, 39, 40, 41, 42]
        assert item["relay_events"] == 5
        assert item["trn_events"] == 81


def test_event_blend_top_down_current_grid_has_no_300ms_pair_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_EVENT_BLEND_CURRENT_PAIR_PATH.read_text())
    assert artifact["protocol"]["duration_ms"] == 300.0
    assert artifact["protocol"]["scoring_window"] == "whole_epoch"
    assert artifact["status"] == "no-stage-2b-survivor"
    assert artifact["stage_2b_survivor_currents_pA"] == []
    counts = []
    for item in artifact["outcomes"]:
        match = item["conditions"]["match"]
        mismatch = item["conditions"]["mismatch"]
        counts.append(
            (
                item["top_down_current_pA"],
                len(match["relay_spike_times_ms"]),
                len(mismatch["relay_spike_times_ms"]),
                len(match["trn_spike_times_ms"]),
                len(mismatch["trn_spike_times_ms"]),
            )
        )
        assert match["nonspecific_spike_times_ms"] == []
        assert mismatch["nonspecific_spike_times_ms"] == []
        assert not item["stage_2b_pass"]
    assert counts == [
        (600.0, 389, 432, 81, 81),
        (800.0, 441, 419, 81, 81),
        (1000.0, 418, 426, 81, 81),
    ]


def test_nonspecific_event_blend_grid_is_cue_safe_and_restores_output() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_EVENT_BLEND_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_NONSPECIFIC_EVENT_BLEND_CUE_PATH.read_text())
    mismatch = yaml.safe_load(FIGURE7_NONSPECIFIC_EVENT_BLEND_MISMATCH_PATH.read_text())
    assert profile["dimension"]["grid"] == [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    assert cue["stage_1_survivor_blend_fractions"] == profile["dimension"]["grid"]
    assert mismatch["stage_2a_survivor_blend_fractions"] == [0.3, 0.5, 0.7, 1.0]
    assert all(item["stage_1_pass"] for item in cue["outcomes"])
    events = {
        item["nonspecific_spike_event_proximal_blend_fraction"]: len(
            item["result"]["nonspecific_spike_times_ms"]
        )
        for item in mismatch["outcomes"]
    }
    assert events == {0.0: 0, 0.1: 0, 0.2: 0, 0.3: 1, 0.5: 1, 0.7: 2, 1.0: 2}


def test_nonspecific_event_blend_has_no_match_mismatch_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_EVENT_BLEND_COMPARISON_PATH.read_text())
    assert artifact["status"] == "no-stage-2b-survivor"
    assert artifact["stage_2b_survivor_blend_fractions"] == []
    expected_nonspecific_events = {0.3: 1, 0.5: 1, 0.7: 2, 1.0: 2}
    for item in artifact["outcomes"]:
        blend = item["nonspecific_spike_event_proximal_blend_fraction"]
        match = item["match"]
        mismatch = item["mismatch"]
        assert len(match["relay_spike_times_ms"]) == 20
        assert len(mismatch["relay_spike_times_ms"]) == 20
        assert len(match["trn_spike_times_ms"]) == 81
        assert len(mismatch["trn_spike_times_ms"]) == 81
        assert len(match["nonspecific_spike_times_ms"]) == expected_nonspecific_events[blend]
        assert len(mismatch["nonspecific_spike_times_ms"]) == (expected_nonspecific_events[blend])
        assert not item["stage_2b_pass"]


def test_feedback_arrival_audit_registers_only_archived_delay_landmarks() -> None:
    audit = yaml.safe_load(FIGURE7_FEEDBACK_ARRIVAL_AUDIT_PATH.read_text())
    assert audit["status"] == "protocol-timing-discrepancy-localized"
    assert audit["fixed_operating_point"] == {
        "top_down_current_pA": 800.0,
        "first_category_event_ms_after_current_onset": 5.83,
        "first_bottom_up_relay_event_ms_after_stimulus_onset": 4.1,
    }
    assert [
        item["first_receptor_arrival_ms_after_current_onset"]
        for item in audit["archived_feedback_delays"]
    ] == [7.83, 8.83, 9.83]
    assert audit["registered_diagnostic"]["top_down_cue_lead_ms"] == [
        0.0,
        7.83,
        8.83,
        9.83,
    ]


def test_feedback_arrival_alignment_has_no_early_pathway_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_FEEDBACK_ARRIVAL_GRID_PATH.read_text())
    assert artifact["status"] == "no-stage-2a-survivor"
    assert artifact["stage_2a_survivor_leads_ms"] == []
    assert [item["top_down_cue_lead_ms"] for item in artifact["outcomes"]] == [
        0.0,
        7.83,
        8.83,
        9.83,
    ]
    for item in artifact["outcomes"]:
        match = item["conditions"]["match"]
        mismatch = item["conditions"]["mismatch"]
        assert set(match["relay_spike_indices"]) == {38, 39, 40, 41, 42}
        assert set(mismatch["relay_spike_indices"]) == {22, 31, 40, 49, 58}
        assert len(match["trn_spike_times_ms"]) == 81
        assert len(mismatch["trn_spike_times_ms"]) == 81
        assert not item["stage_2a_pass"]


def test_local_trn_blend_arrival_interaction_has_no_mismatch_survivor() -> None:
    match = yaml.safe_load(FIGURE7_TRN_BLEND_ARRIVAL_MATCH_PATH.read_text())
    mismatch = yaml.safe_load(FIGURE7_TRN_BLEND_ARRIVAL_MISMATCH_PATH.read_text())
    assert match["stage_1_survivor_blend_fractions"] == [0.48, 0.49, 0.5]
    assert mismatch["status"] == "no-stage-2-survivor"
    assert mismatch["stage_2_survivor_blend_fractions"] == []
    for item in mismatch["outcomes"]:
        assert set(item["match"]["relay_spike_indices"]) == {38, 39, 40, 41, 42}
        assert set(item["mismatch"]["relay_spike_indices"]) == {
            22,
            31,
            40,
            49,
            58,
        }
        assert len(item["match"]["trn_spike_times_ms"]) == 81
        assert len(item["mismatch"]["trn_spike_times_ms"]) == 81
        assert not item["stage_2_pass"]


def test_inhibitory_arrival_alignment_never_generates_top_down_only_trn() -> None:
    artifact = yaml.safe_load(FIGURE7_INHIBITORY_ARRIVAL_PATH.read_text())
    assert artifact["status"] == "no-stage-2-survivor"
    assert artifact["stage_2_survivor_leads_ms"] == []
    assert [item["top_down_cue_lead_ms"] for item in artifact["outcomes"]] == [
        13.65,
        13.75,
    ]
    for item in artifact["outcomes"]:
        assert item["match"]["cue_lead_trn_spike_times_ms"] == []
        assert item["match"]["cue_lead_relay_spike_times_ms"] == []
        assert set(item["match"]["relay_spike_indices"]) == {
            31,
            38,
            39,
            40,
            41,
            42,
            49,
        }
        assert item["mismatch"] is None
        assert not item["stage_1_pass"]


def test_figure6_source_strength_reassessment_promotes_only_verifiable_claims() -> None:
    artifact = yaml.safe_load(FIGURE6_SOURCE_STRENGTH_REASSESSMENT_PATH.read_text())
    correction = artifact["source_strength_correction"]
    assessment = artifact["assessment"]
    assert artifact["status"] == "qualitative-figure6-reproduced"
    assert correction["absolute_map_amplitude"] == "not-identifiable"
    assert correction["historical_2_0_peak_gate"] == "retracted-unsupported"
    assert all(artifact["gates"].values())
    assert artifact["observed"]["combined_adaptive_final_peak"] == pytest.approx(0.893)
    assert assessment["qualitative_figure6_reproduced"] is True
    assert assessment["exact_absolute_amplitude_reproduced"] is None
    assert assessment["figure7_eligible_as_source_strength_prerequisite"] is True


def test_cold_network_trn_volley_releases_a_pre_stimulus_latch() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_DETECTOR_CYCLE_PATH.read_text())
    observed = artifact["observed"]
    assert artifact["interpretation"] == ("pre_stimulus_latched_arm_released_by_stimulus")
    assert observed["active_relay_indices"] == [38, 39, 40, 41, 42]
    assert observed["trn_events"] == 81
    assert observed["pre_stimulus_latched_release_inferred"] is True
    assert set(observed["threshold_upcrossings_by_index"].values()) == {0}
    assert set(observed["release_transitions_by_index"].values()) == {1}
    assert set(observed["emitted_events_by_index"].values()) == {1}
    assert (
        max(item[2] for item in observed["post_first_event_detector_voltage_range_mV_by_index"])
        < 8.0
    )


def test_same_network_candidate_has_no_evoked_trn_and_global_relay_output() -> None:
    artifact = yaml.safe_load(FIGURE7_SAME_NETWORK_DETECTOR_PATH.read_text())
    observed = artifact["observed"]
    assert artifact["interpretation"] == "unified_candidate_has_no_evoked_trn_output"
    assert observed["relay_events"] == 181
    assert observed["active_relay_indices"] == list(range(81))
    assert observed["trn_events"] == 0
    assert observed["nonspecific_events"] == 0
    assert set(observed["threshold_upcrossings_by_index"].values()) == {0}
    assert set(observed["arm_transitions_by_index"].values()) == {0}
    assert set(observed["release_transitions_by_index"].values()) == {0}
    assert max(item[2] for item in observed["detector_voltage_range_mV_by_index"]) < -9.0


def test_figure6_relay_train_has_four_genuine_detector_cycles_per_cell() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_DETECTOR_CYCLE_PATH.read_text())
    control = artifact["control"]
    training = artifact["training"]
    assessment = artifact["assessment"]
    assert control["relay_events"] == 0
    assert control["active_relay_indices"] == []
    assert set(control["threshold_upcrossings_by_index"].values()) == {0}
    assert set(control["arm_transitions_by_index"].values()) == {0}
    assert set(control["release_transitions_by_index"].values()) == {0}
    assert set(control["final_armed_by_index"].values()) == {0.0}
    for key in (
        "relay_events_by_index",
        "threshold_upcrossings_by_index",
        "arm_transitions_by_index",
        "release_transitions_by_index",
    ):
        assert set(training[key].values()) == {4}
    assert assessment == {
        "control_latched_without_release": False,
        "training_cycle_valid": True,
        "interpretation": "figure6_relay_event_train_is_detector-cycle-valid",
    }


def test_visual_spike_equation_audit_fixes_temporal_order_without_promotion() -> None:
    artifact = yaml.safe_load(SPIKE_EVENT_EQUATION_VISUAL_AUDIT_PATH.read_text())
    assert artifact["status"] == ("printed-equations-visually-verified_no-detector-promotion")
    assert artifact["printed_rules"] == {
        "common_temporal_form": (
            "delta(t)=1 when V(t)<0 and V(t-delta_t)>V_theta; otherwise delta(t)=0"
        ),
        "smart_threshold_mV": 30.0,
        "kinness_hodgkin_huxley_threshold_mV": -20.0,
    }
    implementation = artifact["implementation_assessment"]
    assert implementation["literal_previous_sample_matches_printed_temporal_form"]
    assert not implementation["legacy_spikeevents_source_body_recovered"]
    assessment = artifact["assessment"]
    assert assessment["prior_kinness_equation_number_corrected_from"] == 18
    assert assessment["prior_kinness_equation_number_corrected_to"] == 14
    assert not assessment["literal_minus20_promoted"]
    assert not assessment["official_smart_reproduced"]
    assert not assessment["downstream_holdouts_unlocked"]


def test_receptor_arrival_alignment_is_single_value_preregistered() -> None:
    profile = yaml.safe_load(FIGURE7_RECEPTOR_ALIGNMENT_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_RECEPTOR_ALIGNMENT_REGISTRATION_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["protocol"]["top_down_cue_lead_ms"] == pytest.approx(7.85)
    assert profile["protocol"]["top_down_current_mode"] == ("until_cued_cell_first_event")
    assert registration["dimension"]["registered_values"] == [7.85]
    assert registration["dimension"]["derivation"] == (
        "5.85-ms selected-category latency plus the archived 2-ms relay-NMDA delay"
    )
    assert registration["fixed_choices"]["source_delays_unchanged"]
    assert registration["mismatch_lock"].startswith("Do not run mismatch")
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_receptor_arrival_alignment_fails_match_and_keeps_mismatch_locked() -> None:
    artifact = yaml.safe_load(FIGURE7_RECEPTOR_ALIGNMENT_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["survivor_currents_pA"] == []
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert result["cue_lead_category_spike_indices"] == [40]
    assert result["cue_lead_category_spike_times_ms"] == [pytest.approx(5.85)]
    assert result["cue_lead_relay_spike_times_ms"] == []
    assert len(result["cue_lead_nonspecific_spike_times_ms"]) == 1
    assert result["top_down_current_termination_time_ms"] == pytest.approx(5.85)
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert set(outcome["relay_event_counts_by_index"].values()) == {2}
    assert len(result["trn_spike_times_ms"]) == 604
    assert len(result["nonspecific_spike_times_ms"]) == 6
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert not outcome["gates"]["nonspecific_40_hz"]
    assert not outcome["pass"]
    assert not artifact["assessment"]["advance_to_mismatch"]


def test_aligned_on_center_headroom_grid_is_preregistered_and_bounded() -> None:
    profile = yaml.safe_load(FIGURE7_ALIGNED_HEADROOM_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_ALIGNED_HEADROOM_REGISTRATION_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["dimension"]["grid"] == [0.25, 0.5, 0.75, 1.0]
    assert profile["selection_rule"].startswith("lowest headroom fraction")
    assert registration["dimension"]["projection_ids"] == [
        "modeldb112923.projection.005",
        "modeldb112923.projection.007",
    ]
    assert registration["dimension"]["selected_source_index"] == 40
    assert registration["fixed_choices"]["source_delays_unchanged"]
    assert registration["fixed_choices"]["trn_to_relay_gaba_unchanged"]
    assert registration["stage_1"]["consults"] == "figure7_match_only"
    assert registration["stage_2"]["mismatch_locked"]
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_aligned_on_center_screen_selects_only_the_hard_bound() -> None:
    artifact = yaml.safe_load(FIGURE7_ALIGNED_HEADROOM_SCREEN_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["stage_1_survivor_headroom_fractions"] == [1.0]
    assert artifact["selected_headroom_fraction"] == 1.0
    assert [item["applied_common_weight_factor"] for item in artifact["outcomes"]] == pytest.approx(
        [
            1.5915980313491906,
            2.1831960626983813,
            2.7747940940475724,
            3.366392125396763,
        ]
    )
    assert [len(item["result"]["nonspecific_spike_times_ms"]) for item in artifact["outcomes"]] == [
        6,
        6,
        6,
        4,
    ]
    selected = artifact["outcomes"][-1]
    assert set(selected["relay_event_counts_by_index"].values()) == {3}
    assert len(selected["result"]["trn_spike_times_ms"]) == 633
    assert selected["pass"]
    assert artifact["assessment"]["advance_to_diagnostic_match"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_receptor_peak_alignment_is_single_value_and_source_derived() -> None:
    profile = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_REGISTRATION_PATH.read_text())
    peak_ms = biexponential_peak_time_ms(2.0, 7.0)
    expected_lead_ms = 5.85 + 2.0 + peak_ms
    assert profile["status"] == ("registered-before-execution-source-derived-timing-diagnostic")
    assert profile["dimension"]["grid"] == [1.0]
    assert profile["protocol"]["top_down_cue_lead_ms"] == pytest.approx(expected_lead_ms)
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["source_derivation"]["receptor_peak_after_arrival_ms"] == pytest.approx(
        peak_ms
    )
    assert registration["source_derivation"]["registered_top_down_cue_lead_ms"] == pytest.approx(
        expected_lead_ms
    )
    assert registration["fixed"]["source_delays_unchanged"]
    assert registration["fixed"]["receptor_kinetics_unchanged"]
    assert registration["fixed"]["weights_fixed_before_timing_test"]
    assert registration["stopping_rule"].startswith("Run the sole match candidate")
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_receptor_peak_match_passes_and_verification_is_locked_to_it() -> None:
    screen = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_SCREEN_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_VERIFICATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_VERIFICATION_REGISTRATION_PATH.read_text())
    outcome = screen["outcomes"][0]
    result = outcome["result"]
    assert screen["stage_1_survivor_headroom_fractions"] == [1.0]
    assert outcome["pass"]
    assert result["cue_lead_relay_spike_times_ms"] == []
    assert set(outcome["relay_event_counts_by_index"].values()) == {3}
    assert len(result["trn_spike_times_ms"]) == 635
    assert len(result["nonspecific_spike_times_ms"]) == 4
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert profile["verification_screen_artifact"] == str(
        FIGURE7_RECEPTOR_PEAK_SCREEN_PATH.relative_to(ROOT)
    )
    assert profile["dimension"]["grid"] == [1.0]
    assert registration["verification"]["candidates"] == 1
    assert registration["verification"]["independently_rebuilt_network"]
    assert registration["mismatch_lock"].startswith("Do not run mismatch")


def test_receptor_peak_verification_unlocks_only_fixed_mismatch() -> None:
    verification = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_VERIFICATION_RESULT_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_MISMATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_MISMATCH_REGISTRATION_PATH.read_text())
    outcome = verification["outcomes"][0]
    result = outcome["result"]
    assert verification["assessment"]["advance_to_mismatch"]
    assert outcome["pass"]
    assert set(outcome["relay_event_counts_by_index"].values()) == {3}
    assert len(result["trn_spike_times_ms"]) == 635
    assert len(result["nonspecific_spike_times_ms"]) == 4
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert profile["match_verification_artifact"] == str(
        FIGURE7_RECEPTOR_PEAK_VERIFICATION_RESULT_PATH.relative_to(ROOT)
    )
    assert profile["protocol"]["top_down_cue_lead_ms"] == pytest.approx(11.35773631178703)
    assert registration["execution_limit"].startswith("exactly one mismatch")
    assert registration["registered_holdout"]["receptor_peak_timing_fixed"]
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_receptor_peak_pair_repairs_match_but_fails_mismatch_separation() -> None:
    artifact = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_PAIR_RESULT_PATH.read_text())
    mismatch = artifact["mismatch_result"]
    gates = artifact["gates"]
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assert len(artifact["match_scoring_summary"]["relay_spike_times_ms"]) == 15
    assert len(artifact["match_scoring_summary"]["trn_spike_times_ms"]) == 635
    assert len(artifact["match_scoring_summary"]["nonspecific_spike_times_ms"]) == 4
    assert set(mismatch["relay_spike_indices"]) == {22, 31, 40, 49, 58}
    assert all(mismatch["relay_spike_indices"].count(index) == 3 for index in (22, 31, 40, 49, 58))
    assert len(mismatch["trn_spike_times_ms"]) == 653
    assert len(mismatch["nonspecific_spike_times_ms"]) == 4
    assert gates["match_relay_spatial_pattern"]
    assert gates["match_nonspecific_40_hz"]
    assert gates["sampled_mismatch_trn_events_have_fresh_cycles"]
    assert not gates["mismatch_relay_overlap_only"]
    assert not gates["match_more_active_relay_cells"]
    assert not gates["match_more_trn_events"]
    assert not gates["mismatch_more_nonspecific_events"]
    assert not gates["mismatch_nonspecific_70_hz"]


def test_receptor_peak_gaba_interaction_reuses_closed_capacity_grid() -> None:
    profile = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_GABA_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_GABA_REGISTRATION_PATH.read_text())
    assert profile["status"] == "registered-before-execution-finite-interaction"
    assert profile["protocol"]["condition"] == "match"
    assert profile["protocol"]["top_down_cue_lead_ms"] == pytest.approx(11.35773631178703)
    assert profile["dimension"]["grid"] == [1.125, 1.25, 1.5, 2.0, 3.0]
    assert registration["registered_dimension"]["common_gains"] == profile["dimension"]["grid"]
    assert registration["registered_dimension"]["candidate_count"] == 5
    assert registration["stopping_rule"].startswith("Run all five matches")
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_receptor_peak_gaba_screen_selects_lowest_exact_match_survivor() -> None:
    screen = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_GABA_SCREEN_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_GABA_VERIFICATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_RECEPTOR_PEAK_GABA_VERIFICATION_REGISTRATION_PATH.read_text()
    )
    assert screen["match_survivor_gains"] == [1.5, 2.0, 3.0]
    assert screen["selected_gain"] == 1.5
    selected = next(outcome for outcome in screen["outcomes"] if outcome["common_gain"] == 1.5)
    assert selected["pass"]
    assert selected["relay_active_indices"] == [38, 39, 40, 41, 42]
    assert selected["relay_events"] == 10
    assert selected["trn_events"] == 584
    assert selected["nonspecific_events"] == 4
    assert profile["dimension"]["grid"] == [1.5]
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["verification"]["candidates"] == 1
    assert registration["mismatch_lock"].startswith("Do not run mismatch")


def test_receptor_peak_gaba_verification_unlocks_only_gain_1_5_mismatch() -> None:
    verification = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_GABA_VERIFICATION_RESULT_PATH.read_text())
    profile = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_GABA_MISMATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_GABA_MISMATCH_REGISTRATION_PATH.read_text())
    outcome = verification["outcomes"][0]
    assert outcome["pass"]
    assert outcome["common_gain"] == 1.5
    assert outcome["relay_events"] == 10
    assert outcome["trn_events"] == 584
    assert outcome["nonspecific_events"] == 4
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert verification["assessment"]["advance_to_mismatch"]
    assert profile["recognition_trn_to_relay_common_gain"]["value"] == 1.5
    assert profile["recognition_trn_to_relay_common_gain"]["training_scales_unchanged"]
    assert registration["execution_limit"] == "exactly one mismatch at gain 1.5"
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_receptor_peak_gaba_pair_is_condition_invariant_and_closes_family() -> None:
    artifact = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_GABA_PAIR_RESULT_PATH.read_text())
    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    gates = artifact["gates"]
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assert len(match["relay_spike_times_ms"]) == len(mismatch["relay_spike_times_ms"]) == 10
    assert len(match["trn_spike_times_ms"]) == len(mismatch["trn_spike_times_ms"]) == 584
    assert (
        len(match["nonspecific_spike_times_ms"]) == len(mismatch["nonspecific_spike_times_ms"]) == 4
    )
    assert set(mismatch["relay_spike_indices"]) == {22, 31, 40, 49, 58}
    assert all(mismatch["relay_spike_indices"].count(index) == 2 for index in (22, 31, 40, 49, 58))
    assert gates["sampled_mismatch_trn_events_have_fresh_cycles"]
    assert not gates["mismatch_relay_overlap_only"]
    assert not gates["match_more_active_relay_cells"]
    assert not gates["match_more_trn_events"]
    assert not gates["mismatch_more_nonspecific_events"]
    assert not gates["mismatch_nonspecific_70_hz"]


def test_receptor_peak_annulus_is_one_parameter_free_match_endpoint() -> None:
    profile = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_ANNULUS_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_ANNULUS_REGISTRATION_PATH.read_text())
    assert (
        profile["runtime_overrides"]["corticoreticular_ring_kernel_convention"] == "radial_annulus"
    )
    assert profile["learned_state"]["selected_headroom_fraction"] == 1.0
    assert profile["dimension"]["grid"] == [1.0]
    assert profile["protocol"]["top_down_cue_lead_ms"] == pytest.approx(11.35773631178703)
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["fixed"]["corticoreticular_common_gain"] == 1.0
    assert registration["source_boundary"].startswith(
        "Radial annulus is reconstruction calibration"
    )
    assert registration["stopping_rule"].startswith("Run one full-detector match")


def test_receptor_peak_annulus_fails_match_rate_and_locks_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_ANNULUS_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert artifact["match_survivors"] == []
    assert artifact["selected_candidate"] is None
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["relay_spike_times_ms"]) == 10
    assert len(result["trn_spike_times_ms"]) == 607
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert not outcome["gates"]["nonspecific_events"]
    assert not outcome["gates"]["nonspecific_40_hz"]
    assert not outcome["pass"]


def test_adjacent_annulus_radius_is_geometry_derived_and_single_value() -> None:
    profile = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_ADJACENT_ANNULUS_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_RECEPTOR_PEAK_ADJACENT_ANNULUS_REGISTRATION_PATH.read_text()
    )
    expected_scale = 1 / (2**0.5 * 1.5)
    assert (
        profile["runtime_overrides"]["corticoreticular_ring_kernel_convention"] == "radial_annulus"
    )
    assert profile["runtime_overrides"]["corticoreticular_ring_peak_radius_scale"] == pytest.approx(
        expected_scale
    )
    assert profile["dimension"]["grid"] == [1.0]
    assert registration["fixed"]["target_peak_lattice_distance"] == 1.0
    assert registration["fixed"]["ring_peak_radius_scale"] == pytest.approx(expected_scale)
    assert registration["fixed"]["sensory_input_unchanged"]
    assert registration["stopping_rule"].startswith("Run one full-detector match")


def test_adjacent_annulus_fails_figure6_before_holdout_consultation() -> None:
    artifact = yaml.safe_load(FIGURE7_RECEPTOR_PEAK_ADJACENT_ANNULUS_RESULT_PATH.read_text())
    assert artifact["status"] == "figure6-prerequisite-failed"
    assert artifact["holdouts_consulted"] == ["figure6_prerequisite"]
    assert not artifact["mismatch_consulted"]
    assert artifact["handoff_figure6_population_spikes"] == {
        "thalamic_relay": 25,
        "layer6ii_excitatory_v1": 12,
        "layer4_excitatory_v1": 36,
    }
    assert artifact["assessment"]["figure6_relay_target"] == 20
    assert not artifact["assessment"]["figure6_relay_pass"]
    assert not artifact["assessment"]["advance_to_full_state_match_verification"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_learned_comparator_floor_is_preregistered_as_reconstruction() -> None:
    profile = yaml.safe_load(FIGURE7_LEARNED_COMPARATOR_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_LEARNED_COMPARATOR_REGISTRATION_PATH.read_text())
    assert profile["status"] == ("registered-before-execution-calibrated-reconstruction")
    assert profile["dimension"]["grid"] == [0.0, 0.25, 0.5, 0.75]
    assert profile["dimension"]["control_floor"] == 1.0
    assert profile["dimension"]["selection_rule"].startswith("highest")
    assert registration["registered_dimension"]["candidate_count"] == 4
    assert registration["scope_boundary"]["classification"] == (
        "calibrated-reconstruction-not-recovered-source"
    )
    assert registration["scope_boundary"]["affects"].startswith("active bottom-up relay pixels")
    assert (
        "Figure 6 training inputs and dynamics"
        in registration["scope_boundary"]["leaves_unchanged"]
    )
    assert registration["stopping_rule"].startswith("Run all four match-only")
    assert registration["locked_holdouts"][0] == "figure7_mismatch"


def test_smooth_learned_comparator_floor_fails_match_before_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_LEARNED_COMPARATOR_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["classification"] == ("calibrated-reconstruction-not-recovered-source")
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    assert not artifact["mismatch_consulted"]
    assert artifact["handoff_figure6_population_spikes"]["thalamic_relay"] == 20
    assert artifact["match_survivor_floors"] == []
    assert artifact["selected_comparator_floor"] is None
    expected = {
        0.0: (6, {39, 40, 41}, 573, 6),
        0.25: (6, {39, 40, 41}, 573, 6),
        0.5: (8, {38, 39, 40, 41, 42}, 609, 6),
        0.75: (13, {38, 39, 40, 41, 42}, 616, 5),
    }
    for outcome in artifact["outcomes"]:
        result = outcome["result"]
        relay_count, relay_indices, trn_count, nonspecific_count = expected[
            outcome["comparator_floor"]
        ]
        assert len(result["relay_spike_times_ms"]) == relay_count
        assert set(result["relay_spike_indices"]) == relay_indices
        assert len(result["trn_spike_times_ms"]) == trn_count
        assert len(result["nonspecific_spike_times_ms"]) == nonspecific_count
        assert not outcome["pass"]
    assert not artifact["assessment"]["advance_to_independent_match_verification"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_half_max_comparator_is_one_parameter_free_registered_candidate() -> None:
    profile = yaml.safe_load(FIGURE7_HALF_MAX_COMPARATOR_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_HALF_MAX_COMPARATOR_REGISTRATION_PATH.read_text())
    assert profile["dimension"] == {
        "name": "learned_expectation_half_max_saturated_gate",
        "kind": "half_max_binary",
        "grid": [0.5],
        "candidate_count": 1,
        "transform": profile["dimension"]["transform"],
    }
    assert registration["registered_candidate"] == {
        "transform": "half_max_binary",
        "support_threshold": 0.5,
        "candidate_count": 1,
        "gain_above_or_equal_threshold": 1.0,
        "gain_below_threshold": 0.0,
    }
    assert registration["scope_boundary"]["classification"] == (
        "calibrated-reconstruction-not-recovered-source"
    )
    assert registration["stopping_rule"].startswith("Run the sole half-max")
    assert registration["locked_holdouts"][0] == "figure7_mismatch"


def test_half_max_comparator_fails_match_without_threshold_fitting() -> None:
    artifact = yaml.safe_load(FIGURE7_HALF_MAX_COMPARATOR_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert artifact["status"] == "complete"
    assert artifact["classification"] == ("calibrated-reconstruction-not-recovered-source")
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    assert not artifact["mismatch_consulted"]
    assert outcome["support_threshold"] == 0.5
    assert result["comparator_transform"] == "half_max_binary"
    assert len(result["relay_spike_times_ms"]) == 9
    assert set(result["relay_spike_indices"]) == {39, 40, 41}
    assert outcome["relay_event_counts_by_index"] == {
        "38": 0,
        "39": 3,
        "40": 3,
        "41": 3,
        "42": 0,
    }
    assert len(result["trn_spike_times_ms"]) == 607
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert not outcome["pass"]
    assert artifact["match_survivor_thresholds"] == []
    assert artifact["selected_support_threshold"] is None
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_top5_comparator_cardinality_is_derived_before_execution() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_COMPARATOR_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TOP5_COMPARATOR_REGISTRATION_PATH.read_text())
    assert profile["dimension"]["kind"] == "top_k_binary"
    assert profile["dimension"]["grid"] == [5]
    assert profile["dimension"]["candidate_count"] == 1
    assert profile["dimension"]["derivation"] == (
        "archived training stimulus has exactly five active sensory pixels"
    )
    assert registration["registered_candidate"] == {
        "transform": "top_k_binary",
        "target_count": 5,
        "candidate_count": 1,
        "selected_target_gain": 1.0,
        "unselected_target_gain": 0.0,
        "tie_break": "ascending_flat_sheet_index",
    }
    assert registration["scope_boundary"]["classification"] == (
        "calibrated-reconstruction-not-recovered-source"
    )
    assert registration["stopping_rule"].startswith("Run the sole top-five")
    assert registration["locked_holdouts"][0] == "figure7_mismatch"


def test_top5_comparator_is_an_exact_match_screen_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_COMPARATOR_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert artifact["classification"] == ("calibrated-reconstruction-not-recovered-source")
    assert artifact["match_survivor_target_counts"] == [5]
    assert artifact["selected_target_count"] == 5
    assert outcome["target_count"] == 5
    assert result["comparator_transform"] == "top_k_binary"
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert set(outcome["relay_event_counts_by_index"].values()) == {3}
    assert len(result["relay_spike_times_ms"]) == 15
    assert len(result["trn_spike_times_ms"]) == 635
    assert len(result["nonspecific_spike_times_ms"]) == 4
    assert outcome["pass"]
    assert artifact["assessment"]["advance_to_independent_match_verification"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_top5_verification_registers_only_the_screen_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_COMPARATOR_VERIFICATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_TOP5_COMPARATOR_VERIFICATION_REGISTRATION_PATH.read_text()
    )
    assert profile["dimension"]["grid"] == [5]
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["screen_result"] == {
        "selected_target_count": 5,
        "relay_active_indices": [38, 39, 40, 41, 42],
        "relay_events": 15,
        "trn_events": 635,
        "nonspecific_events": 4,
        "nonspecific_rate_hz": 40.0,
        "all_match_gates_passed": True,
    }
    assert registration["verification"]["candidates"] == 1
    assert registration["verification"]["independently_rebuilt_network"]
    assert registration["verification"]["full_detector_cycle_diagnostics"]
    assert registration["mismatch_lock"].startswith("Do not run mismatch")


def test_top5_match_verifies_with_complete_sampled_detector_cycles() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_COMPARATOR_VERIFICATION_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert outcome["pass"]
    assert len(result["relay_spike_times_ms"]) == 15
    assert len(result["trn_spike_times_ms"]) == 635
    assert len(result["nonspecific_spike_times_ms"]) == 4
    assert set(outcome["relay_event_counts_by_index"].values()) == {3}
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert (
        outcome["sampled_trn_event_counts_by_index"]
        == outcome["sampled_trn_threshold_upcrossings_by_index"]
    )
    assert (
        outcome["sampled_trn_event_counts_by_index"]
        == outcome["sampled_trn_arm_transitions_by_index"]
    )
    assert (
        outcome["sampled_trn_event_counts_by_index"]
        == outcome["sampled_trn_release_transitions_by_index"]
    )
    assert artifact["assessment"]["advance_to_mismatch"]
    assert not artifact["assessment"]["mismatch_remains_locked"]


def test_top5_mismatch_is_fixed_only_after_verified_match() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_COMPARATOR_MISMATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TOP5_COMPARATOR_MISMATCH_REGISTRATION_PATH.read_text())
    assert profile["comparator"] == {
        "transform": "top_k_binary",
        "target_count": 5,
    }
    assert registration["authorization"]["independently_verified_match"]
    assert registration["authorization"]["match_counts"] == {
        "relay_events": 15,
        "trn_events": 635,
        "nonspecific_events": 4,
    }
    assert registration["authorization"]["sampled_match_trn_detector_cycles_complete"]
    assert registration["execution"]["condition"] == "mismatch"
    assert registration["execution"]["run_count"] == 1
    assert registration["stopping_rule"].startswith("Run one fixed vertical")


def test_top5_pair_recovers_pathway_but_not_70hz_arousal() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_COMPARATOR_PAIR_RESULT_PATH.read_text())
    mismatch = artifact["mismatch_result"]
    assert artifact["status"] == "figure7-failed"
    assert artifact["classification"] == ("calibrated-reconstruction-not-recovered-source")
    assert not artifact["reproduced"]
    assert set(mismatch["relay_spike_indices"]) == {40}
    assert len(mismatch["relay_spike_times_ms"]) == 3
    assert len(mismatch["trn_spike_times_ms"]) == 560
    assert len(mismatch["nonspecific_spike_times_ms"]) == 5
    assert artifact["assessment"]["arousal"]["mismatch_rate_hz"] == 50.0
    for passed_gate in (
        "match_relay_spatial_set",
        "mismatch_relay_overlap_only",
        "match_more_active_relay_cells",
        "match_more_trn_events",
        "match_nonspecific_40_hz",
        "mismatch_more_nonspecific_events",
        "figure7_target_duration",
        "sampled_mismatch_trn_events_have_fresh_cycles",
    ):
        assert artifact["gates"][passed_gate]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    assert (
        artifact["sampled_mismatch_trn_event_counts_by_index"]
        == artifact["sampled_mismatch_trn_threshold_upcrossings_by_index"]
    )
    assert (
        artifact["sampled_mismatch_trn_event_counts_by_index"]
        == artifact["sampled_mismatch_trn_arm_transitions_by_index"]
    )
    assert (
        artifact["sampled_mismatch_trn_event_counts_by_index"]
        == artifact["sampled_mismatch_trn_release_transitions_by_index"]
    )


def test_top5_nonspecific_transfer_reuses_predeclared_finite_grid() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_BLEND_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_BLEND_REGISTRATION_PATH.read_text())
    prior = yaml.safe_load((ROOT / profile["prior_transfer_profile"]).read_text())
    assert profile["dimension"]["grid"] == [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    assert profile["dimension"]["baseline_control"] == 0.0
    assert [0.0, *profile["dimension"]["grid"]] == prior["dimension"]["grid"]
    assert profile["dimension"]["selection_rule"].startswith("highest")
    assert registration["registered_dimension"]["candidate_count"] == 6
    assert registration["fixed"]["comparator_transform"] == "top_k_binary"
    assert registration["fixed"]["comparator_target_count"] == 5
    assert registration["scope_boundary"]["changes"].startswith(
        "nonspecific output event coordinate"
    )
    assert registration["stopping_rule"].startswith("Run all six positive")
    assert registration["locked_holdouts"][0] == ("figure7_mismatch_with_output_transfer")


def test_shared_arm_release_blend_has_no_exact_match_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_BLEND_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    assert not artifact["mismatch_with_output_transfer_consulted"]
    assert artifact["match_survivor_blend_fractions"] == []
    assert artifact["selected_nonspecific_event_blend_fraction"] is None
    expected_nonspecific = {
        0.1: 0,
        0.2: 0,
        0.3: 0,
        0.5: 0,
        0.7: 0,
        1.0: 1,
    }
    for outcome in artifact["outcomes"]:
        result = outcome["result"]
        assert len(result["relay_spike_times_ms"]) == 15
        assert len(result["trn_spike_times_ms"]) == 635
        assert (
            len(result["nonspecific_spike_times_ms"])
            == expected_nonspecific[outcome["nonspecific_event_blend_fraction"]]
        )
        assert outcome["gates"]["relay_events"]
        assert outcome["gates"]["trn_events"]
        assert not outcome["gates"]["nonspecific_events"]
        assert not outcome["pass"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_split_nonspecific_detector_reuses_arm_grid_and_somatic_release() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_SPLIT_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_SPLIT_REGISTRATION_PATH.read_text())
    assert profile["dimension"]["grid"] == [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    assert profile["dimension"]["release_proximal_blend_fraction"] == 0.0
    assert profile["dimension"]["selection_rule"].startswith("highest")
    assert registration["registered_dimension"] == {
        "arm_proximal_blend_fractions": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        "release_proximal_blend_fraction": 0.0,
        "baseline_arm_fraction": 0.0,
        "candidate_count": 6,
        "selection_rule": "highest exact-match survivor",
    }
    assert registration["scope_boundary"]["changes"].startswith(
        "nonspecific event arming coordinate"
    )
    assert registration["scope_boundary"]["preserves"] == ("somatic release/rearming coordinate")
    assert registration["stopping_rule"].startswith("Run all six registered")


def test_split_nonspecific_detector_has_no_exact_match_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_SPLIT_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["release_proximal_blend_fraction"] == 0.0
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    assert not artifact["mismatch_with_output_transfer_consulted"]
    assert artifact["match_survivor_blend_fractions"] == []
    assert artifact["selected_nonspecific_event_blend_fraction"] is None
    expected_nonspecific = {
        0.1: 0,
        0.2: 0,
        0.3: 0,
        0.5: 0,
        0.7: 0,
        1.0: 100,
    }
    for outcome in artifact["outcomes"]:
        result = outcome["result"]
        assert len(result["relay_spike_times_ms"]) == 15
        assert len(result["trn_spike_times_ms"]) == 635
        assert (
            len(result["nonspecific_spike_times_ms"])
            == expected_nonspecific[outcome["nonspecific_event_blend_fraction"]]
        )
        assert outcome["gates"]["relay_events"]
        assert outcome["gates"]["trn_events"]
        assert not outcome["gates"]["nonspecific_events"]
        assert not outcome["pass"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_projection049_supplement_tuple_is_registered_as_discrete_source_choice() -> None:
    profile = yaml.safe_load(FIGURE6_NONSPECIFIC_DISTAL_GABA_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE6_NONSPECIFIC_DISTAL_GABA_REGISTRATION_PATH.read_text())
    assert profile["source_conflict"]["projection_id"] == ("modeldb112923.projection.049")
    assert profile["source_conflict"]["modeldb_serialized"] == {
        "channel_conductance_mS_cm2": 1.461,
        "rise_fall_ms": [1.0, 4.0],
    }
    assert profile["source_conflict"]["paper_supplement"] == {
        "channel_conductance_mS_cm2": 1.5,
        "rise_fall_ms": [1.0, 7.0],
    }
    assert (
        profile["runtime_overrides"]["nonspecific_distal_gaba_source_convention"]
        == "paper_supplement_1p5_1_7"
    )
    assert registration["resolution_policy"]["historical_default_unchanged"]
    assert registration["resolution_policy"]["new_runtime_fingerprint_required"]
    assert registration["resolution_policy"]["changes_only_projection_049"]
    assert registration["stopping_rule"].startswith("Run one full Figure 6")


def test_projection049_supplement_tuple_fails_figure6_prerequisite() -> None:
    artifact = yaml.safe_load(FIGURE6_NONSPECIFIC_DISTAL_GABA_RESULT_PATH.read_text())
    assert artifact["status"] == "figure6-failed"
    assert not artifact["pass"]
    assert artifact["runtime_fingerprint"] == artifact["result"]["convention_fingerprint"]
    assert artifact["result"]["population_spikes"] == {
        "thalamic_relay": 104,
        "layer6ii_excitatory_v1": 12,
        "layer4_excitatory_v1": 100,
    }
    assert artifact["relay_event_counts_by_index"] == {
        "38": 5,
        "39": 6,
        "40": 6,
        "41": 6,
        "42": 5,
    }
    assert not artifact["gates"]["relay_active_indices"]
    assert not artifact["gates"]["relay_events_per_active_index"]
    assert not artifact["gates"]["relay_events"]
    assert not artifact["assessment"]["cortical_chain_gate_valid"]
    assert artifact["assessment"]["decisive_failure"].startswith("relay recruitment")
    assert artifact["gates"]["causal_learning_pair"]
    assert artifact["gates"]["top_down_horizontal_contrast"]
    assert not artifact["assessment"]["advance_to_figure7_match"]
    assert artifact["assessment"]["figure7_holdouts_remain_locked"]


def test_top5_nonspecific_gaba_transfer_is_preregistered_match_first() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_REGISTRATION_PATH.read_text()
    )
    expected_ids = [
        "modeldb112923.projection.047",
        "modeldb112923.projection.048",
        "modeldb112923.projection.049",
    ]
    assert profile["nonspecific_gaba_transfer"]["projection_ids"] == expected_ids
    assert profile["nonspecific_gaba_transfer"]["common_scale_grid"] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert profile["nonspecific_gaba_transfer"]["selection_rule"].startswith("weakest")
    assert registration["registered_dimension"]["projection_ids"] == expected_ids
    assert registration["registered_dimension"]["common_scales"] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert registration["fixed"]["relative_projection_weights_unchanged"]
    assert registration["scope_boundary"]["applies_during"] == [
        "Figure 6 learning",
        "Figure 7 recognition",
    ]
    assert registration["stopping_rule"].startswith("For each registered scale")


def test_top5_nonspecific_gaba_transfer_selects_weakest_match_survivor() -> None:
    superseded = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_SUPERSEDED_PATH.read_text())
    artifact = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_RESULT_PATH.read_text())
    assert superseded["status"] == "superseded-runner-monitor-omission"
    assert superseded["superseded_by"] == str(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_RESULT_PATH.relative_to(ROOT)
    )
    assert artifact["status"] == "complete"
    assert not artifact["mismatch_consulted"]
    assert artifact["match_survivor_common_scales"] == [0.75, 1.0]
    assert artifact["selected_common_scale"] == 0.75
    expected_nonspecific = {0.25: 2, 0.5: 2, 0.75: 4, 1.0: 4}
    for outcome in artifact["outcomes"]:
        assert outcome["figure6_pass"]
        assert all(outcome["figure6_gates"].values())
        result = outcome["match_result"]
        assert len(result["relay_spike_times_ms"]) == 15
        assert len(result["trn_spike_times_ms"]) == 635
        assert (
            len(result["nonspecific_spike_times_ms"])
            == (expected_nonspecific[outcome["common_scale"]])
        )
        assert outcome["match_gates"]["sampled_trn_events_have_fresh_cycles"]
        assert outcome["pass"] == (outcome["common_scale"] >= 0.75)
    assert artifact["assessment"]["advance_to_independent_match_verification"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_top5_nonspecific_gaba_transfer_verification_is_fixed_to_screen() -> None:
    profile = yaml.safe_load(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_VERIFICATION_PROFILE_PATH.read_text()
    )
    registration = yaml.safe_load(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_VERIFICATION_REGISTRATION_PATH.read_text()
    )
    assert profile["verification_screen_artifact"] == str(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_RESULT_PATH.relative_to(ROOT)
    )
    assert profile["nonspecific_gaba_transfer"]["common_scale_grid"] == [0.75]
    assert registration["screen_result"]["selected_common_scale"] == 0.75
    assert registration["fixed_candidate"]["common_scale"] == 0.75
    assert registration["verification"]["candidate_count"] == 1
    assert registration["verification"]["complete_figure6_cortical_monitoring"]
    assert registration["stopping_rule"].startswith("Run one fresh Figure 6")


def test_top5_nonspecific_gaba_transfer_verification_reproduces_match() -> None:
    artifact = yaml.safe_load(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_VERIFICATION_RESULT_PATH.read_text()
    )
    assert artifact["status"] == "complete"
    assert artifact["match_survivor_common_scales"] == [0.75]
    assert artifact["selected_common_scale"] == 0.75
    assert not artifact["mismatch_consulted"]
    outcome = artifact["outcomes"][0]
    assert outcome["figure6_pass"]
    assert all(outcome["figure6_gates"].values())
    assert outcome["match_pass"]
    assert all(outcome["match_gates"].values())
    result = outcome["match_result"]
    assert len(result["relay_spike_times_ms"]) == 15
    assert len(result["trn_spike_times_ms"]) == 635
    assert len(result["nonspecific_spike_times_ms"]) == 4
    for index, event_count in outcome["sampled_match_trn_event_counts_by_index"].items():
        assert event_count == outcome["sampled_match_trn_threshold_upcrossings_by_index"][index]
        assert event_count == outcome["sampled_match_trn_arm_transitions_by_index"][index]
        assert event_count == outcome["sampled_match_trn_release_transitions_by_index"][index]
    assert artifact["assessment"]["advance_to_mismatch"]
    assert not artifact["assessment"]["mismatch_remains_locked"]


def test_top5_nonspecific_gaba_transfer_mismatch_is_fixed_to_verification() -> None:
    profile = yaml.safe_load(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_MISMATCH_PROFILE_PATH.read_text()
    )
    registration = yaml.safe_load(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_MISMATCH_REGISTRATION_PATH.read_text()
    )
    assert profile["match_verification_artifact"] == str(
        FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_VERIFICATION_RESULT_PATH.relative_to(ROOT)
    )
    assert profile["nonspecific_gaba_transfer"]["common_scale"] == 0.75
    assert profile["official_gates"]["mismatch_nonspecific_events"] == 7
    assert registration["authorization"]["selected_common_scale"] == 0.75
    assert registration["execution"]["run_count"] == 1
    assert registration["execution"]["fresh_complete_figure6_handoff"]
    assert registration["stopping_rule"].startswith("Run exactly one fresh vertical mismatch")


def test_top5_nonspecific_gaba_transfer_pair_fails_arousal_only() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_NONSPECIFIC_GABA_TRANSFER_PAIR_RESULT_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["nonspecific_gaba_common_scale"] == 0.75
    assert all(artifact["figure6_gates"].values())
    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert mismatch["relay_spike_indices"] == [40, 40, 40]
    assert len(mismatch["trn_spike_times_ms"]) == 560
    assert len(mismatch["nonspecific_spike_times_ms"]) == 4
    assert artifact["gates"]["mismatch_relay_overlap_only"]
    assert artifact["gates"]["match_more_trn_events"]
    assert not artifact["gates"]["mismatch_more_nonspecific_events"]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    assert artifact["gates"]["sampled_mismatch_trn_events_have_fresh_cycles"]
    for index, event_count in artifact["sampled_mismatch_trn_event_counts_by_index"].items():
        assert event_count == artifact["sampled_mismatch_trn_threshold_upcrossings_by_index"][index]
        assert event_count == artifact["sampled_mismatch_trn_arm_transitions_by_index"][index]
        assert event_count == artifact["sampled_mismatch_trn_release_transitions_by_index"][index]
    assert not artifact["reproduced"]


def test_methods_global_calcium_is_registered_as_literal_source_choice() -> None:
    profile = yaml.safe_load(FIGURE6_METHODS_GLOBAL_CALCIUM_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE6_METHODS_GLOBAL_CALCIUM_REGISTRATION_PATH.read_text())
    assert profile["runtime_overrides"]["calcium_density_convention"] == ("methods_global_250")
    assert profile["protocol"]["complete_cortical_monitoring"]
    assert registration["resolution_policy"]["literal_global_application"]
    assert registration["resolution_policy"]["add_no_new_calcium_channels"]
    assert registration["resolution_policy"]["historical_default_unchanged"]
    assert registration["fixed"]["all_projection_and_protocol_parameters_unchanged"]
    assert registration["stopping_rule"].startswith("Run one complete, fully monitored Figure 6")


def test_methods_global_calcium_fails_figure6_selectivity() -> None:
    artifact = yaml.safe_load(FIGURE6_METHODS_GLOBAL_CALCIUM_RESULT_PATH.read_text())
    assert artifact["status"] == "figure6-failed"
    assert not artifact["pass"]
    assert artifact["runtime_fingerprint"] == artifact["result"]["convention_fingerprint"]
    assert artifact["result"]["population_spikes"] == {
        "thalamic_relay": 81,
        "layer5_excitatory_v1": 162,
        "layer6ii_excitatory_v1": 83,
        "layer6i_excitatory_v1": 162,
        "layer23_excitatory_v1": 81,
        "layer4_excitatory_v1": 243,
    }
    assert len(artifact["relay_event_counts_by_index"]) == 81
    assert set(artifact["relay_event_counts_by_index"].values()) == {1}
    assert not artifact["gates"]["relay_active_indices"]
    assert not artifact["gates"]["relay_events_per_active_index"]
    assert not artifact["gates"]["relay_events"]
    assert artifact["gates"]["cortical_chain_complete"]
    assert not artifact["gates"]["causal_learning_pair"]
    assert not artifact["gates"]["top_down_horizontal_contrast"]
    assert not artifact["assessment"]["advance_to_figure7_match"]
    assert artifact["assessment"]["figure7_holdouts_remain_locked"]


def test_nonspecific_voltage_audit_is_registered_as_read_only() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_VOLTAGE_AUDIT_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_NONSPECIFIC_VOLTAGE_AUDIT_REGISTRATION_PATH.read_text())
    assert profile["diagnostic"]["parameter_changes"] == "none"
    assert profile["protocol"]["conditions"] == ["match", "mismatch"]
    assert profile["diagnostic"]["positive_local_maximum_floor_mV"] == 0.0
    assert profile["diagnostic"]["detector_threshold_upcrossing_mV"] == 30.0
    assert profile["diagnostic"]["detector_recovery_downcrossing_mV"] == 0.0
    assert registration["execution"]["parameter_changes"] == "none"
    assert registration["execution"]["run_count_per_condition"] == 1
    assert registration["stopping_rule"].startswith(
        "Run exactly one fixed match and one fixed mismatch"
    )


def test_nonspecific_voltage_audit_closes_shared_detector_threshold() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_VOLTAGE_AUDIT_RESULT_PATH.read_text())
    match = artifact["match_diagnostic"]
    mismatch = artifact["mismatch_diagnostic"]
    assert artifact["status"] == "complete-read-only-diagnostic"
    assert artifact["parameter_changes"] == "none"
    assert match["nonspecific_event_count"] == 4
    assert mismatch["nonspecific_event_count"] == 5
    assert match["positive_soma_local_maximum_count"] == 22
    assert mismatch["positive_soma_local_maximum_count"] == 23
    for condition in (match, mismatch):
        assert condition["nonspecific_event_count"] == condition["detector_threshold_upcrossings"]
        assert condition["nonspecific_event_count"] == condition["detector_arm_transitions"]
        assert condition["nonspecific_event_count"] == condition["detector_release_transitions"]
        assert condition["detector_final_armed"] == 0.0
    threshold = artifact["derived_shared_soma_threshold_analysis"]
    assert threshold["current_threshold_peak_counts"] == {
        "match": 4,
        "mismatch": 5,
    }
    assert not threshold["exact_shared_threshold_exists"]
    assert threshold["maximum_mismatch_peaks_while_preserving_match_count"] == 5
    assert not artifact["reproduced"]


def test_nonspecific_peak_current_audit_is_read_only_and_complete() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_PEAK_CURRENT_AUDIT_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_PEAK_CURRENT_AUDIT_REGISTRATION_PATH.read_text()
    )
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_PEAK_CURRENT_AUDIT_RESULT_PATH.read_text())
    assert profile["diagnostic"]["parameter_changes"] == "none"
    assert registration["execution"]["parameter_changes"] == "none"
    assert registration["execution"]["run_count_per_condition"] == 1
    assert artifact["status"] == "complete-read-only-diagnostic"
    assert artifact["parameter_changes"] == "none"
    assert artifact["runtime_fingerprint"] == (
        "0aa301be0a82a34b0a3337030eedb66dbc33e0ec0532ead902638088a434b4d9"
    )
    match = artifact["match_result"]
    mismatch = artifact["mismatch_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert len(mismatch["relay_spike_times_ms"]) == 3
    assert set(mismatch["relay_spike_indices"]) == {40}
    assert len(mismatch["trn_spike_times_ms"]) == 560
    assert len(mismatch["nonspecific_spike_times_ms"]) == 5
    current_source_count = len(profile["diagnostic"]["current_readouts"])
    gate_source_count = len(profile["diagnostic"]["gate_readouts"])
    for condition in (match, mismatch):
        peak_count = len(condition["nonspecific_positive_soma_local_maxima_ms_mV"])
        assert len(condition["nonspecific_peak_current_samples_pA"]) == (
            peak_count * current_source_count
        )
        assert len(condition["nonspecific_peak_gate_samples"]) == (peak_count * gate_source_count)
    assert not artifact["reproduced"]


def test_top5_trn_volley_audit_localizes_downstream_transfer_loss() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_TRN_VOLLEY_AUDIT_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TOP5_TRN_VOLLEY_AUDIT_REGISTRATION_PATH.read_text())
    artifact = yaml.safe_load(FIGURE7_TOP5_TRN_VOLLEY_AUDIT_RESULT_PATH.read_text())
    assert profile["active_bin_minimum_event_count_rule"] == (
        "ceiling_fraction_times_population_size"
    )
    assert registration["execution"]["network_rerun"] is False
    assert artifact["network_rerun"] is False
    assert artifact["parameter_changes"] == "none"
    assert artifact["active_bin_minimum_event_count"] == 9
    assert artifact["classification"] == (
        "downstream_nonspecific_transfer_loses_two_of_seven_trn_volleys"
    )
    match = artifact["conditions"]["match"]
    mismatch = artifact["conditions"]["mismatch"]
    assert (match["volley_count"], match["responding_volley_count"]) == (10, 4)
    assert (mismatch["volley_count"], mismatch["responding_volley_count"]) == (
        7,
        5,
    )
    assert [volley["event_count"] for volley in mismatch["volleys"]] == [
        81,
        81,
        81,
        81,
        81,
        81,
        57,
    ]
    assert not mismatch["volleys"][0]["nonspecific_response_times_ms"]
    assert not mismatch["volleys"][1]["nonspecific_response_times_ms"]
    assert all(volley["nonspecific_response_times_ms"] for volley in mismatch["volleys"][2:])
    assert not artifact["reproduced"]


def test_top5_arrival_interaction_is_one_match_only_candidate() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_MATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_MATCH_REGISTRATION_PATH.read_text())
    assert profile["dimension"]["grid"] == [5]
    assert profile["protocol"]["top_down_cue_lead_ms"] == 7.85
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["execution"]["condition"] == "match"
    assert registration["execution"]["candidate_count"] == 1
    assert (
        registration["rationale"]["arrival_aligned_evidence"]["mismatch_counts"]["nonspecific"] == 7
    )
    assert registration["rationale"]["top5_evidence"]["mismatch_counts"]["relay"] == 3
    assert registration["mismatch_lock"].startswith("Do not inspect mismatch")


def test_top5_arrival_interaction_passes_match_before_verification() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_MATCH_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert outcome["pass"]
    assert artifact["match_survivor_target_counts"] == [5]
    assert sorted(set(result["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert len(result["relay_spike_times_ms"]) == 15
    assert len(result["trn_spike_times_ms"]) == 633
    assert len(result["nonspecific_spike_times_ms"]) == 4
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert artifact["assessment"]["advance_to_independent_match_verification"]
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_top5_arrival_verification_registers_exact_screen_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_VERIFICATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_VERIFICATION_REGISTRATION_PATH.read_text())
    assert profile["dimension"]["grid"] == [5]
    assert profile["protocol"]["top_down_cue_lead_ms"] == 7.85
    assert registration["screen_result"]["relay_events"] == 15
    assert registration["screen_result"]["trn_events"] == 633
    assert registration["screen_result"]["nonspecific_events"] == 4
    assert registration["verification"]["candidates"] == 1
    assert registration["verification"]["independently_rebuilt_match"]
    assert registration["verification"]["full_detector_cycle_diagnostics"]
    assert registration["mismatch_lock"].startswith("Do not run mismatch")


def test_top5_arrival_match_verifies_before_mismatch_registration() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_VERIFICATION_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert outcome["pass"]
    assert len(result["relay_spike_times_ms"]) == 15
    assert len(result["trn_spike_times_ms"]) == 633
    assert len(result["nonspecific_spike_times_ms"]) == 4
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert artifact["assessment"]["advance_to_mismatch"]
    assert not artifact["assessment"]["mismatch_remains_locked"]


def test_top5_arrival_mismatch_is_fixed_after_verified_match() -> None:
    profile = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_MISMATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_MISMATCH_REGISTRATION_PATH.read_text())
    assert profile["comparator"]["target_count"] == 5
    assert profile["protocol"]["top_down_cue_lead_ms"] == 7.85
    assert registration["authorization"]["match_counts"] == {
        "relay": 15,
        "trn": 633,
        "nonspecific": 4,
    }
    assert registration["authorization"]["sampled_match_trn_detector_cycles_complete"]
    assert registration["execution"]["condition"] == "vertical_mismatch"
    assert registration["execution"]["run_count"] == 1
    assert registration["stopping_rule"].startswith("Run exactly one fresh vertical mismatch")


def test_top5_arrival_pair_preserves_spatial_path_but_loses_arousal() -> None:
    artifact = yaml.safe_load(FIGURE7_TOP5_ARRIVAL_PAIR_RESULT_PATH.read_text())
    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 633
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert sorted(set(mismatch["relay_spike_indices"])) == [40]
    assert len(mismatch["relay_spike_times_ms"]) == 3
    assert len(mismatch["trn_spike_times_ms"]) == 560
    assert len(mismatch["nonspecific_spike_times_ms"]) == 4
    for passed_gate in (
        "match_relay_spatial_set",
        "mismatch_relay_overlap_only",
        "match_more_active_relay_cells",
        "match_more_trn_events",
        "match_nonspecific_40_hz",
        "figure7_target_duration",
        "sampled_mismatch_trn_events_have_fresh_cycles",
    ):
        assert artifact["gates"][passed_gate]
    assert not artifact["gates"]["mismatch_more_nonspecific_events"]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    assert (
        artifact["sampled_mismatch_trn_event_counts_by_index"]
        == artifact["sampled_mismatch_trn_threshold_upcrossings_by_index"]
    )
    assert (
        artifact["sampled_mismatch_trn_event_counts_by_index"]
        == artifact["sampled_mismatch_trn_arm_transitions_by_index"]
    )
    assert (
        artifact["sampled_mismatch_trn_event_counts_by_index"]
        == artifact["sampled_mismatch_trn_release_transitions_by_index"]
    )


def test_corticointralaminar_ablation_is_diagnostic_not_candidate() -> None:
    profile = yaml.safe_load(FIGURE7_CORTICOINTRALAMINAR_ABLATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_CORTICOINTRALAMINAR_ABLATION_REGISTRATION_PATH.read_text()
    )
    expected = [
        "modeldb112923.projection.050",
        "modeldb112923.projection.051",
    ]
    assert profile["causal_ablation"]["disabled_projection_ids"] == expected
    assert profile["causal_ablation"]["stage"] == ("recognition_only_after_fresh_figure6_handoff")
    assert profile["protocol"]["conditions"] == ["match", "mismatch"]
    assert registration["causal_intervention"]["disabled_projection_ids"] == expected
    assert registration["execution"]["run_count_per_condition"] == 1
    assert registration["classification_rule"].endswith("Do not promote an ablated network.")


def test_corticointralaminar_ablation_is_rate_neutral_and_exposes_one_event() -> None:
    artifact = yaml.safe_load(FIGURE7_CORTICOINTRALAMINAR_ABLATION_RESULT_PATH.read_text())
    readout = artifact["causal_readout"]
    match = artifact["match_result"]
    mismatch = artifact["mismatch_result"]
    expected_disabled = [
        "modeldb112923.projection.050",
        "modeldb112923.projection.051",
    ]
    assert artifact["status"] == "complete-causal-diagnostic-not-candidate"
    assert not artifact["promotable"]
    assert not artifact["reproduced"]
    assert match["disabled_projection_ids"] == expected_disabled
    assert mismatch["disabled_projection_ids"] == expected_disabled
    assert readout["corticointralaminar_gates_are_zero"]
    assert match["nonspecific_layer6ii_ampa_peak"] == 0.0
    assert match["nonspecific_layer6ii_nmda_peak"] == 0.0
    assert mismatch["nonspecific_layer6ii_ampa_peak"] == 0.0
    assert mismatch["nonspecific_layer6ii_nmda_peak"] == 0.0
    assert readout["match_nonspecific_events"] == 4
    assert readout["mismatch_nonspecific_events"] == 5
    assert readout["event_count_delta_mismatch_minus_match"] == 1
    assert readout["match_trn_events"] == 635
    assert readout["mismatch_trn_events"] == 560
    assert readout["match_trn_gaba_integral_ms"] > readout["mismatch_trn_gaba_integral_ms"]
    assert readout["classification"] == ("latent_trn_disinhibition_has_correct_sign")


def test_nonspecific_gaba_compartment_ablation_is_fixed_and_nonpromotable() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_GABA_COMPARTMENT_ABLATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_GABA_COMPARTMENT_ABLATION_REGISTRATION_PATH.read_text()
    )
    assert [item["label"] for item in profile["causal_ablations"]] == [
        "without_somatic_gaba",
        "without_proximal_gaba",
        "without_distal_gaba",
        "without_all_trn_gaba",
    ]
    assert registration["execution"]["ablation_count"] == 4
    assert registration["execution"]["conditions_per_ablation"] == [
        "match",
        "mismatch",
    ]
    assert not registration["scope_boundary"]["promotable"]
    assert registration["scope_boundary"]["no_weight_or_cell_parameter_calibration"]
    assert registration["stopping_rule"].startswith("Execute all four fixed ablations")


def test_nonspecific_gaba_compartment_ablation_localizes_rebound_roles() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_GABA_COMPARTMENT_ABLATION_RESULT_PATH.read_text())
    assert artifact["id"] == ("figure7-top5-nonspecific-gaba-compartment-ablation-348")
    assert artifact["status"] == "complete-causal-localization-not-candidate"
    assert artifact["classification"] == ("calibrated-reconstruction-causal-ablation")
    assert not artifact["promotable"]
    assert not artifact["reproduced"]

    outcomes = {item["label"]: item for item in artifact["outcomes"]}
    assert set(outcomes) == {
        "without_somatic_gaba",
        "without_proximal_gaba",
        "without_distal_gaba",
        "without_all_trn_gaba",
    }
    expected = {
        "without_somatic_gaba": (4, 4, ["modeldb112923.projection.047"]),
        "without_proximal_gaba": (0, 0, ["modeldb112923.projection.048"]),
        "without_distal_gaba": (5, 5, ["modeldb112923.projection.049"]),
        "without_all_trn_gaba": (
            0,
            0,
            [
                "modeldb112923.projection.047",
                "modeldb112923.projection.048",
                "modeldb112923.projection.049",
            ],
        ),
    }
    for label, (match_events, mismatch_events, disabled_ids) in expected.items():
        outcome = outcomes[label]
        readout = outcome["causal_readout"]
        assert outcome["disabled_projection_ids"] == disabled_ids
        assert outcome["match_result"]["disabled_projection_ids"] == disabled_ids
        assert outcome["mismatch_result"]["disabled_projection_ids"] == disabled_ids
        assert readout["match_nonspecific_events"] == match_events
        assert readout["mismatch_nonspecific_events"] == mismatch_events
        assert readout["event_count_delta_mismatch_minus_match"] == 0
        assert readout["match_trn_events"] == 635
        assert readout["mismatch_trn_events"] == 560

    no_soma = outcomes["without_somatic_gaba"]["causal_readout"]
    no_proximal = outcomes["without_proximal_gaba"]["causal_readout"]
    no_distal = outcomes["without_distal_gaba"]["causal_readout"]
    no_gaba = outcomes["without_all_trn_gaba"]["causal_readout"]
    assert no_soma["match_trn_gaba_integral_ms"] > no_soma["mismatch_trn_gaba_integral_ms"]
    assert no_proximal["match_trn_gaba_integral_ms"] > no_proximal["mismatch_trn_gaba_integral_ms"]
    assert no_distal["match_trn_gaba_integral_ms"] > no_distal["mismatch_trn_gaba_integral_ms"]
    assert no_gaba["match_trn_gaba_integral_ms"] == 0.0
    assert no_gaba["mismatch_trn_gaba_integral_ms"] == 0.0


def test_nonspecific_paper_ttype_discriminator_is_preregistered_and_locked() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_PAPER_TTYPE_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_NONSPECIFIC_PAPER_TTYPE_REGISTRATION_PATH.read_text())
    assert profile["runtime_overrides"]["nonspecific_calcium_kinetics_convention"] == "paper_2008"
    assert registration["registered_dimension"]["candidate_count"] == 1
    assert registration["fixed"]["trn_calcium_kinetics"] == ("modeldb_reticular_112923")
    assert registration["fixed"]["relay_calcium_kinetics"] == "modeldb_112923"
    assert registration["fixed"]["nonspecific_gaba_weights_and_kinetics_unchanged"]
    assert registration["scope_boundary"]["changes"] == ("nonspecific T-type gate equations only")
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert "figure7_mismatch" in registration["locked_holdouts"]


def test_nonspecific_paper_ttype_fails_exact_match_before_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_PAPER_TTYPE_RESULT_PATH.read_text())
    assert artifact["id"] == "figure7-top5-nonspecific-paper-ttype-match-350"
    assert artifact["status"] == "match-failed"
    assert artifact["classification"] == ("official-source-discriminator-not-yet-baseline")
    assert artifact["nonspecific_calcium_kinetics_convention"] == "paper_2008"
    assert artifact["figure6_pass"]
    assert all(artifact["figure6_gates"].values())
    assert not artifact["match_pass"]
    assert not artifact["mismatch_consulted"]
    assert not artifact["promotable"]
    assert not artifact["reproduced"]
    assert not artifact["advance_to_independent_match_verification"]

    match = artifact["match_result"]
    assert sorted(set(match["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 639
    assert len(match["nonspecific_spike_times_ms"]) == 8
    assert not artifact["match_gates"]["nonspecific_events"]
    assert not artifact["match_gates"]["nonspecific_40_hz"]
    assert all(
        value
        for key, value in artifact["match_gates"].items()
        if key not in {"nonspecific_events", "nonspecific_40_hz"}
    )


def test_nonspecific_somatic_gaba_sensitivity_is_single_and_locked() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_SOMATIC_GABA_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_NONSPECIFIC_SOMATIC_GABA_REGISTRATION_PATH.read_text())
    transfer = profile["nonspecific_gaba_transfer"]
    assert transfer["projection_ids"] == ["modeldb112923.projection.047"]
    assert transfer["common_scale_grid"] == [2.0]
    assert transfer["selection_rule"] == "sole preregistered non-source candidate"
    assert registration["registered_dimension"] == {
        "projection_id": "modeldb112923.projection.047",
        "target_compartment": "soma",
        "source_weight": 0.01,
        "persistent_scale": 2.0,
        "effective_weight": 0.02,
        "candidate_count": 1,
    }
    assert registration["fixed"]["proximal_gaba_scale"] == 1.0
    assert registration["fixed"]["distal_gaba_scale"] == 1.0
    assert registration["scope_boundary"]["classification"] == (
        "calibrated-reconstruction-not-recovered-source"
    )
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert "figure7_mismatch" in registration["locked_holdouts"]


def test_nonspecific_somatic_gaba_sensitivity_passes_figure6_and_match() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_SOMATIC_GABA_RESULT_PATH.read_text())
    assert artifact["id"] == "figure7-top5-nonspecific-somatic-gaba-match-352"
    assert artifact["status"] == "complete"
    assert artifact["classification"] == ("calibrated-reconstruction-not-recovered-source")
    assert artifact["match_survivor_common_scales"] == [2.0]
    assert artifact["selected_common_scale"] == 2.0
    assert not artifact["mismatch_consulted"]
    assert artifact["assessment"]["advance_to_independent_match_verification"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    assert artifact["assessment"]["mismatch_remains_locked"]

    outcome = artifact["outcomes"][0]
    assert outcome["projection_weight_scales"]["modeldb112923.projection.047"] == 2.0
    assert outcome["figure6_pass"]
    assert all(outcome["figure6_gates"].values())
    assert outcome["match_pass"]
    assert all(outcome["match_gates"].values())
    assert len(outcome["match_result"]["relay_spike_times_ms"]) == 15
    assert len(outcome["match_result"]["trn_spike_times_ms"]) == 635
    assert len(outcome["match_result"]["nonspecific_spike_times_ms"]) == 4


def test_nonspecific_somatic_gaba_verification_is_fixed_and_locked() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_SOMATIC_GABA_VERIFICATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_SOMATIC_GABA_VERIFICATION_REGISTRATION_PATH.read_text()
    )
    assert profile["verification_screen_artifact"] == (
        "docs/validation-results/figure7-top5-nonspecific-somatic-gaba-match-352.yaml"
    )
    assert profile["nonspecific_gaba_transfer"]["common_scale_grid"] == [2.0]
    assert registration["verification"]["candidate_count"] == 1
    assert registration["verification"]["persistent_somatic_scale"] == 2.0
    assert registration["verification"]["independently_rebuilt_network"]
    assert registration["mismatch_lock"].startswith("Do not run mismatch")


def test_nonspecific_somatic_gaba_verification_repeats_exact_match() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_SOMATIC_GABA_VERIFICATION_RESULT_PATH.read_text())
    assert artifact["id"] == ("figure7-top5-nonspecific-somatic-gaba-verification-354")
    assert artifact["status"] == "complete"
    assert artifact["selected_common_scale"] == 2.0
    assert not artifact["mismatch_consulted"]
    assert artifact["assessment"]["advance_to_mismatch"]
    assert not artifact["assessment"]["mismatch_remains_locked"]
    outcome = artifact["outcomes"][0]
    assert outcome["figure6_pass"]
    assert all(outcome["figure6_gates"].values())
    assert outcome["match_pass"]
    assert all(outcome["match_gates"].values())
    assert len(outcome["match_result"]["relay_spike_times_ms"]) == 15
    assert len(outcome["match_result"]["trn_spike_times_ms"]) == 635
    assert len(outcome["match_result"]["nonspecific_spike_times_ms"]) == 4


def test_nonspecific_somatic_gaba_mismatch_is_fixed_before_execution() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_SOMATIC_GABA_MISMATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_SOMATIC_GABA_MISMATCH_REGISTRATION_PATH.read_text()
    )
    assert profile["match_verification_artifact"] == (
        "docs/validation-results/figure7-top5-nonspecific-somatic-gaba-verification-354.yaml"
    )
    assert profile["nonspecific_gaba_transfer"] == {
        "common_scale": 2.0,
        "projection_ids": ["modeldb112923.projection.047"],
    }
    assert registration["holdout"]["condition"] == "mismatch"
    assert registration["holdout"]["execution_count"] == 1
    assert registration["fixed"]["persistent_somatic_scale"] == 2.0
    assert registration["official_gates"]["mismatch_nonspecific_events"] == 7
    assert registration["stopping_rule"].startswith("Execute one fresh")


def test_nonspecific_somatic_gaba_pair_preserves_path_but_misses_70_hz() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_SOMATIC_GABA_PAIR_RESULT_PATH.read_text())
    assert artifact["id"] == "figure7-top5-nonspecific-somatic-gaba-pair-356"
    assert artifact["status"] == "figure7-failed"
    assert artifact["classification"] == ("calibrated-reconstruction-not-recovered-source")
    assert artifact["nonspecific_gaba_common_scale"] == 2.0
    assert artifact["projection_weight_scales"]["modeldb112923.projection.047"] == 2.0
    assert not artifact["reproduced"]
    assert all(artifact["figure6_gates"].values())

    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert sorted(set(mismatch["relay_spike_indices"])) == [40]
    assert len(mismatch["relay_spike_times_ms"]) == 3
    assert len(mismatch["trn_spike_times_ms"]) == 560
    assert len(mismatch["nonspecific_spike_times_ms"]) == 5
    assert artifact["gates"]["mismatch_relay_overlap_only"]
    assert artifact["gates"]["match_more_trn_events"]
    assert artifact["gates"]["mismatch_more_nonspecific_events"]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    assert artifact["gates"]["sampled_mismatch_trn_events_have_fresh_cycles"]


def test_nonspecific_kinness_axial_source_discriminator_is_isolated() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_KINNESS_AXIAL_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_NONSPECIFIC_KINNESS_AXIAL_REGISTRATION_PATH.read_text())
    assert profile["runtime_overrides"]["nonspecific_axial_convention"] == (
        "kinness_serialized_edge"
    )
    assert profile["runtime_expectations"] == {
        "nonspecific_axial_convention": "kinness_serialized_edge"
    }
    assert registration["registered_dimension"]["candidate_count"] == 1
    assert registration["fixed"]["relay_axial_convention"] == ("kinness_serialized_edge")
    assert registration["fixed"]["trn_axial_convention"] == "paper_literal"
    assert registration["fixed"]["cortical_axial_convention"] == "paper_literal"
    assert registration["scope_boundary"]["changes"] == (
        "nonspecific axial equation/conductance interpretation only"
    )
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert "figure7_mismatch" in registration["locked_holdouts"]


def test_nonspecific_kinness_axial_source_fails_match_before_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_KINNESS_AXIAL_RESULT_PATH.read_text())
    assert artifact["id"] == "figure7-top5-nonspecific-kinness-axial-match-358"
    assert artifact["status"] == "match-failed"
    assert artifact["classification"] == ("official-source-discriminator-not-yet-baseline")
    assert artifact["runtime_discriminator"] == {
        "nonspecific_axial_convention": "kinness_serialized_edge"
    }
    assert artifact["figure6_pass"]
    assert all(artifact["figure6_gates"].values())
    assert not artifact["match_pass"]
    assert not artifact["mismatch_consulted"]
    assert not artifact["promotable"]
    assert not artifact["reproduced"]
    assert not artifact["advance_to_independent_match_verification"]

    match = artifact["match_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 0
    assert not artifact["match_gates"]["nonspecific_events"]
    assert not artifact["match_gates"]["nonspecific_40_hz"]
    assert all(
        value
        for key, value in artifact["match_gates"].items()
        if key not in {"nonspecific_events", "nonspecific_40_hz"}
    )
    soma_range = {
        name: (minimum, maximum)
        for name, minimum, maximum in match["nonspecific_voltage_range_mV_by_compartment"]
    }["soma"]
    assert soma_range[1] < -50.0


def test_nonspecific_modeldb_intrinsic_source_is_complete_and_isolated() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_MODELDB_INTRINSIC_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_INTRINSIC_REGISTRATION_PATH.read_text()
    )
    assert profile["runtime_overrides"]["nonspecific_intrinsic_cell_convention"] == "modeldb_112923"
    assert profile["runtime_expectations"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923"
    }
    assert registration["registered_dimension"]["candidate_count"] == 1
    assert registration["source_contrast"]["paper_soma_mS_cm2"] == {
        "sodium": 100.0,
        "potassium": 100.0,
    }
    assert registration["source_contrast"]["modeldb_soma_mS_cm2"] == {
        "sodium": 50.0,
        "potassium": 30.0,
    }
    assert registration["fixed"]["nonspecific_axial_convention"] == ("paper_literal")
    assert registration["scope_boundary"]["changes"] == (
        "complete nonspecific intrinsic cell record only"
    )
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert "figure7_mismatch" in registration["locked_holdouts"]


def test_nonspecific_modeldb_intrinsic_source_fails_match_before_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_MODELDB_INTRINSIC_RESULT_PATH.read_text())
    assert artifact["id"] == ("figure7-top5-nonspecific-modeldb-intrinsic-match-360")
    assert artifact["status"] == "match-failed"
    assert artifact["classification"] == ("official-source-discriminator-not-yet-baseline")
    assert artifact["runtime_discriminator"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923"
    }
    assert artifact["figure6_pass"]
    assert all(artifact["figure6_gates"].values())
    assert not artifact["match_pass"]
    assert not artifact["mismatch_consulted"]
    assert not artifact["promotable"]
    assert not artifact["reproduced"]
    assert not artifact["advance_to_independent_match_verification"]

    match = artifact["match_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 0
    assert not artifact["match_gates"]["nonspecific_events"]
    assert not artifact["match_gates"]["nonspecific_40_hz"]
    assert all(
        value
        for key, value in artifact["match_gates"].items()
        if key not in {"nonspecific_events", "nonspecific_40_hz"}
    )
    soma_maximum = {
        name: maximum
        for name, _minimum, maximum in match["nonspecific_voltage_range_mV_by_compartment"]
    }["soma"]
    assert 28.0 < soma_maximum < 30.0


def test_nonspecific_modeldb_kinness_detector_pair_is_source_fixed() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_MODELDB_KINNESS_DETECTOR_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_KINNESS_DETECTOR_REGISTRATION_PATH.read_text()
    )
    assert profile["runtime_expectations"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923",
        "nonspecific_spike_event_threshold_mV": -20.0,
    }
    dimensions = registration["registered_dimensions"]
    assert dimensions["intrinsic_cell_source"] == "modeldb_112923"
    assert dimensions["event_threshold_source"] == "kinness_2008"
    assert dimensions["event_arm_mV"] == -20.0
    assert dimensions["event_release_mV"] == 0.0
    assert dimensions["candidate_count"] == 1
    assert registration["fixed"]["all_trn_and_relay_event_detectors_unchanged"]
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert "figure7_mismatch" in registration["locked_holdouts"]


def test_nonspecific_modeldb_kinness_detector_pair_rearms_pathologically() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_MODELDB_KINNESS_DETECTOR_RESULT_PATH.read_text())
    assert artifact["id"] == ("figure7-top5-nonspecific-modeldb-kinness-detector-match-362")
    assert artifact["status"] == "match-failed"
    assert artifact["runtime_discriminator"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923",
        "nonspecific_spike_event_threshold_mV": -20.0,
    }
    assert artifact["figure6_pass"]
    assert not artifact["match_pass"]
    assert not artifact["mismatch_consulted"]
    match = artifact["match_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 287
    first_interval = match["nonspecific_spike_times_ms"][1] - match["nonspecific_spike_times_ms"][0]
    assert first_interval == pytest.approx(0.02)
    assert not artifact["promotable"]
    assert not artifact["reproduced"]


def test_nonspecific_modeldb_kinness_hysteresis_is_fixed_and_isolated() -> None:
    profile = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_KINNESS_HYSTERESIS_PROFILE_PATH.read_text()
    )
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_KINNESS_HYSTERESIS_REGISTRATION_PATH.read_text()
    )
    assert profile["runtime_expectations"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923",
        "nonspecific_spike_event_threshold_mV": -20.0,
        "nonspecific_spike_event_rule": "hysteretic_threshold_then_zero",
    }
    dimensions = registration["registered_dimensions"]
    assert dimensions["event_arm_mV"] == -20.0
    assert dimensions["event_release_mV"] == 0.0
    assert dimensions["event_rule"] == "hysteretic_threshold_then_zero"
    assert dimensions["candidate_count"] == 1
    assert registration["fixed"]["all_trn_and_relay_event_detectors_unchanged"]
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert "figure7_mismatch" in registration["locked_holdouts"]


def test_nonspecific_modeldb_kinness_hysteresis_still_overfires_match() -> None:
    artifact = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_KINNESS_HYSTERESIS_RESULT_PATH.read_text()
    )
    assert artifact["id"] == ("figure7-top5-nonspecific-modeldb-kinness-hysteresis-match-364")
    assert artifact["status"] == "match-failed"
    assert artifact["runtime_discriminator"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923",
        "nonspecific_spike_event_threshold_mV": -20.0,
        "nonspecific_spike_event_rule": "hysteretic_threshold_then_zero",
    }
    assert artifact["figure6_pass"]
    assert all(artifact["figure6_gates"].values())
    assert not artifact["match_pass"]
    assert not artifact["mismatch_consulted"]
    assert not artifact["promotable"]
    assert not artifact["reproduced"]
    match = artifact["match_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 22
    assert not artifact["match_gates"]["nonspecific_events"]
    assert not artifact["match_gates"]["nonspecific_40_hz"]
    times = match["nonspecific_spike_times_ms"]
    intervals = [later - earlier for earlier, later in pairwise(times)]
    assert min(intervals) > 1.0


def test_nonspecific_modeldb_uniform_handler_is_fixed_and_isolated() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_MODELDB_UNIFORM_HANDLER_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_UNIFORM_HANDLER_REGISTRATION_PATH.read_text()
    )
    assert profile["runtime_expectations"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923",
        "nonspecific_spike_event_threshold_mV": -20.0,
        "nonspecific_spike_event_release_mV": -30.0,
        "nonspecific_spike_event_rule": "latched_peak_then_zero",
    }
    dimensions = registration["registered_dimension"]
    assert dimensions["candidate_count"] == 1
    assert dimensions["nonspecific_spike_event_threshold_mV"] == -20.0
    assert dimensions["nonspecific_spike_event_release_mV"] == -30.0
    assert registration["fixed"]["trn_handler"] == {
        "arm_mV": -20.0,
        "release_mV": -30.0,
    }
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert "figure7_mismatch" in registration["locked_holdouts"]


def test_nonspecific_modeldb_uniform_handler_still_overfires_match() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_MODELDB_UNIFORM_HANDLER_RESULT_PATH.read_text())
    assert artifact["id"] == ("figure7-top5-nonspecific-modeldb-uniform-handler-match-370")
    assert artifact["status"] == "match-failed"
    assert artifact["runtime_discriminator"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923",
        "nonspecific_spike_event_threshold_mV": -20.0,
        "nonspecific_spike_event_release_mV": -30.0,
        "nonspecific_spike_event_rule": "latched_peak_then_zero",
    }
    assert artifact["figure6_pass"]
    assert all(artifact["figure6_gates"].values())
    assert not artifact["match_pass"]
    assert not artifact["mismatch_consulted"]
    assert not artifact["promotable"]
    assert not artifact["reproduced"]
    match = artifact["match_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 22
    assert not artifact["match_gates"]["nonspecific_events"]
    assert not artifact["match_gates"]["nonspecific_40_hz"]


def test_nonspecific_modeldb_calibrated20_is_match_only_and_fixed() -> None:
    profile = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_MATCH_PROFILE_PATH.read_text()
    )
    registration = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_MATCH_REGISTRATION_PATH.read_text()
    )
    assert profile["runtime_expectations"] == {
        "nonspecific_intrinsic_cell_convention": "modeldb_112923",
        "nonspecific_spike_event_threshold_mV": 20.0,
        "nonspecific_spike_event_release_mV": 0.0,
        "nonspecific_spike_event_rule": "latched_peak_then_zero",
    }
    assert registration["classification"] == ("calibrated-reconstruction-not-recovered-source")
    assert registration["registered_dimension"]["candidate_count"] == 1
    assert registration["execution"]["mismatch_runs"] == 0
    assert "figure7_mismatch" in registration["locked_holdouts"]


def test_nonspecific_modeldb_calibrated20_repeats_exact_match() -> None:
    screen = yaml.safe_load(FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_MATCH_RESULT_PATH.read_text())
    verification = yaml.safe_load(
        FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_VERIFICATION_RESULT_PATH.read_text()
    )
    for artifact in (screen, verification):
        assert artifact["status"] == "match-pass"
        assert artifact["figure6_pass"]
        assert all(artifact["figure6_gates"].values())
        assert artifact["match_pass"]
        assert artifact["runtime_fingerprint"] == (
            "7c164995390f7d2df9a618f529d375519114163fd254125d8829bd6642cc8c93"
        )
        match = artifact["match_result"]
        assert len(match["relay_spike_times_ms"]) == 15
        assert len(match["trn_spike_times_ms"]) == 635
        assert len(match["nonspecific_spike_times_ms"]) == 4
        assert artifact["match_gates"]["nonspecific_40_hz"]
        assert not artifact["mismatch_consulted"]


def test_nonspecific_modeldb_calibrated20_fails_mismatch_holdout() -> None:
    artifact = yaml.safe_load(FIGURE7_NONSPECIFIC_MODELDB_CALIBRATED20_PAIR_RESULT_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["runtime_fingerprint"] == (
        "7c164995390f7d2df9a618f529d375519114163fd254125d8829bd6642cc8c93"
    )
    assert artifact["handoff_figure6_population_spikes"]["thalamic_relay"] == 20
    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert len(mismatch["relay_spike_times_ms"]) == 3
    assert set(mismatch["relay_spike_indices"]) == {40}
    assert len(mismatch["trn_spike_times_ms"]) == 560
    assert len(mismatch["nonspecific_spike_times_ms"]) == 5
    assert all(
        value for gate, value in artifact["gates"].items() if gate != "mismatch_nonspecific_70_hz"
    )
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    assert len(mismatch["nonspecific_positive_soma_local_maxima_ms_mV"]) == 6
    assert not artifact["reproduced"]


def test_recovered_nonspecific_cell_does_not_relax_to_quiescent_rest() -> None:
    profile = yaml.safe_load(ISOLATED_NONSPECIFIC_REST_AUDIT_PROFILE_PATH.read_text())
    registration = yaml.safe_load(ISOLATED_NONSPECIFIC_REST_AUDIT_REGISTRATION_PATH.read_text())
    artifact = yaml.safe_load(ISOLATED_NONSPECIFIC_REST_AUDIT_RESULT_PATH.read_text())
    assert profile["protocol"] == {
        "duration_ms": 500.0,
        "dt_ms": 0.01,
        "recording_dt_ms": 0.1,
        "terminal_window_ms": 100.0,
        "applied_input": "none",
    }
    assert registration["fixed_runtime"]["runtime_fingerprint"] == (
        "7c164995390f7d2df9a618f529d375519114163fd254125d8829bd6642cc8c93"
    )
    assert artifact["status"] == "quiescent-rest-not-supported"
    assert artifact["all_state_samples_finite"]
    assert artifact["terminal_detector_event_count"] == 0
    assert artifact["maximum_terminal_peak_to_peak_mV"] == pytest.approx(51.39470959905637)
    assert (
        artifact["compartments"]["soma"]["terminal_peak_to_peak_mV"]
        > profile["operational_gate"]["terminal_peak_to_peak_at_most_mV"]
    )
    assert not artifact["quiescent_rest_supported"]


def test_recovered_nonspecific_cell_has_one_stationary_fixed_point() -> None:
    derivation = yaml.safe_load(ISOLATED_NONSPECIFIC_FIXED_POINT_DERIVATION_PATH.read_text())
    verification = yaml.safe_load(ISOLATED_NONSPECIFIC_FIXED_POINT_VERIFICATION_PATH.read_text())
    assert derivation["attempt_count"] == 125
    assert derivation["converged_attempt_count"] == 54
    assert derivation["unique_root_count"] == 1
    assert derivation["stationarity_result"]["stationary_by_registered_probe"]
    assert verification["status"] == "fixed-point-verification-complete"
    assert verification["expected_fixed_point_pass"]
    assert verification["unique_fixed_points_mV"][0] == pytest.approx(
        [-37.64292925949306, -29.37693872874881, -25.70881637435125]
    )
    assert verification["stationarity_result"]["maximum_peak_to_peak_mV"] == 0.0


def test_nonspecific_fixed_point_fails_registered_somatic_perturbation() -> None:
    profile = yaml.safe_load(ISOLATED_NONSPECIFIC_FIXED_POINT_PERTURBATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        ISOLATED_NONSPECIFIC_FIXED_POINT_PERTURBATION_REGISTRATION_PATH.read_text()
    )
    artifact = yaml.safe_load(ISOLATED_NONSPECIFIC_FIXED_POINT_PERTURBATION_RESULT_PATH.read_text())
    assert profile["stationarity_probe"]["initial_voltage_perturbation_mV"] == [
        0.000001,
        0.0,
        0.0,
    ]
    assert registration["holdout_lock"] == {
        "figure7_match_runs": 0,
        "figure7_mismatch_runs": 0,
    }
    assert artifact["status"] == "fixed-point-stability-probe-failed"
    assert artifact["expected_fixed_point_pass"]
    assert artifact["stationarity_result"]["initial_voltage_perturbation_mV"] == [
        0.000001,
        0.0,
        0.0,
    ]
    assert artifact["stationarity_result"]["maximum_peak_to_peak_mV"] == pytest.approx(
        56.31728131527776
    )
    assert not artifact["stationarity_result"]["stationary_by_registered_probe"]


def test_full_grid_connectfromall_fails_match_before_mismatch() -> None:
    profile = yaml.safe_load(FIGURE7_FULL_GRID_CONNECTFROMALL_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_FULL_GRID_CONNECTFROMALL_REGISTRATION_PATH.read_text())
    artifact = yaml.safe_load(FIGURE7_FULL_GRID_CONNECTFROMALL_RESULT_PATH.read_text())
    assert profile["protocol"]["convergent_external_source_scope"] == ("full_input_grid")
    assert registration["fixed_discriminator"] == {
        "previous_source_count": 5,
        "candidate_source_count": 81,
        "summed_green_value": 600,
        "fitted_parameters": "none",
    }
    assert artifact["convergent_external_source_scope"] == "full_input_grid"
    assert artifact["figure6_pass"]
    assert all(artifact["figure6_gates"].values())
    assert artifact["figure6_result"]["population_spikes"]["thalamic_relay"] == 20
    assert artifact["status"] == "match-failed"
    assert not artifact["match_pass"]
    assert not artifact["mismatch_consulted"]
    match = artifact["match_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 635
    assert len(match["nonspecific_spike_times_ms"]) == 6
    assert not artifact["match_gates"]["nonspecific_40_hz"]


def test_aligned_on_center_verification_registers_only_screen_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_ALIGNED_VERIFICATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_ALIGNED_VERIFICATION_REGISTRATION_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["dimension"]["grid"] == [1.0]
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["screen_result"]["selected_headroom_fraction"] == 1.0
    assert registration["verification"]["candidates"] == 1
    assert registration["verification"]["independently_rebuilt_network"]
    assert registration["mismatch_lock"].startswith("Do not run mismatch")


def test_aligned_on_center_verification_has_fresh_trn_cycles() -> None:
    artifact = yaml.safe_load(FIGURE7_ALIGNED_VERIFICATION_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["stage_1_survivor_headroom_fractions"] == [1.0]
    outcome = artifact["outcomes"][0]
    assert outcome["pass"]
    assert set(outcome["relay_event_counts_by_index"].values()) == {3}
    assert len(outcome["result"]["trn_spike_times_ms"]) == 633
    assert len(outcome["result"]["nonspecific_spike_times_ms"]) == 4
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    for index, event_count in outcome["sampled_trn_event_counts_by_index"].items():
        assert event_count == outcome["sampled_trn_threshold_upcrossings_by_index"][index]
        assert event_count == outcome["sampled_trn_arm_transitions_by_index"][index]
        assert event_count == outcome["sampled_trn_release_transitions_by_index"][index]
    assert artifact["assessment"]["advance_to_mismatch"]
    assert not artifact["assessment"]["mismatch_remains_locked"]


def test_aligned_on_center_mismatch_is_single_locked_holdout() -> None:
    profile = yaml.safe_load(FIGURE7_ALIGNED_MISMATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_ALIGNED_MISMATCH_REGISTRATION_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["learned_state"]["selected_headroom_fraction"] == 1.0
    assert registration["registered_holdout"]["conditions"] == ["mismatch"]
    assert registration["registered_holdout"]["independent_network_rebuild"]
    assert registration["execution_limit"].startswith("exactly one mismatch")
    assert registration["official_gates"]["mismatch_nonspecific_events"] == 7
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_aligned_on_center_pair_hits_rates_but_fails_spatial_subset() -> None:
    artifact = yaml.safe_load(FIGURE7_ALIGNED_PAIR_RESULT_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(mismatch["relay_spike_times_ms"]) == 10
    assert len(match["trn_spike_times_ms"]) == 633
    assert len(mismatch["trn_spike_times_ms"]) == 583
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert len(mismatch["nonspecific_spike_times_ms"]) == 7
    assert set(match["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert set(mismatch["relay_spike_indices"]) == {22, 31, 40, 49, 58}
    gates = artifact["gates"]
    assert gates["match_more_trn_events"]
    assert gates["mismatch_more_nonspecific_events"]
    assert gates["match_nonspecific_40_hz"]
    assert gates["mismatch_nonspecific_70_hz"]
    assert gates["sampled_mismatch_trn_events_have_fresh_cycles"]
    assert not gates["mismatch_relay_overlap_only"]
    assert not gates["match_more_active_relay_cells"]


def test_aligned_sustained_current_is_single_preregistered_waveform() -> None:
    profile = yaml.safe_load(FIGURE7_ALIGNED_SUSTAINED_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_ALIGNED_SUSTAINED_REGISTRATION_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["protocol"]["top_down_current_mode"] == "sustained_epoch"
    assert profile["dimension"]["grid"] == [1.0]
    assert registration["dimension"]["registered_values"] == ["sustained_epoch"]
    assert registration["source_basis"]["duration_reported_for_figure7"] is False
    assert registration["fixed_choices"]["top_down_cue_lead_ms"] == pytest.approx(7.85)
    assert registration["fixed_choices"]["learned_headroom_fraction"] == 1.0
    assert registration["mismatch_lock"].startswith("Do not run mismatch")


def test_aligned_sustained_current_fails_exact_match_and_locks_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_ALIGNED_SUSTAINED_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["stage_1_survivor_headroom_fractions"] == []
    outcome = artifact["outcomes"][0]
    assert set(outcome["relay_event_counts_by_index"].values()) == {3}
    assert len(outcome["result"]["trn_spike_times_ms"]) == 608
    assert len(outcome["result"]["nonspecific_spike_times_ms"]) == 5
    assert outcome["result"]["top_down_current_termination_time_ms"] is None
    assert outcome["gates"]["sustained_current_protocol"]
    assert not outcome["gates"]["nonspecific_40_hz"]
    assert not outcome["pass"]
    assert artifact["assessment"]["mismatch_remains_locked"]
    assert not artifact["assessment"]["advance_to_mismatch"]


def test_mismatch_event_current_audit_changes_readout_only() -> None:
    registration = yaml.safe_load(FIGURE7_EVENT_CURRENT_REGISTRATION_PATH.read_text())
    assert registration["status"] == "registered-before-execution"
    assert registration["candidate_changes"] == "none"
    assert registration["rerun_contract"]["condition"] == "mismatch"
    assert registration["rerun_contract"]["all_randomness_and_parameters_unchanged"]
    assert len(registration["new_readout"]["currents_pA"]) == 9
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_mismatch_event_current_audit_localizes_first_nonoverlap_events() -> None:
    artifact = yaml.safe_load(FIGURE7_EVENT_CURRENT_AUDIT_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(mismatch["relay_spike_times_ms"]) == 10
    assert len(match["trn_spike_times_ms"]) == 633
    assert len(mismatch["trn_spike_times_ms"]) == 583
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert len(mismatch["nonspecific_spike_times_ms"]) == 7

    samples = mismatch["relay_event_current_samples_pA"]
    assert len(samples) == 90
    first_by_index: dict[int, dict[str, float]] = {}
    for index, _time_ms, source, current_pA in samples:
        first_by_index.setdefault(index, {}).setdefault(source, current_pA)

    for index in (22, 31, 49, 58):
        currents = first_by_index[index]
        assert currents["direct_image_input"] > 115.0
        assert currents["top_down_excitation"] < 1.1
        assert currents["distal_calcium"] < 2.0
        assert currents["proximal_calcium"] < 2.0
        assert currents["trn_gaba"] < -1000.0

    assert artifact["gates"]["match_more_trn_events"]
    assert artifact["gates"]["mismatch_more_nonspecific_events"]
    assert not artifact["gates"]["mismatch_relay_overlap_only"]


def test_mismatch_pre_event_trace_audit_is_readout_only_and_preregistered() -> None:
    registration = yaml.safe_load(FIGURE7_PRE_EVENT_TRACE_REGISTRATION_PATH.read_text())
    assert registration["status"] == "registered-before-execution"
    assert registration["candidate_changes"] == "none"
    assert registration["rerun_contract"]["all_randomness_and_parameters_unchanged"]
    assert registration["pre_event_offsets_ms"] == [2.0, 1.0, 0.5, 0.2]
    assert len(registration["new_readouts"]["currents_pA"]) == 9
    assert registration["new_readouts"]["voltages_mV"] == [
        "distal_dendrite",
        "proximal_dendrite",
        "soma",
    ]
    assert registration["new_readouts"]["dimensionless_gates"] == ["trn_gaba_combined"]
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_mismatch_pre_event_trace_audit_excludes_common_inhibition_trough() -> None:
    artifact = yaml.safe_load(FIGURE7_PRE_EVENT_TRACE_AUDIT_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["registration_artifact"].endswith(
        "figure7-mismatch-pre-event-trace-registration-249.yaml"
    )
    assert not artifact["reproduced"]
    mismatch = artifact["mismatch_result"]
    assert len(mismatch["relay_spike_times_ms"]) == 10
    assert len(mismatch["trn_spike_times_ms"]) == 583
    assert len(mismatch["nonspecific_spike_times_ms"]) == 7
    assert len(mismatch["relay_pre_event_current_samples_pA"]) == 360
    assert len(mismatch["relay_pre_event_voltage_samples_mV"]) == 120
    assert len(mismatch["relay_pre_event_trn_gaba_gate_samples"]) == 40

    first_event_by_index: dict[int, float] = {}
    for index, time_ms in zip(
        mismatch["relay_spike_indices"],
        mismatch["relay_spike_times_ms"],
        strict=True,
    ):
        first_event_by_index.setdefault(index, time_ms)

    gate_by_index_and_offset = {
        (index, offset_ms): value
        for index, event_time_ms, offset_ms, value in mismatch[
            "relay_pre_event_trn_gaba_gate_samples"
        ]
        if event_time_ms == first_event_by_index[index]
    }
    assert gate_by_index_and_offset[(31, 0.2)] < gate_by_index_and_offset[(31, 2.0)]
    assert gate_by_index_and_offset[(22, 0.2)] > gate_by_index_and_offset[(22, 2.0)]

    soma_by_index_and_offset = {
        (index, offset_ms): value
        for index, event_time_ms, offset_ms, source, value in mismatch[
            "relay_pre_event_voltage_samples_mV"
        ]
        if event_time_ms == first_event_by_index[index] and source == "soma"
    }
    for index in (22, 31, 40, 49, 58):
        assert -53.0 < soma_by_index_and_offset[(index, 2.0)] < -50.0
        assert -44.0 < soma_by_index_and_offset[(index, 0.5)] < -42.0
        assert soma_by_index_and_offset[(index, 0.2)] > 40.0


def test_mismatch_gaba_capacity_screen_is_finite_and_diagnostic_only() -> None:
    profile = yaml.safe_load(FIGURE7_GABA_CAPACITY_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_GABA_CAPACITY_REGISTRATION_PATH.read_text())
    assert profile["status"] == "registered-before-execution-diagnostic"
    assert profile["dimension"]["control_gain"] == 1.0
    assert profile["dimension"]["grid"] == [1.125, 1.25, 1.5, 2.0, 3.0]
    assert registration["dimension"]["registered_common_gains"] == profile["dimension"]["grid"]
    assert registration["dimension"]["source_status"] == ("post_holdout_causal_capacity_diagnostic")
    assert registration["capacity_criterion"]["overlap_only_active_indices"] == [40]
    assert registration["execution_contract"][-1].startswith("Do not select or promote")
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_mismatch_gaba_capacity_screen_has_no_overlap_only_window() -> None:
    artifact = yaml.safe_load(FIGURE7_GABA_CAPACITY_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["diagnostic_only"]
    assert artifact["overlap_only_gains"] == []
    assert not artifact["assessment"]["scalar_gain_has_overlap_only_window"]
    assert not artifact["assessment"]["candidate_promoted"]
    assert not artifact["assessment"]["downstream_holdouts_unlocked"]
    outcomes = artifact["outcomes"]
    assert [outcome["common_gain"] for outcome in outcomes] == [
        1.125,
        1.25,
        1.5,
        2.0,
        3.0,
    ]
    assert all(outcome["relay_active_indices"] == [22, 31, 40, 49, 58] for outcome in outcomes)
    assert [outcome["relay_events"] for outcome in outcomes] == [10, 10, 10, 10, 5]
    assert [outcome["nonspecific_events"] for outcome in outcomes] == [5, 4, 4, 4, 4]


def test_radial_annulus_gain_calibration_is_figure6_first_and_explicitly_exploratory() -> None:
    profile = yaml.safe_load(RADIAL_ANNULUS_GABA_GAIN_PROFILE_PATH.read_text())
    registration = yaml.safe_load(RADIAL_ANNULUS_GABA_GAIN_REGISTRATION_PATH.read_text())
    assert profile["runtime_overrides"]["ring_kernel_convention"] == ("radial_annulus")
    assert profile["dimension"]["grid"] == [1.125, 1.25, 1.5, 2.0, 3.0]
    assert registration["dimension"]["source_status"] == ("exploratory_behavior_calibration")
    assert registration["execution_order"][-1].endswith("without consulting Figure 7.")
    assert "post-hoc exploratory" in registration["relationship_to_prior_registration"]
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_radial_annulus_gain_family_fails_before_figure7() -> None:
    artifact = yaml.safe_load(RADIAL_ANNULUS_GABA_GAIN_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["holdouts_consulted"] is False
    assert artifact["stage_1_survivor_scales"] == []
    assert artifact["stage_2_outcomes"] == []
    assert artifact["stage_2_survivor_scales"] == []
    assert not artifact["assessment"]["advance_to_same_network_match"]
    outcomes = artifact["stage_1_outcomes"]
    assert [outcome["scale"] for outcome in outcomes] == [
        1.125,
        1.25,
        1.5,
        2.0,
        3.0,
    ]
    assert all(outcome["population_spikes"]["thalamic_relay"] == 5 for outcome in outcomes)
    assert all(outcome["population_spikes"]["trn"] == 307 for outcome in outcomes)
    assert all(not outcome["pass"] for outcome in outcomes)


def test_two_event_current_match_is_the_sole_discrete_duration_intermediate() -> None:
    profile = yaml.safe_load(FIGURE7_TWO_EVENT_MATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TWO_EVENT_MATCH_REGISTRATION_PATH.read_text())
    assert profile["protocol"]["top_down_current_mode"] == ("until_cued_cell_event_limit")
    assert profile["protocol"]["top_down_current_event_limit"] == 2
    assert profile["dimension"]["grid"] == [1.0]
    assert registration["dimension"]["candidates"] == 1
    assert registration["dimension"]["current_termination_event_limit"] == 2
    assert registration["execution_order"][-1].startswith("Keep mismatch locked")
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_two_event_current_fails_exact_match_and_locks_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_TWO_EVENT_MATCH_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["stage_1_survivor_headroom_fractions"] == []
    assert artifact["assessment"]["mismatch_remains_locked"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert result["top_down_current_mode"] == "until_cued_cell_event_limit"
    assert result["top_down_current_event_limit"] == 2
    assert result["top_down_current_termination_time_ms"] == pytest.approx(49.88)
    assert len(result["relay_spike_times_ms"]) == 15
    assert len(result["trn_spike_times_ms"]) == 626
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert outcome["gates"]["current_terminated_on_selected_event"]
    assert not outcome["gates"]["nonspecific_40_hz"]
    assert not outcome["pass"]


def test_trn_arrival_alignment_is_source_derived_and_match_only() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_ARRIVAL_MATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TRN_ARRIVAL_MATCH_REGISTRATION_PATH.read_text())
    assert profile["protocol"]["top_down_cue_lead_ms"] == pytest.approx(9.85)
    assert profile["protocol"]["top_down_current_mode"] == ("until_cued_cell_first_event")
    source = registration["source_basis"]
    assert set(source["category_to_relay_delays_ms"].values()) == {2.0}
    assert source["category_to_trn_delays_ms"] == {
        "modeldb112923.projection.009": 3.0,
        "modeldb112923.projection.012": 4.0,
    }
    assert source["registered_slowest_trn_arrival_lead_ms"] == pytest.approx(
        source["selected_category_first_event_ms"] + 4.0
    )
    assert registration["execution_order"][-1].startswith("Keep mismatch locked")
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_trn_arrival_alignment_fails_exact_match_and_locks_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_ARRIVAL_MATCH_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["stage_1_survivor_headroom_fractions"] == []
    assert artifact["assessment"]["mismatch_remains_locked"]
    assert not artifact["assessment"]["advance_to_mismatch"]
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert result["top_down_cue_lead_ms"] == pytest.approx(9.85)
    assert result["cue_lead_category_spike_indices"] == [40]
    assert result["cue_lead_category_spike_times_ms"] == pytest.approx([5.85])
    assert result["cue_lead_relay_spike_times_ms"] == []
    assert len(result["relay_spike_times_ms"]) == 15
    assert len(result["trn_spike_times_ms"]) == 649
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert outcome["gates"]["current_terminated_on_selected_event"]
    assert not outcome["gates"]["nonspecific_40_hz"]
    assert not outcome["pass"]


def test_corticoreticular_gain_screen_is_finite_match_only_calibration() -> None:
    profile = yaml.safe_load(FIGURE7_CORTICORETICULAR_GAIN_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_CORTICORETICULAR_GAIN_REGISTRATION_PATH.read_text())
    assert profile["dimension"]["projection_ids"] == [
        "modeldb112923.projection.009",
        "modeldb112923.projection.012",
    ]
    assert profile["dimension"]["grid"] == [1.25, 1.5, 2.0, 4.0, 8.0]
    assert registration["dimension"]["source_status"] == (
        "post_holdout_exploratory_behavior_calibration"
    )
    assert registration["execution_order"][-1] == ("Do not inspect mismatch during this screen.")
    assert registration["fixed_choices"]["relay_to_trn_projection_010_unchanged"]
    assert not registration["official_status_before_execution"]["figure7_reproduced"]


def test_corticoreticular_gain_screen_selects_only_eightfold_match() -> None:
    artifact = yaml.safe_load(FIGURE7_CORTICORETICULAR_GAIN_RESULT_PATH.read_text())
    assert artifact["status"] == "complete"
    assert artifact["mismatch_consulted"] is False
    assert artifact["match_survivor_gains"] == [8.0]
    assert artifact["selected_gain"] == 8.0
    assert artifact["assessment"]["advance_to_full_state_match_verification"]
    assert artifact["assessment"]["mismatch_remains_locked"]
    outcomes = artifact["outcomes"]
    assert [outcome["common_gain"] for outcome in outcomes] == [
        1.25,
        1.5,
        2.0,
        4.0,
        8.0,
    ]
    assert [len(outcome["result"]["relay_spike_times_ms"]) for outcome in outcomes] == [
        10,
        10,
        10,
        15,
        15,
    ]
    assert [len(outcome["result"]["nonspecific_spike_times_ms"]) for outcome in outcomes] == [
        6,
        6,
        5,
        6,
        4,
    ]


def test_corticoreticular_verification_registers_only_screen_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_CORTICORETICULAR_VERIFICATION_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE7_CORTICORETICULAR_VERIFICATION_REGISTRATION_PATH.read_text()
    )
    assert profile["dimension"]["grid"] == [8.0]
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["screen_result"]["survivor_gains"] == [8.0]
    assert registration["fixed_candidate"]["independent_network_rebuild"]
    assert registration["diagnostic_gate"]["require_event_count_equals_release_transitions"]
    assert registration["mismatch_lock"].startswith("Do not run mismatch")


def test_corticoreticular_verification_authorizes_one_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_CORTICORETICULAR_VERIFICATION_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    assert artifact["verification_screen_artifact"] == str(
        FIGURE7_CORTICORETICULAR_GAIN_RESULT_PATH.relative_to(ROOT)
    )
    assert artifact["selected_gain"] == 8.0
    assert outcome["pass"]
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert artifact["assessment"]["advance_to_mismatch"]
    assert not artifact["assessment"]["mismatch_remains_locked"]


def test_corticoreticular_mismatch_is_fixed_to_verified_endpoint() -> None:
    profile = yaml.safe_load(FIGURE7_CORTICORETICULAR_MISMATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_CORTICORETICULAR_MISMATCH_REGISTRATION_PATH.read_text())
    assert profile["match_verification_artifact"] == str(
        FIGURE7_CORTICORETICULAR_VERIFICATION_RESULT_PATH.relative_to(ROOT)
    )
    assert profile["corticoreticular_common_gain"] == {
        "projection_ids": [
            "modeldb112923.projection.009",
            "modeldb112923.projection.012",
        ],
        "value": 8.0,
    }
    assert profile["protocol"]["condition"] == "mismatch"
    assert registration["registered_holdout"]["no_mismatch_tuning"]
    assert registration["execution_limit"].startswith("exactly one mismatch")


def test_mismatch_runner_keeps_recognition_gain_out_of_figure6() -> None:
    source = (ROOT / "scripts/run_figure7_aligned_on_center_mismatch.py").read_text()
    assert "training_scales = dict(scales)" in source
    assert "projection_weight_scales=training_scales" in source
    assert "persistent_projection_weight_scales=scales" in source


def test_corticoreticular_gain_endpoint_fails_mismatch_holdout() -> None:
    artifact = yaml.safe_load(FIGURE7_CORTICORETICULAR_MISMATCH_RESULT_PATH.read_text())
    mismatch = artifact["mismatch_result"]
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assert sorted(set(mismatch["relay_spike_indices"])) == [22, 31, 40, 49, 58]
    assert len(mismatch["relay_spike_times_ms"]) == 15
    assert len(mismatch["trn_spike_times_ms"]) == 565
    assert len(mismatch["nonspecific_spike_times_ms"]) == 5
    assert artifact["gates"]["match_more_trn_events"]
    assert artifact["gates"]["mismatch_more_nonspecific_events"]
    assert not artifact["gates"]["mismatch_relay_overlap_only"]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    assert not artifact["gates"]["sampled_mismatch_trn_events_have_fresh_cycles"]


def test_corticoreticular_margin_audit_is_readout_only() -> None:
    registration = yaml.safe_load(FIGURE7_CORTICORETICULAR_MARGIN_REGISTRATION_PATH.read_text())
    assert registration["source_holdout_artifact"] == str(
        FIGURE7_CORTICORETICULAR_MISMATCH_RESULT_PATH.relative_to(ROOT)
    )
    assert registration["candidate_changes"] == "none"
    assert registration["diagnostic_only"]
    assert registration["pre_event_offsets_ms"] == [2.0, 1.0, 0.5, 0.2]
    assert registration["sampled_relay_indices"] == [22, 31, 40, 49, 58]


def test_corticoreticular_margin_audit_localizes_late_inhibition() -> None:
    artifact = yaml.safe_load(FIGURE7_CORTICORETICULAR_MARGIN_RESULT_PATH.read_text())
    mismatch = artifact["mismatch_result"]
    assert artifact["original_holdout_registration_artifact"].endswith(
        "figure7-corticoreticular-mismatch-registration-263.yaml"
    )
    assert len(mismatch["relay_spike_times_ms"]) == 15
    first_event_by_index: dict[int, float] = {}
    for index, time_ms in zip(
        mismatch["relay_spike_indices"],
        mismatch["relay_spike_times_ms"],
        strict=True,
    ):
        first_event_by_index.setdefault(index, time_ms)
    current = {
        (index, offset_ms, source): value
        for index, event_time_ms, offset_ms, source, value in mismatch[
            "relay_pre_event_current_samples_pA"
        ]
        if event_time_ms == first_event_by_index[index]
    }
    voltage = {
        (index, offset_ms, source): value
        for index, event_time_ms, offset_ms, source, value in mismatch[
            "relay_pre_event_voltage_samples_mV"
        ]
        if event_time_ms == first_event_by_index[index]
    }
    for index in (22, 31, 49, 58):
        early_depolarizing = (
            current[index, 2.0, "direct_image_input"] + current[index, 2.0, "soma_axial"]
        )
        assert early_depolarizing > abs(current[index, 2.0, "trn_gaba"])
        assert current[index, 0.5, "soma_sodium"] > 4_000.0
        assert voltage[index, 0.5, "soma"] > -44.0


def test_targeted_annular_screen_changes_only_corticoreticular_ring_geometry() -> None:
    profile = yaml.safe_load(FIGURE7_TARGETED_ANNULAR_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_TARGETED_ANNULAR_REGISTRATION_PATH.read_text())
    assert (
        profile["runtime_overrides"]["corticoreticular_ring_kernel_convention"] == "radial_annulus"
    )
    assert profile["dimension"]["grid"] == [1.0, 8.0]
    assert registration["fixed_topology_change"]["unrelated_ring_projections_unchanged"]
    assert registration["conditions_consulted"] == ["figure7_match"]
    assert registration["mismatch_lock"].startswith("do not inspect mismatch")


def test_targeted_annular_screen_closes_without_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_TARGETED_ANNULAR_RESULT_PATH.read_text())
    assert artifact["match_survivor_gains"] == []
    assert artifact["selected_gain"] is None
    assert artifact["mismatch_consulted"] is False
    assert artifact["assessment"]["mismatch_remains_locked"]
    outcomes = artifact["outcomes"]
    assert [len(item["result"]["relay_spike_times_ms"]) for item in outcomes] == [
        10,
        10,
    ]
    assert [len(item["result"]["trn_spike_times_ms"]) for item in outcomes] == [
        597,
        692,
    ]
    assert [len(item["result"]["nonspecific_spike_times_ms"]) for item in outcomes] == [5, 3]


def test_gain8_ampa_arrival_interaction_is_single_match_only_endpoint() -> None:
    profile = yaml.safe_load(FIGURE7_GAIN8_AMPA_ARRIVAL_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_GAIN8_AMPA_ARRIVAL_REGISTRATION_PATH.read_text())
    assert profile["dimension"]["grid"] == [8.0]
    assert profile["protocol"]["top_down_cue_lead_ms"] == 9.85
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["fixed_candidate"]["source_delays_changed"] is False
    assert registration["fixed_candidate"]["source_kinetics_changed"] is False
    assert registration["conditions_consulted"] == ["figure7_match"]
    assert registration["execution_limit"].startswith("exactly one")


def test_gain8_ampa_arrival_interaction_fails_without_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_GAIN8_AMPA_ARRIVAL_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert not outcome["pass"]
    assert artifact["mismatch_consulted"] is False
    assert artifact["assessment"]["mismatch_remains_locked"]
    assert sorted(set(result["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert len(result["relay_spike_times_ms"]) == 15
    assert len(result["trn_spike_times_ms"]) == 594
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert len(result["cue_lead_trn_spike_times_ms"]) == 81
    assert len(result["cue_lead_nonspecific_spike_times_ms"]) == 1
    assert not outcome["gates"]["sampled_trn_events_have_fresh_cycles"]


def test_corticoreticular_ampa_delay_requires_complete_figure6_first() -> None:
    profile = yaml.safe_load(FIGURE6_CORTICORETICULAR_AMPA_DELAY_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE6_CORTICORETICULAR_AMPA_DELAY_REGISTRATION_PATH.read_text())
    assert profile["runtime_overrides"]["corticoreticular_ampa_delay_ms"] == 2.0
    assert registration["timing_alternative"]["projection_id"].endswith(".012")
    assert registration["timing_alternative"]["unchanged_delays"] == {
        "modeldb112923.projection.003": 2.0,
        "modeldb112923.projection.009": 3.0,
    }
    assert registration["conditions_consulted"] == ["figure6"]
    assert registration["figure7_lock"].startswith("do not inspect Figure 7")


def test_corticoreticular_ampa_delay2_first_run_is_detector_mismatched() -> None:
    artifact = yaml.safe_load(FIGURE6_CORTICORETICULAR_AMPA_DELAY_RESULT_PATH.read_text())
    assert artifact["status"] == "partial-figure6b-pass-figure6c-fail"
    assert artifact["population_spikes"]["thalamic_relay"] == 104
    assert artifact["relay_recruitment"]["active_indices"] == list(range(81))
    assert not artifact["relay_recruitment"]["confined_to_horizontal_bar_at_40_hz"]
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert not artifact["maps"]["top_down_oriented"]
    assert not artifact["assessment"]["promoted"]
    profile = yaml.safe_load(FIGURE6_CORTICORETICULAR_AMPA_DELAY_PROFILE_PATH.read_text())
    assert "detector" not in profile


def test_corticoreticular_ampa_delay3_is_final_integer_intermediate() -> None:
    profile = yaml.safe_load(FIGURE6_CORTICORETICULAR_AMPA_DELAY3_PROFILE_PATH.read_text())
    registration = yaml.safe_load(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY3_REGISTRATION_PATH.read_text()
    )
    assert profile["runtime_overrides"]["corticoreticular_ampa_delay_ms"] == 3.0
    assert registration["timing_alternative"]["candidate_delay_ms"] == 3.0
    assert registration["conditions_consulted"] == ["figure6"]
    assert registration["figure7_lock"].startswith("do not inspect Figure 7")
    assert registration["selection_boundary"].startswith("Do not interpolate")


def test_delay2_and_delay3_first_runs_share_invalid_default_detector_trace() -> None:
    delay2 = yaml.safe_load(FIGURE6_CORTICORETICULAR_AMPA_DELAY_RESULT_PATH.read_text())
    delay3 = yaml.safe_load(FIGURE6_CORTICORETICULAR_AMPA_DELAY3_RESULT_PATH.read_text())
    assert delay2["population_spikes"] == delay3["population_spikes"]
    assert delay2["relay_recruitment"] == delay3["relay_recruitment"]
    assert delay2["runtime_fingerprint"] != delay3["runtime_fingerprint"]
    assert delay2["population_spikes"]["trn"] == 81


def test_corrected_delay2_uses_exact_two_stage_detector_prerequisite() -> None:
    profile = yaml.safe_load(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY2_CORRECTED_PROFILE_PATH.read_text()
    )
    registration = yaml.safe_load(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY2_CORRECTED_REGISTRATION_PATH.read_text()
    )
    assert profile["detector"] == {"arm_mV": -20.0, "release_mV": -30.0}
    assert profile["runtime_overrides"]["corticoreticular_ampa_delay_ms"] == 2.0
    assert profile["stage_1_protocol"]["stimulus_ms"] == 55.0
    assert profile["stage_2_protocol"]["stimulus_ms"] == 100.0
    assert registration["complete_figure6_first"]
    assert registration["figure7_lock"].startswith("do not inspect Figure 7")


def test_corrected_delay2_passes_complete_figure6() -> None:
    artifact = yaml.safe_load(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY2_CORRECTED_RESULT_PATH.read_text()
    )
    stage1 = artifact["stage_1_outcomes"][0]
    stage2 = artifact["stage_2_outcomes"][0]
    assert stage1["pass"]
    assert stage1["population_spikes"]["thalamic_relay"] == 10
    assert stage1["population_spikes"]["trn"] == 368
    assert stage2["pass"]
    assert stage2["population_spikes"]["thalamic_relay"] == 20
    assert stage2["population_spikes"]["trn"] == 565
    assert all(stage2["gates"].values())
    assert artifact["assessment"]["advance_to_figure7"]


def test_delay2_gain8_match_is_fixed_to_complete_figure6_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_DELAY2_GAIN8_MATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_DELAY2_GAIN8_MATCH_REGISTRATION_PATH.read_text())
    assert profile["figure6_artifact"] == str(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY2_CORRECTED_RESULT_PATH.relative_to(ROOT)
    )
    assert profile["runtime_overrides"]["corticoreticular_ampa_delay_ms"] == 2.0
    assert profile["dimension"]["grid"] == [8.0]
    assert profile["protocol"]["top_down_cue_lead_ms"] == 7.85
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["figure6_prerequisite_passed"]
    assert registration["mismatch_lock"].startswith("do not inspect mismatch")


def test_delay2_gain8_fails_exact_match_without_mismatch() -> None:
    artifact = yaml.safe_load(FIGURE7_DELAY2_GAIN8_MATCH_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert not outcome["pass"]
    assert artifact["mismatch_consulted"] is False
    assert artifact["assessment"]["mismatch_remains_locked"]
    assert sorted(set(result["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert len(result["relay_spike_times_ms"]) == 10
    assert len(result["trn_spike_times_ms"]) == 623
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert not outcome["gates"]["nonspecific_40_hz"]


def test_corrected_delay3_is_final_exact_two_stage_prerequisite() -> None:
    profile = yaml.safe_load(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY3_CORRECTED_PROFILE_PATH.read_text()
    )
    registration = yaml.safe_load(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY3_CORRECTED_REGISTRATION_PATH.read_text()
    )
    assert profile["detector"] == {"arm_mV": -20.0, "release_mV": -30.0}
    assert profile["runtime_overrides"]["corticoreticular_ampa_delay_ms"] == 3.0
    assert registration["complete_figure6_first"]
    assert registration["selection_boundary"].startswith("no fractional")
    assert registration["figure7_lock"].startswith("do not inspect Figure 7")


def test_corrected_delay3_passes_complete_figure6() -> None:
    artifact = yaml.safe_load(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY3_CORRECTED_RESULT_PATH.read_text()
    )
    stage1 = artifact["stage_1_outcomes"][0]
    stage2 = artifact["stage_2_outcomes"][0]
    assert stage1["pass"]
    assert stage1["population_spikes"]["thalamic_relay"] == 10
    assert stage1["population_spikes"]["trn"] == 368
    assert stage2["pass"]
    assert stage2["population_spikes"]["thalamic_relay"] == 20
    assert stage2["population_spikes"]["trn"] == 552
    assert all(stage2["gates"].values())
    assert artifact["assessment"]["advance_to_figure7"]


def test_delay3_gain8_match_is_final_integer_endpoint() -> None:
    profile = yaml.safe_load(FIGURE7_DELAY3_GAIN8_MATCH_PROFILE_PATH.read_text())
    registration = yaml.safe_load(FIGURE7_DELAY3_GAIN8_MATCH_REGISTRATION_PATH.read_text())
    assert profile["figure6_artifact"] == str(
        FIGURE6_CORTICORETICULAR_AMPA_DELAY3_CORRECTED_RESULT_PATH.relative_to(ROOT)
    )
    assert profile["runtime_overrides"]["corticoreticular_ampa_delay_ms"] == 3.0
    assert profile["dimension"]["grid"] == [8.0]
    assert profile["protocol"]["record_relay_diagnostics"]
    assert registration["figure6_prerequisite_passed"]
    assert registration["selection_boundary"].startswith("no fractional")
    assert registration["mismatch_lock"].startswith("do not inspect mismatch")


def test_delay3_gain8_fails_exact_match_and_closes_integer_family() -> None:
    artifact = yaml.safe_load(FIGURE7_DELAY3_GAIN8_MATCH_RESULT_PATH.read_text())
    outcome = artifact["outcomes"][0]
    result = outcome["result"]
    assert not outcome["pass"]
    assert artifact["mismatch_consulted"] is False
    assert artifact["assessment"]["mismatch_remains_locked"]
    assert sorted(set(result["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert len(result["relay_spike_times_ms"]) == 14
    assert len(result["trn_spike_times_ms"]) == 601
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert not outcome["gates"]["nonspecific_40_hz"]


def test_contract_rejects_holdout_leakage(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text())
    raw["training_targets"].append(raw["holdout_targets"][0])
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="targets overlap"):
        load_calibration_contract(path)


def test_contract_rejects_published_free_parameter(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text())
    raw["dimensions"]["top_down_current_pA"]["status"] = "published"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="not calibration-admissible"):
        load_calibration_contract(path)


def test_projection012_kinetics_are_closed_by_source_semantics() -> None:
    artifact = yaml.safe_load(
        (
            ROOT / "docs/validation-results/kinness-projection012-kinetics-semantics-283.yaml"
        ).read_text()
    )
    assert artifact["status"] == "source-semantics-resolved-no-calibration-authorized"
    assert artifact["implementation_audit"] == {
        "normalized_dual_exponential": "implemented",
        "last_two_arrivals_only": "implemented",
        "bounded_pair_combination": "implemented",
    }
    assert not artifact["assessment"]["semantic_ambiguity_remaining"]
    assert not artifact["assessment"]["source_faithful_kinetic_sweep_authorized"]


def test_headroom_corticoreticular_surface_has_one_exact_match_survivor() -> None:
    artifact = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-headroom-corticoreticular-interaction-match-285.yaml"
        ).read_text()
    )
    assert artifact["assessment"]["registered_candidate_count"] == 15
    assert artifact["assessment"]["completed_candidate_count"] == 15
    assert artifact["match_survivors"] == [{"headroom_fraction": 0.75, "common_gain": 4.0}]
    survivor = next(item for item in artifact["outcomes"] if item["pass"])
    assert len(survivor["result"]["relay_spike_times_ms"]) == 15
    assert len(survivor["result"]["trn_spike_times_ms"]) == 528
    assert len(survivor["result"]["nonspecific_spike_times_ms"]) == 4
    assert artifact["assessment"]["mismatch_remains_locked"]


def test_headroom_corticoreticular_match_verification_has_fresh_cycles() -> None:
    artifact = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-headroom-corticoreticular-verification-287.yaml"
        ).read_text()
    )
    outcome = artifact["outcomes"][0]
    assert outcome["pass"]
    assert outcome["headroom_fraction"] == 0.75
    assert outcome["common_gain"] == 4.0
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert artifact["assessment"]["advance_to_mismatch"]


def test_headroom_corticoreticular_pair_fails_official_mismatch() -> None:
    artifact = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-headroom-corticoreticular-pair-289.yaml"
        ).read_text()
    )
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assert artifact["gates"]["sampled_mismatch_trn_events_have_fresh_cycles"]
    assert not artifact["gates"]["mismatch_relay_overlap_only"]
    assert not artifact["gates"]["match_more_active_relay_cells"]
    assert not artifact["gates"]["match_more_trn_events"]
    assert not artifact["gates"]["mismatch_nonspecific_70_hz"]
    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    assert len(match["relay_spike_times_ms"]) == 15
    assert len(match["trn_spike_times_ms"]) == 528
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert sorted(set(mismatch["relay_spike_indices"])) == [22, 31, 40, 49, 58]
    assert all(mismatch["relay_spike_indices"].count(index) == 3 for index in [22, 31, 40, 49, 58])
    assert len(mismatch["trn_spike_times_ms"]) == 586
    assert len(mismatch["nonspecific_spike_times_ms"]) == 5


def test_annular_headroom_gain_interaction_is_bounded_before_execution() -> None:
    profile = yaml.safe_load(
        (
            ROOT
            / "configs/calibration/figure7_annular_headroom_corticoreticular_interaction_match_v1.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-annular-headroom-corticoreticular-interaction-registration-290.yaml"
        ).read_text()
    )
    assert (
        profile["runtime_overrides"]["corticoreticular_ring_kernel_convention"] == "radial_annulus"
    )
    assert profile["learned_state"]["headroom_grid"] == [0.25, 0.5, 0.75]
    assert profile["dimension"]["grid"] == [1.25, 1.5, 2.0, 4.0, 8.0]
    assert registration["registered_surface"]["candidate_count"] == 15
    assert registration["fixed"]["projection012_delay_ms"] == 4.0
    assert registration["fixed"]["projection012_rise_fall_ms"] == [4.0, 5.0]
    assert registration["fixed"]["unrelated_ring_projections_unchanged"]
    assert registration["locked_holdouts"][0] == "figure7_mismatch"
    assert "without inventing another ring radius" in registration["stopping_rule"]


def test_annular_headroom_gain_surface_selects_one_verification_candidate() -> None:
    artifact = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-annular-headroom-corticoreticular-interaction-match-291.yaml"
        ).read_text()
    )
    verification = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-annular-headroom-corticoreticular-verification-registration-292.yaml"
        ).read_text()
    )
    assert artifact["assessment"]["registered_candidate_count"] == 15
    assert artifact["assessment"]["completed_candidate_count"] == 15
    assert artifact["match_survivors"] == [{"headroom_fraction": 0.75, "common_gain": 8.0}]
    survivor = next(item for item in artifact["outcomes"] if item["pass"])
    assert len(survivor["result"]["relay_spike_times_ms"]) == 10
    assert sorted(set(survivor["result"]["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert len(survivor["result"]["trn_spike_times_ms"]) == 708
    assert len(survivor["result"]["nonspecific_spike_times_ms"]) == 4
    assert verification["selected_candidate"] == {
        "ring_kernel": "radial_annulus",
        "learned_headroom_fraction": 0.75,
        "common_corticoreticular_gain": 8.0,
    }


def test_annular_candidate_verifies_before_single_mismatch() -> None:
    artifact = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-annular-headroom-corticoreticular-verification-293.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-annular-headroom-corticoreticular-mismatch-registration-294.yaml"
        ).read_text()
    )
    outcome = artifact["outcomes"][0]
    assert outcome["pass"]
    assert outcome["gates"]["sampled_trn_events_have_fresh_cycles"]
    assert artifact["assessment"]["advance_to_mismatch"]
    assert registration["authorization"] == {
        "independently_verified_match": True,
        "match_counts": {
            "relay_events": 10,
            "trn_events": 708,
            "nonspecific_events": 4,
        },
        "sampled_trn_detector_cycles_complete": True,
    }
    assert registration["execution"]["run_count"] == 1
    assert registration["fixed_candidate"]["expected_common_weight_factor"] == pytest.approx(
        artifact["outcomes"][0]["applied_common_weight_factor"]
    )
    assert registration["registration_correction"]["status"] == (
        "corrected-before-holdout-execution"
    )
    assert "Do not alter ring geometry" in registration["stopping_rule"]


def test_annular_candidate_fails_fixed_official_mismatch() -> None:
    artifact = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-annular-headroom-corticoreticular-pair-295.yaml"
        ).read_text()
    )
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assert artifact["gates"]["sampled_mismatch_trn_events_have_fresh_cycles"]
    for failed_gate in (
        "mismatch_relay_overlap_only",
        "match_more_active_relay_cells",
        "match_more_trn_events",
        "mismatch_more_nonspecific_events",
        "mismatch_nonspecific_70_hz",
    ):
        assert not artifact["gates"][failed_gate]
    match = artifact["match_scoring_summary"]
    mismatch = artifact["mismatch_result"]
    assert len(match["relay_spike_times_ms"]) == 10
    assert len(match["trn_spike_times_ms"]) == 708
    assert len(match["nonspecific_spike_times_ms"]) == 4
    assert sorted(set(mismatch["relay_spike_indices"])) == [22, 31, 40, 49, 58]
    assert all(mismatch["relay_spike_indices"].count(index) == 2 for index in [22, 31, 40, 49, 58])
    assert len(mismatch["trn_spike_times_ms"]) == 708
    assert len(mismatch["nonspecific_spike_times_ms"]) == 3


def test_declared_input_simultaneous_match_retains_startup_failure_and_transfer() -> None:
    result_path = ROOT / "docs/validation-results/declared-input-simultaneous-match-425.yaml"
    assessment_path = (
        ROOT / "docs/validation-results/declared-input-simultaneous-match-assessment-426.yaml"
    )
    artifact = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(assessment_path.read_text())
    result = artifact["match_result"]

    assert artifact["training_repeat_verified"]
    assert sorted(set(result["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert all(result["relay_spike_indices"].count(index) == 4 for index in range(38, 43))
    assert result["interneuron_spike_indices"] == [38, 39, 40, 41, 42]
    assert len(result["trn_spike_times_ms"]) == 605
    assert result["nonspecific_spike_times_ms"] == pytest.approx([0.74, 50.36, 56.04, 73.81, 94.08])
    assert not artifact["match_prerequisites_pass"]
    assert not artifact["gates"]["nonspecific_40_hz"]

    transfer = assessment["interneuron_to_relay_transfer_audit"]
    assert transfer["compiled_direct_assay"]["center_postsynaptic_gate_peak"] > 1.8
    assert transfer["compiled_direct_assay"]["center_postsynaptic_current_min_pA"] < -800
    assert transfer["match_event_samples"]["outer_active_cells_38_42_current_pA_at_6_03_ms"] < -700
    assert not assessment["assessment"]["original_smart_reproduced"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_declared_input_simultaneous_mismatch_is_only_a_localization_run() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/declared-input-simultaneous-mismatch-registration-427.yaml"
        ).read_text()
    )
    assert registration["prior_observation"]["match_pass"] is False
    assert registration["protocol"]["condition"] == "mismatch"
    assert registration["protocol"]["top_down_cue_lead_ms"] == 0.0
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "mismatch_runs": 1,
        "parameter_search": False,
    }
    assert "not an independent holdout" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_declared_input_zero_lead_pair_precedes_feedback_arrival() -> None:
    pair = yaml.safe_load(
        (ROOT / "docs/validation-results/declared-input-simultaneous-pair-428.yaml").read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/declared-input-simultaneous-pair-assessment-429.yaml"
        ).read_text()
    )
    mismatch = pair["mismatch_result"]

    assert sorted(set(mismatch["relay_spike_indices"])) == [22, 31, 40, 49, 58]
    assert all(mismatch["relay_spike_indices"].count(index) == 4 for index in [22, 31, 40, 49, 58])
    assert len(mismatch["trn_spike_times_ms"]) == 603
    assert mismatch["nonspecific_spike_times_ms"] == pytest.approx(
        [0.74, 50.23, 55.94, 74.11, 94.09]
    )
    assert pair["gates"]["match_more_trn_events"]
    for gate in (
        "mismatch_overlap_relay_set",
        "match_more_active_relay_cells",
        "match_nonspecific_40_hz",
        "mismatch_more_nonspecific_events",
        "mismatch_nonspecific_70_hz",
    ):
        assert not pair["gates"][gate]

    timing = assessment["timing_audit"]
    assert timing["earliest_learned_feedback_arrival_ms"] == pytest.approx(7.85)
    assert timing["first_relay_events_ms"] == pytest.approx([6.03, 6.05])
    assert timing["top_down_excitation_current_pA_at_first_relay_events"] == 0.0
    assert not assessment["assessment"]["baseline_promoted"]


def test_arrival_aligned_sustained_match_is_registered_before_execution() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/declared-input-arrival-aligned-sustained-match-registration-430.yaml"
        ).read_text()
    )
    assert registration["source_constraints"]["derived_command_lead_ms"] == 7.85
    assert registration["single_protocol_change_from_417"] == {
        "top_down_current_mode": {
            "from": "until_cued_cell_first_event",
            "to": "sustained_epoch",
        }
    }
    assert registration["protocol"]["top_down_cue_lead_ms"] == 7.85
    assert registration["protocol"]["top_down_current_mode"] == "sustained_epoch"
    assert registration["execution"]["mismatch_runs"] == 0
    assert registration["baseline_promoted"] is False


def test_arrival_aligned_sustained_match_fails_and_localization_is_bounded() -> None:
    match = yaml.safe_load(
        (
            ROOT / "docs/validation-results/declared-input-arrival-aligned-sustained-match-431.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/declared-input-arrival-aligned-sustained-assessment-432.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/declared-input-arrival-aligned-sustained-mismatch-registration-433.yaml"
        ).read_text()
    )
    result = match["match_result"]

    assert sorted(set(result["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert all(result["relay_spike_indices"].count(index) == 2 for index in range(38, 43))
    assert len(result["trn_spike_times_ms"]) == 564
    assert result["nonspecific_spike_times_ms"] == pytest.approx(
        [32.39, 45.18, 58.24, 71.85, 91.86]
    )
    assert result["top_down_current_termination_time_ms"] is None
    assert not match["match_prerequisites_pass"]
    assert assessment["assessment"]["current_duration_family_closed"]
    assert not registration["prior_observation"]["match_pass"]
    assert registration["protocol"]["top_down_current_mode"] == "sustained_epoch"
    assert "not an independent holdout" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_arrival_aligned_sustained_pair_rejects_overlap_hypothesis() -> None:
    pair = yaml.safe_load(
        (
            ROOT / "docs/validation-results/declared-input-arrival-aligned-sustained-pair-434.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/declared-input-arrival-aligned-sustained-pair-assessment-435.yaml"
        ).read_text()
    )
    mismatch = pair["mismatch_result"]

    assert pair["training_repeat_verified"]
    assert sorted(set(mismatch["relay_spike_indices"])) == [22, 31, 40, 49, 58]
    assert all(mismatch["relay_spike_indices"].count(index) == 2 for index in [22, 31, 40, 49, 58])
    assert len(mismatch["trn_spike_times_ms"]) == 568
    assert mismatch["nonspecific_spike_times_ms"] == pytest.approx(
        [32.39, 45.18, 58.24, 72.25, 92.24]
    )
    for failed_gate in (
        "mismatch_overlap_relay_set",
        "match_more_active_relay_cells",
        "match_more_trn_events",
        "match_nonspecific_40_hz",
        "mismatch_more_nonspecific_events",
        "mismatch_nonspecific_70_hz",
    ):
        assert not pair["gates"][failed_gate]
    assert assessment["timing_localization"]["startup_trn_volley"] == {
        "event_count": 81,
        "time_ms": 5.68,
        "relative_to_sensory_onset_ms": -2.17,
    }
    assert assessment["assessment"]["arrival_aligned_sustained_hypothesis_rejected"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_uniform_relay_input_gain_screen_is_bounded_and_not_a_comparator() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-uniform-relay-input-gain-screen-registration-436.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["protocol"]["uniform_relay_input_gains"] == [1.0, 0.75, 0.5, 0.25]
    assert profile["protocol"]["duration_ms"] == 25.0
    assert profile["protocol"]["top_down_cue_lead_ms"] == 0.0
    assert registration["screen"]["selection_rule"] == (
        "strongest gain passing every spatial/pathway gate"
    )
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "short_match_runs": 4,
        "short_mismatch_runs": 4,
    }
    assert "not recovered SMART parameters" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_uniform_relay_input_gain_screen_has_no_survivor() -> None:
    result = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-uniform-relay-input-gain-screen-437.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-uniform-relay-input-gain-screen-assessment-438.yaml"
        ).read_text()
    )

    assert result["training_repeat_verified"]
    assert result["surviving_gains"] == []
    assert result["selected_gain"] is None
    by_gain = {item["uniform_relay_input_gain"]: item for item in result["outcomes"]}
    assert sorted(set(by_gain[1.0]["match"]["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert sorted(set(by_gain[1.0]["mismatch"]["relay_spike_indices"])) == [22, 31, 40, 49, 58]
    for gain in (0.75, 0.5, 0.25):
        assert by_gain[gain]["match"]["relay_spike_indices"] == []
        assert by_gain[gain]["mismatch"]["relay_spike_indices"] == []
        assert len(by_gain[gain]["match"]["trn_spike_indices"]) == 243
        assert len(by_gain[gain]["mismatch"]["trn_spike_indices"]) == 243
    assert assessment["assessment"]["uniform_relay_input_gain_family_closed"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_declared_input_headroom_endpoint_is_single_and_source_bounded() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-declared-input-headroom-endpoint-registration-439.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["learned_state"] == {"source_index": 40, "headroom_fraction": 1.0}
    assert profile["protocol"]["uniform_relay_input_gain"] == 1.0
    assert profile["protocol"]["top_down_cue_lead_ms"] == 0.0
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "short_match_runs": 1,
        "short_mismatch_runs": 1,
    }
    assert registration["candidate"]["selection"] == (
        "sole archived-bound endpoint; no grid or interpolation"
    )
    assert "not the actual Figure 6 learned state" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_declared_input_headroom_endpoint_fails_before_feedback_arrival() -> None:
    result = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-declared-input-headroom-endpoint-440.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-declared-input-headroom-endpoint-assessment-441.yaml"
        ).read_text()
    )

    assert result["applied_common_weight_factor"] == pytest.approx(3.6531686628985414)
    assert sorted(set(result["match"]["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert sorted(set(result["mismatch"]["relay_spike_indices"])) == [22, 31, 40, 49, 58]
    assert result["match"]["relay_spike_times_ms"] == pytest.approx([6.03, 6.03, 6.05, 6.05, 6.05])
    assert result["mismatch"]["relay_spike_times_ms"] == pytest.approx(
        [6.03, 6.03, 6.05, 6.05, 6.05]
    )
    assert not result["mechanism_screen_pass"]
    assert assessment["assessment"]["maximal_bounded_headroom_rejected"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_interneuron_gaba_gain_screen_is_finite_and_recognition_only() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-interneuron-gaba-gain-screen-registration-442.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["pathway"]["projection_id"] == "modeldb112923.projection.002"
    assert profile["pathway"]["gains"] == [1.0, 2.0, 4.0, 8.0]
    assert profile["protocol"]["uniform_relay_input_gain"] == 1.0
    assert registration["screen"]["selection_rule"] == (
        "weakest gain passing every spatial/pathway gate"
    )
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "short_match_runs": 4,
        "short_mismatch_runs": 4,
    }
    assert "recognition-only" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_interneuron_gaba_gain_screen_has_no_match_selective_survivor() -> None:
    result_path = ROOT / "docs/validation-results/figure7-interneuron-gaba-gain-screen-443.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-interneuron-gaba-gain-screen-assessment-444.yaml"
        ).read_text()
    )

    assert result["surviving_gains"] == []
    assert result["selected_gain"] is None
    by_gain = {item["projection_002_gain"]: item for item in result["outcomes"]}
    for gain in (1.0, 2.0):
        assert sorted(set(by_gain[gain]["match"]["relay_spike_indices"])) == [
            38,
            39,
            40,
            41,
            42,
        ]
        assert sorted(set(by_gain[gain]["mismatch"]["relay_spike_indices"])) == [
            22,
            31,
            40,
            49,
            58,
        ]
    for gain in (4.0, 8.0):
        assert by_gain[gain]["match"]["relay_spike_indices"] == []
        assert by_gain[gain]["mismatch"]["relay_spike_indices"] == []
    for item in result["outcomes"]:
        assert len(item["match"]["trn_spike_indices"]) == len(item["mismatch"]["trn_spike_indices"])
        assert item["match"]["interneuron_spike_indices"]
        assert item["mismatch"]["interneuron_spike_indices"]
        assert not item["pass"]
    assert assessment["result_sha256"] == (
        "7ebad431a6b05d991fd25e50f3913be6a2a6836254b93908a39d9fd3f2c96fcc"
    )
    assert assessment["assessment"]["projection_002_gain_grid_closed"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_declared_input_current_recheck_is_bounded_and_single_factor() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-declared-input-top-down-current-recheck-registration-445.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["dimension"] == {
        "name": "top_down_current_pA",
        "source_status": "not_identifiable",
        "previously_registered_bounds_pA": [600.0, 1000.0],
        "fixed_control_pA": 800.0,
        "endpoints_to_recheck_pA": [600.0, 1000.0],
    }
    assert profile["protocol"] == {
        "duration_ms": 25.0,
        "dt_ms": 0.01,
        "top_down_current_mode": "until_cued_cell_first_event",
        "top_down_cue_lead_ms": 0.0,
        "uniform_relay_input_gain": 1.0,
        "projection_002_gain": 1.0,
        "learned_feedback_delay_ms": 2.0,
    }
    assert registration["screen"]["fixed_control_observation"] == {
        "category_first_event_ms": 5.85,
        "learned_feedback_first_arrival_ms": 7.85,
        "relay_first_event_ms": 6.03,
        "spatial_pair_pass": False,
    }
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "short_match_runs": 2,
        "short_mismatch_runs": 2,
    }
    assert "Do not interpolate" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_declared_input_current_recheck_closes_every_bounded_endpoint() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-declared-input-top-down-current-recheck-446.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-declared-input-top-down-current-recheck-assessment-447.yaml"
        ).read_text()
    )

    assert result["surviving_currents_pA"] == []
    assert result["selected_current_pA"] is None
    outcomes = {item["top_down_current_pA"]: item for item in result["outcomes"]}
    assert outcomes[600.0]["timing_ms"] == {
        "match_category_first_event": 8.92,
        "match_feedback_first_arrival": 10.92,
        "match_relay_first_event": 6.03,
        "mismatch_category_first_event": 8.92,
        "mismatch_feedback_first_arrival": 10.92,
        "mismatch_relay_first_event": 6.03,
    }
    assert outcomes[1000.0]["timing_ms"] == {
        "match_category_first_event": 4.49,
        "match_feedback_first_arrival": 6.49,
        "match_relay_first_event": 6.03,
        "mismatch_category_first_event": 4.49,
        "mismatch_feedback_first_arrival": 6.49,
        "mismatch_relay_first_event": 6.03,
    }
    for item in outcomes.values():
        assert not item["gates"]["learned_feedback_arrives_before_first_relay_event"]
        assert sorted(set(item["match"]["relay_spike_indices"])) == [
            38,
            39,
            40,
            41,
            42,
        ]
        assert sorted(set(item["mismatch"]["relay_spike_indices"])) == [
            22,
            31,
            40,
            49,
            58,
        ]
        assert len(item["match"]["trn_spike_indices"]) == 194
        assert len(item["mismatch"]["trn_spike_indices"]) == 194
        assert not item["pass"]
    assert assessment["result_sha256"] == (
        "c0aa1a38b1f262cff71aebd3eadfd707f26ebea5288512b2798a561049abd961"
    )
    assert assessment["assessment"]["source_bounded_current_family_closed"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_receptor_simultaneity_diagnostic_is_one_factor_and_nonpromotable() -> None:
    registration = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-receptor-simultaneity-registration-448.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["intervention"]["source_index"] == 40
    assert profile["intervention"]["projection_ids"] == [
        "modeldb112923.projection.003",
        "modeldb112923.projection.005",
        "modeldb112923.projection.006",
        "modeldb112923.projection.007",
        "modeldb112923.projection.009",
        "modeldb112923.projection.012",
    ]
    assert profile["protocol"] == {
        "duration_ms": 25.0,
        "dt_ms": 0.01,
        "top_down_current_pA": 800.0,
        "top_down_current_mode": "until_cued_cell_first_event",
        "top_down_cue_lead_ms": 0.0,
        "uniform_relay_input_gain": 1.0,
        "projection_002_gain": 1.0,
    }
    assert registration["intervention"]["modifies_weights"] is False
    assert registration["intervention"]["modifies_delays"] is False
    assert registration["intervention"]["recovered_legacy_state"] is False
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "short_match_runs": 1,
        "short_mismatch_runs": 1,
    }
    assert "not source recovery" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_receptor_simultaneity_prime_silences_both_conditions() -> None:
    result_path = ROOT / "docs/validation-results/figure7-receptor-simultaneity-diagnostic-449.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-receptor-simultaneity-assessment-450.yaml"
        ).read_text()
    )

    assert result["gates"]["all_six_expectation_records_primed"]
    assert result["match"]["relay_spike_indices"] == []
    assert result["mismatch"]["relay_spike_indices"] == []
    assert len(result["match"]["trn_spike_indices"]) == 200
    assert len(result["mismatch"]["trn_spike_indices"]) == 200
    assert result["match"]["category_spike_times_ms"] == pytest.approx([5.85])
    assert result["mismatch"]["category_spike_times_ms"] == pytest.approx([5.85])
    assert not result["mechanism_screen_pass"]
    assert assessment["result_sha256"] == (
        "57cbc3c8805734c6ea88351fed7516ea856f0776f2339182cf0d7f2384b9ac1e"
    )
    assert assessment["assessment"]["combined_prime_closed_without_tuning"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_receptor_prime_decomposition_has_two_fixed_disjoint_arms() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-receptor-prime-decomposition-registration-451.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())
    relay_ids = profile["arms"]["relay_on_center_only"]["projection_ids"]
    trn_ids = profile["arms"]["trn_off_surround_only"]["projection_ids"]

    assert relay_ids == [
        "modeldb112923.projection.003",
        "modeldb112923.projection.005",
        "modeldb112923.projection.006",
        "modeldb112923.projection.007",
    ]
    assert trn_ids == [
        "modeldb112923.projection.009",
        "modeldb112923.projection.012",
    ]
    assert not set(relay_ids) & set(trn_ids)
    assert profile["fixed_prime"] == {
        "source_index": 40,
        "arrival_time_ms_from_stimulus_onset": 0.0,
        "amplitude": 1.0,
    }
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "short_match_runs": 2,
        "short_mismatch_runs": 2,
    }
    assert "Do not tune" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_receptor_prime_decomposition_localizes_dominant_off_surround() -> None:
    result_path = ROOT / "docs/validation-results/figure7-receptor-prime-decomposition-452.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-receptor-prime-decomposition-assessment-453.yaml"
        ).read_text()
    )
    outcomes = {item["arm"]: item for item in result["outcomes"]}
    relay = outcomes["relay_on_center_only"]
    off_surround = outcomes["trn_off_surround_only"]

    assert result["passing_arms"] == []
    assert sorted(set(relay["match"]["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert sorted(set(relay["mismatch"]["relay_spike_indices"])) == [
        22,
        31,
        40,
        49,
        58,
    ]
    assert len(relay["match"]["trn_spike_indices"]) == 194
    assert len(relay["mismatch"]["trn_spike_indices"]) == 194
    assert off_surround["match"]["relay_spike_indices"] == []
    assert off_surround["mismatch"]["relay_spike_indices"] == []
    assert len(off_surround["match"]["trn_spike_indices"]) == 200
    assert len(off_surround["mismatch"]["trn_spike_indices"]) == 200
    assert relay["gates"]["exact_arm_primed"]
    assert off_surround["gates"]["exact_arm_primed"]
    assert assessment["result_sha256"] == (
        "b4d5c00f05fa7dbe6d50f7273bead385468267ceb7a1de13d61d5dce969e7b7e"
    )
    assert assessment["assessment"]["causal_imbalance_localized"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_receptor_prime_max_headroom_is_one_fixed_endpoint_cross() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-receptor-prime-max-headroom-registration-454.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["learned_state"] == {
        "source_index": 40,
        "headroom_fraction": 1.0,
        "expected_common_weight_factor": pytest.approx(3.6531686628985414),
    }
    assert profile["receptor_prime"]["projection_ids"] == [
        "modeldb112923.projection.003",
        "modeldb112923.projection.005",
        "modeldb112923.projection.006",
        "modeldb112923.projection.007",
        "modeldb112923.projection.009",
        "modeldb112923.projection.012",
    ]
    assert registration["candidate"] == {
        "count": 1,
        "headroom_fraction": 1.0,
        "receptor_prime": "combined_on_center_and_off_surround",
        "selection": "sole previously fixed endpoint cross",
    }
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "short_match_runs": 1,
        "short_mismatch_runs": 1,
    }
    assert "Do not interpolate" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_receptor_prime_max_headroom_has_partial_wrong_geometry() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-receptor-prime-max-headroom-endpoint-455.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-receptor-prime-max-headroom-assessment-456.yaml"
        ).read_text()
    )

    assert result["applied_common_weight_factor"] == pytest.approx(3.6531686628985414)
    assert result["match"]["relay_spike_indices"] == [38, 42]
    assert result["match"]["relay_spike_times_ms"] == pytest.approx([11.91, 11.91])
    assert result["mismatch"]["relay_spike_indices"] == []
    assert len(result["match"]["trn_spike_indices"]) == 189
    assert len(result["mismatch"]["trn_spike_indices"]) == 200
    assert result["gates"]["match_more_active_relay_cells"]
    assert not result["gates"]["match_horizontal_relay_set"]
    assert not result["gates"]["mismatch_overlap_only"]
    assert not result["gates"]["match_more_trn_events"]
    assert result["gates"]["all_six_expectation_records_primed"]
    assert not result["mechanism_screen_pass"]
    assert assessment["result_sha256"] == (
        "1fda1cf3aad48a71cf561d5975d081a8361d3f58540f4befcf9b12811125e649"
    )
    assert assessment["assessment"]["maximal_interaction_endpoint_closed"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_max_headroom_prime_trace_is_readout_only_with_fixed_prefix() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-receptor-prime-max-headroom-trace-registration-457.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["base_profile"] == (
        "configs/calibration/figure7_receptor_prime_max_headroom_endpoint_v1.yaml"
    )
    assert profile["protocol"] == {
        "duration_ms": 55.0,
        "dt_ms": 0.01,
        "prefix_identity_window_ms": 25.0,
    }
    assert profile["readout"]["fixed_sample_times_ms"] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        11.0,
        11.9,
        12.0,
        15.0,
    ]
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "diagnostic_match_runs": 1,
        "diagnostic_mismatch_runs": 1,
    }
    assert "adds observations only" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_max_headroom_prime_trace_localizes_late_match_without_promotion() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-receptor-prime-max-headroom-trace-458.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-receptor-prime-max-headroom-trace-assessment-459.yaml"
        ).read_text()
    )

    assert result["prefix_identity_verified"]
    assert sorted(set(result["match"]["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert result["mismatch"]["relay_spike_indices"] == []
    assert len(result["match"]["trn_spike_indices"]) == 377
    assert len(result["mismatch"]["trn_spike_indices"]) == 443
    assert assessment["result_sha256"] == (
        "ceed70520cf4852229fd4f3c9970f26a2f24d41eccaab65daf2966e1bb9779d0"
    )
    identity = assessment["fixed_time_localization"]["overlap_identity_through_15_ms"]
    assert identity["compared_current_samples"] == 117
    assert identity["compared_voltage_samples"] == 39
    assert identity["maximum_absolute_current_difference_pA"] == 0.0
    assert identity["maximum_absolute_voltage_difference_mV"] == 0.0
    balance = assessment["fixed_time_localization"]["overlap_balance_at_4_ms"]
    assert balance["direct_plus_top_down_minus_trn_pA"] > 0
    assert balance["four_path_net_pA"] < 0
    assert not assessment["assessment"]["original_smart_reproduced"]
    assert not assessment["assessment"]["baseline_promoted"]
    assert not assessment["assessment"]["parameter_selected"]


def test_relay_interneuron_mixed_gate_audit_rejects_invented_combination() -> None:
    audit = yaml.safe_load(
        (
            ROOT / "docs/validation-results/relay-interneuron-mixed-gate-resolution-460.yaml"
        ).read_text()
    )

    assert audit["archived_gate"]["gate"]["dependency"] == "input"
    assert audit["archived_gate"]["gate"]["sensitivities"]["green"] == pytest.approx(0.37)
    assert audit["archived_gate"]["direct_method"] == "connectFromOne"
    assert audit["archived_gate"]["nested_projection"]["source"] == "Layer_4"
    assert audit["assessment"]["current_declared_external_input_retained"]
    assert not audit["assessment"]["combined_interpretation_authorized"]
    assert not audit["assessment"]["exact_legacy_precedence_recovered"]
    assert not audit["assessment"]["original_smart_reproduced"]
    assert not audit["assessment"]["baseline_promoted"]


def test_subunit_interneuron_gain_screen_is_finite_and_post_source() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-receptor-prime-max-headroom-interneuron-subunit-gain-registration-461.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["pathway"]["projection_id"] == "modeldb112923.projection.002"
    assert profile["pathway"]["source_weight_control"] == 1.0
    assert profile["pathway"]["gains_in_execution_order"] == [0.9, 0.8, 0.7, 0.6]
    assert profile["learned_state"]["expected_common_weight_factor"] == pytest.approx(
        3.6531686628985414
    )
    assert len(profile["receptor_prime"]["projection_ids"]) == 6
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "short_match_runs": 4,
        "short_mismatch_runs": 4,
    }
    assert "post-source behavior calibration" in registration["interpretation_boundary"]
    assert "without interpolation or extension" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_subunit_interneuron_gain_screen_has_sole_0p8_survivor() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure7-receptor-prime-max-headroom-interneuron-subunit-gain-462.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-receptor-prime-max-headroom-interneuron-subunit-gain-assessment-463.yaml"
        ).read_text()
    )

    assert result["surviving_gains"] == [0.8]
    assert result["selected_gain"] == pytest.approx(0.8)
    survivor = next(item for item in result["outcomes"] if item["projection_002_gain"] == 0.8)
    assert survivor["pass"]
    assert set(survivor["match"]["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert survivor["mismatch"]["relay_spike_indices"] == [40]
    assert len(survivor["match"]["trn_spike_indices"]) == 183
    assert len(survivor["mismatch"]["trn_spike_indices"]) == 165
    assert assessment["result_sha256"] == (
        "407a61207cc2207dfad6f7d54f3b2c7fe35eb8759d0a423397ae3a990631f5be"
    )
    assert not assessment["assessment"]["original_smart_reproduced"]
    assert not assessment["assessment"]["baseline_promoted"]
    assert not assessment["assessment"]["source_parameter_recovered"]


def test_persistent_gain0p8_figure6_prerequisite_is_fixed() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-declared-interneuron-input-gain0p8-registration-464.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["trn_to_relay_gaba"]["scales"] == {
        "modeldb112923.projection.000": 0.01,
        "modeldb112923.projection.001": 0.01,
        "modeldb112923.projection.002": 0.8,
        "modeldb112923.projection.004": 0.03,
    }
    assert registration["change"] == {
        "projection_id": "modeldb112923.projection.002",
        "effective_gain": 0.8,
        "lifecycle": "persistent_during_learning_and_recognition",
    }
    assert registration["execution"]["parameter_search"] is False
    assert "does not authorize a nearby value" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_persistent_gain0p8_passes_figure6_and_locks_complete_pair() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure6-declared-interneuron-input-gain0p8-465.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-declared-interneuron-input-gain0p8-assessment-466.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gain0p8-complete-pair-registration-467.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert result["pass"]
    assert all(result["gates"].values())
    assert result["relay_event_counts_by_index"] == {
        "38": 4,
        "39": 4,
        "40": 4,
        "41": 4,
        "42": 4,
    }
    assert assessment["result_sha256"] == (
        "45540020e4a125a7b00e846640221cabf8f2b2ccd424ae2f85867fbf335e34a6"
    )
    assert assessment["result_summary"]["maximal_archived_bound_common_factor"] == pytest.approx(
        3.6651418062578482
    )
    assert profile["protocol"]["duration_ms"] == 100.0
    assert profile["official_rate_gates"] == {
        "match_nonspecific_events": 4,
        "match_rate_hz": 40.0,
        "mismatch_nonspecific_events": 7,
        "mismatch_rate_hz": 70.0,
    }
    assert registration["execution"]["parameter_search"] is False
    assert registration["baseline_promoted"] is False


def test_persistent_gain0p8_complete_pair_is_closed_without_promotion() -> None:
    result = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-persistent-gain0p8-complete-pair-468.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gain0p8-complete-pair-assessment-469.yaml"
        ).read_text()
    )

    assert sorted(set(result["match"]["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert len(result["match"]["relay_spike_indices"]) == 20
    assert sorted(set(result["mismatch"]["relay_spike_indices"])) == [22, 31, 40, 49, 58]
    assert result["mismatch"]["relay_spike_indices"][0] == 40
    assert result["mismatch"]["relay_spike_times_ms"][0] == pytest.approx(11.75)
    assert result["mismatch"]["relay_spike_times_ms"][1] == pytest.approx(49.31)
    assert len(result["match"]["trn_spike_indices"]) == 581
    assert len(result["mismatch"]["trn_spike_indices"]) == 625
    assert len(result["match"]["nonspecific_spike_times_ms"]) == 5
    assert len(result["mismatch"]["nonspecific_spike_times_ms"]) == 4
    assert assessment["result_sha256"] == (
        "d4480a74e480035d769829d5f99ea906152fa2690646f44ab0b8aa193e3bc7b5"
    )
    assert not assessment["assessment"]["preregistered_contract_passed"]
    assert assessment["assessment"]["candidate_closed_without_interpolation"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not assessment["assessment"]["original_smart_reproduced"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_figure7_temporal_gate_audit_requires_mechanism_not_posthoc_time() -> None:
    audit = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-specific-thalamus-temporal-gate-audit-470.yaml"
        ).read_text()
    )

    assert audit["assessment"]["source_requires_initial_overlap_selection"]
    assert audit["assessment"]["source_allows_later_t_type_burst_output"]
    assert not audit["assessment"]["exact_temporal_cutoff_identifiable"]
    assert not audit["assessment"]["artifact_468_reclassified_as_pass"]
    assert "Do not use 49.31 ms" in audit["anti_posthoc_boundaries"][0]
    future = audit["correction"]["future_rule"]
    assert "hyperpolarization" in future["later_mismatch_output"]
    assert "T-type-calcium" in future["later_mismatch_output"]
    assert "exactly four match and seven" in future["fixed_numeric_output"]
    assert not audit["assessment"]["original_smart_reproduced"]
    assert not audit["assessment"]["baseline_promoted"]


def test_gain0p8_calcium_trace_is_readout_only_and_identity_locked() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gain0p8-mismatch-calcium-trace-registration-471.yaml"
        ).read_text()
    )
    profile = yaml.safe_load((ROOT / registration["profile"]).read_text())

    assert profile["condition"] == "mismatch"
    assert profile["trace"]["recorded_relay_indices"] == [22, 31, 38, 39, 40, 41, 42, 49, 58]
    assert {
        "i_ca_distal_dendrite",
        "i_ca_proximal_dendrite",
        "i_ca_soma",
        "m_ca_soma",
        "h_ca_soma",
    } <= set(profile["trace"]["required_variables"])
    assert registration["reference_pair_sha256"] == (
        "d4480a74e480035d769829d5f99ea906152fa2690646f44ab0b8aa193e3bc7b5"
    )
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "mismatch_runs": 1,
        "parameter_search": False,
        "intervention": "none",
        "added_observation": "lossless_relay_state_trace",
    }
    assert "exactly match Artifact 468" in registration["identity_requirement"]
    assert "cannot alter Artifact 468" in registration["interpretation_boundary"]
    assert registration["baseline_promoted"] is False


def test_gain0p8_calcium_trace_preserves_events_and_rejects_rebound() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gain0p8-mismatch-calcium-trace-472.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gain0p8-mismatch-calcium-trace-assessment-473.yaml"
        ).read_text()
    )

    assert result["event_train_identity_verified"]
    assert result["trace"]["sha256"] == (
        "6085eb6dbdeea1cf4238c8aef2d57303e0a43fa42014cad3cbf7e7047f1db7c3"
    )
    assert result["trace"]["sample_count"] == 10000
    assert result["trace"]["required_variables_present"]
    assert assessment["result_sha256"] == (
        "e1b06ab9253c7507dc617d270a54b6262d850d5fc8bbf34b64790eda3bce9a26"
    )
    tests = assessment["classification_tests"]
    assert tests["every_first_nonoverlap_escape_lacks_preceding_hyperpolarization"]
    assert tests["every_first_nonoverlap_escape_lacks_t_channel_availability_recovery"]
    assert tests["each_nonoverlap_interevent_interval_contains_one_positive_soma_peak"]
    assert not tests["late_nonoverlap_t_type_rebound_supported"]
    for cell in ("22", "31", "49", "58"):
        first = assessment["first_nonoverlap_events"][cell]
        assert not first["voltage_below_initial_before_first_event"]
        assert not first["h_ca_increased_before_first_event"]
        assert len(first["positive_soma_peaks_before_first_event"]) == 1
        for compartment in ("soma", "proximal_dendrite", "distal_dendrite"):
            values = first["compartments"][compartment]
            assert values["initial_voltage_mV"] == -60.0
            assert values["minimum_voltage_before_first_event"] == {
                "value": -60.0,
                "time_ms": 0.0,
            }
            assert values["h_ca_immediately_before_first_event"] < values["initial_h_ca"]
    assert assessment["assessment"]["artifact_468_failure_unchanged"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not assessment["assessment"]["original_smart_reproduced"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_legacy_thalamus_benchmark_recovers_falling_minus20_detector() -> None:
    audit = yaml.safe_load(
        (ROOT / "docs/validation-results/legacy-thalamus-axon-detector-audit-476.yaml").read_text()
    )

    assert audit["archive"]["archive_sha256"] == (
        "6c3047d281f4fe432c5144748171a05b2e2ef8bcc4cdd6361c3d7612962f352a"
    )
    assert audit["serialized_and_observed"]["trace_step_ms"] == 0.05
    assert audit["serialized_and_observed"]["axonal_delay_ms"] == 2.0
    assert audit["serialized_and_observed"]["tonic"]["axon_event_count"] == 199
    assert audit["serialized_and_observed"]["burst"]["axon_event_count"] == 6
    assert audit["detector_alignment"]["event_count_checked"] == 205
    assert (
        audit["detector_alignment"]["falling_minus20_mV"][
            "maximum_absolute_error_from_serialized_delay_ms"
        ]
        == 0.05
    )
    assert audit["assessment"]["falling_minus20_detector_implementation_authorized"]
    assert not audit["assessment"]["source_parameter_fitted_from_figure7"]
    assert not audit["assessment"]["original_smart_reproduced"]
    assert not audit["assessment"]["baseline_promoted"]


def test_legacy_falling_minus20_figure6_is_preregistered_and_source_coherent() -> None:
    profile = yaml.safe_load(
        (ROOT / "configs/calibration/figure6_legacy_falling_minus20_source_v1.yaml").read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-falling-minus20-source-registration-477.yaml"
        ).read_text()
    )

    overrides = profile["runtime_overrides"]
    assert overrides["spike_event_coordinate"] == "absolute_physical"
    assert overrides["spike_event_threshold_mV"] == -20.0
    assert overrides["spike_event_rule"] == "falling_threshold_crossing"
    assert overrides["trn_spike_event_threshold_mV"] is None
    assert overrides["nonspecific_spike_event_rule"] is None
    assert profile["projection_weight_scales"] == {}
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "figure7_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Run one complete, fully monitored Figure 6")
    assert "figure7_match" in registration["locked_holdouts"]


def test_legacy_detector_calibrated_transfer_cross_is_fixed_before_execution() -> None:
    profile = yaml.safe_load(
        (
            ROOT / "configs/calibration/figure6_legacy_falling_minus20_calibrated_transfer_v1.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-falling-minus20-calibrated-transfer-registration-480.yaml"
        ).read_text()
    )

    assert profile["runtime_overrides"]["spike_event_rule"] == ("falling_threshold_crossing")
    assert profile["runtime_overrides"]["spike_event_threshold_mV"] == -20.0
    assert profile["projection_weight_scales"] == {
        "modeldb112923.projection.000": 0.01,
        "modeldb112923.projection.001": 0.01,
        "modeldb112923.projection.004": 0.03,
    }
    assert registration["execution"]["parameter_search"] is False
    assert registration["execution"]["figure7_runs"] == 0
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert registration["interpretation_boundary"].startswith("A pass would show compatibility")


def test_unscaled_legacy_detector_cross_fails_before_recognition() -> None:
    result = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure6-legacy-falling-minus20-source-478.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-falling-minus20-source-assessment-479.yaml"
        ).read_text()
    )

    assert result["runtime_fingerprint"] == (
        "4ccdcc18e9bc1dc61192c9ccce42c6ccbcdf13bd74960edd7ff8a7e01e95069f"
    )
    assert result["projection_weight_scales"] == {}
    assert not result["holdouts_consulted"]
    assert result["relay_recruitment"]["active_indices"] == [38, 39, 40, 41, 42]
    assert result["population_spikes"]["thalamic_relay"] == 5
    assert result["population_spikes"]["trn"] == 411
    assert result["top_down_timing"]["following_relay_spike_ms"] is None
    assert assessment["result_sha256"] == (
        "5166736467e091e1da87be537f47a1833facb120ddec63ebabc62dd194b7bb83"
    )
    assert not assessment["assessment"]["preregistered_figure6_contract_passed"]
    assert assessment["assessment"]["exact_candidate_closed"]
    assert not assessment["assessment"]["advance_to_figure7"]
    assert not assessment["assessment"]["original_smart_reproduced"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_legacy_detector_figure7_match_is_locked_to_fresh_weights() -> None:
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-falling-minus20-calibrated-transfer-assessment-482.yaml"
        ).read_text()
    )
    profile = yaml.safe_load(
        (ROOT / "configs/calibration/figure7_legacy_falling_minus20_match_v1.yaml").read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-falling-minus20-match-registration-483.yaml"
        ).read_text()
    )
    runner = (ROOT / "scripts/run_figure7_legacy_detector_match.py").read_text()

    assert assessment["assessment"]["preregistered_figure6_contract_passed"]
    assert assessment["result_sha256"] == registration["training_result_sha256"]
    assert profile["weight_handoff"] == "actual_fresh_figure6_weights_no_expansion"
    assert profile["protocol"]["condition"] == "match"
    assert profile["protocol"]["top_down_cue_lead_ms"] == 0.0
    assert not profile["protocol"]["prime_top_down_receptors_at_stimulus"]
    assert profile["protocol"]["comparator"] == "none"
    assert registration["execution"] == {
        "figure6_repeat_runs": 1,
        "match_runs": 1,
        "mismatch_runs": 0,
        "parameter_search": False,
    }
    assert "learned_weights=training.learned_weights" in runner
    assert "persistent_projection_weight_scales=scales" in runner
    assert "condition=MatchCondition.MATCH" in runner
    assert "MatchCondition.MISMATCH" not in runner


def test_legacy_detector_match_failure_keeps_mismatch_locked() -> None:
    result = yaml.safe_load(
        (ROOT / "docs/validation-results/figure7-legacy-falling-minus20-match-484.yaml").read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-falling-minus20-match-assessment-485.yaml"
        ).read_text()
    )

    match = result["match_result"]
    assert result["training_repeat_verified"]
    assert sorted(set(match["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert {match["relay_spike_indices"].count(index) for index in range(38, 43)} == {4}
    assert len(match["trn_spike_times_ms"]) == 550
    assert len(match["nonspecific_spike_times_ms"]) == 33
    assert match["top_down_current_termination_time_ms"] == 5.95
    assert not result["match_prerequisites_pass"]
    assert not result["mismatch_run_authorized"]
    assert assessment["result_sha256"] == (
        "de1eff19700668fc70e8caf02fca825b8f9977a1e7f0052e036fae79f78278c8"
    )
    assert assessment["assessment"]["failure_localized_to_nonspecific_output"]
    assert not assessment["assessment"]["mismatch_run_authorized"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_modeldb_nonspecific_recovered_detector_cross_starts_at_figure6() -> None:
    profile = yaml.safe_load(
        (
            ROOT / "configs/calibration/figure6_legacy_detector_modeldb_nonspecific_v1.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-registration-486.yaml"
        ).read_text()
    )

    overrides = profile["runtime_overrides"]
    assert overrides["nonspecific_intrinsic_cell_convention"] == "modeldb_112923"
    assert overrides["nonspecific_axial_convention"] == "kinness_serialized_edge"
    assert overrides["nonspecific_calcium_kinetics_convention"] == "modeldb_112923"
    assert overrides["spike_event_rule"] == "falling_threshold_crossing"
    assert overrides["nonspecific_spike_event_rule"] is None
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "figure7_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")
    assert "figure7_match" in registration["locked_holdouts"]


def test_modeldb_nonspecific_detector_cross_passes_figure6_and_locks_match() -> None:
    result = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-487.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-assessment-488.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-match-registration-489.yaml"
        ).read_text()
    )

    assert result["runtime_fingerprint"] == (
        "b48bb584d1799767f18a9633399de37c944d60a9031fcc817a37ef40054e7610"
    )
    assert result["relay_recruitment"]["confined_to_horizontal_bar_at_40_hz"]
    assert result["population_spikes"]["thalamic_relay"] == 20
    assert result["population_spikes"]["thalamic_nonspecific"] == 2
    assert result["recruitment"]["feedforward_chain_complete"]
    assert result["top_down_timing"]["causal_pair_in_learning_window"]
    assert assessment["result_sha256"] == (
        "238cd0dbbba4d3a5714738c65e9545f2c71bc7d9e942d85593b0b6ac57cf6a13"
    )
    assert assessment["assessment"]["preregistered_figure6_contract_passed"]
    assert assessment["assessment"]["recognition_match_authorized"]
    assert registration["execution"] == {
        "figure6_repeat_runs": 1,
        "match_runs": 1,
        "mismatch_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Repeat Artifact 487 exactly")


def test_modeldb_nonspecific_match_fails_at_two_events_without_interpolation() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-match-490.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-match-assessment-491.yaml"
        ).read_text()
    )

    match = result["match_result"]
    assert result["training_repeat_verified"]
    assert sorted(set(match["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert {match["relay_spike_indices"].count(index) for index in range(38, 43)} == {4}
    assert len(match["trn_spike_times_ms"]) == 549
    assert match["nonspecific_spike_times_ms"] == [2.06, 4.9]
    assert not result["mismatch_run_authorized"]
    assert assessment["result_sha256"] == (
        "10a4ec871829c541f65c15b2c449d11cc1560cf44a2c2bb0579c9432fd41b2d5"
    )
    assert not assessment["assessment"]["interpolation_between_cell_sources_authorized"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_paper_axial_official_nonspecific_factorial_is_preregistered() -> None:
    profile = yaml.safe_load(
        (
            ROOT
            / "configs/calibration/figure6_legacy_detector_modeldb_nonspecific_paper_axial_v1.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-axial-registration-492.yaml"
        ).read_text()
    )

    overrides = profile["runtime_overrides"]
    assert overrides["nonspecific_intrinsic_cell_convention"] == "modeldb_112923"
    assert overrides["nonspecific_axial_convention"] == "paper_literal"
    assert overrides["spike_event_rule"] == "falling_threshold_crossing"
    assert profile["factorial_boundary"]["sole_change"] == (
        "nonspecific axial convention from kinness_serialized_edge to paper_literal"
    )
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "figure7_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")


def test_paper_axial_nonspecific_cross_passes_figure6_and_registers_match() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-axial-493.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-axial-assessment-494.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-axial-match-registration-495.yaml"
        ).read_text()
    )

    assert result["runtime_fingerprint"] == (
        "a09c953ac9f24ee709de3ededf80990c6dfb609ba6eece2c7e4c1c2c12a5a8fd"
    )
    assert result["relay_recruitment"]["confined_to_horizontal_bar_at_40_hz"]
    assert result["population_spikes"]["thalamic_relay"] == 20
    assert result["population_spikes"]["thalamic_nonspecific"] == 22
    assert result["recruitment"]["feedforward_chain_complete"]
    assert result["top_down_timing"]["causal_pair_in_learning_window"]
    assert assessment["result_sha256"] == (
        "6132cd1398ffdd7a9bbc8428b9afcb3e2a200f378011456d15d05390dbfda2ea"
    )
    assert assessment["assessment"]["preregistered_figure6_contract_passed"]
    assert not assessment["assessment"]["figure6_nonspecific_count_used_for_selection"]
    assert registration["execution"] == {
        "figure6_repeat_runs": 1,
        "match_runs": 1,
        "mismatch_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Repeat Artifact 493 exactly")


def test_paper_axial_match_fails_without_axial_interpolation() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-axial-match-496.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-axial-match-assessment-497.yaml"
        ).read_text()
    )

    match = result["match_result"]
    assert result["training_repeat_verified"]
    assert sorted(set(match["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert {match["relay_spike_indices"].count(index) for index in range(38, 43)} == {4}
    assert len(match["trn_spike_times_ms"]) == 550
    assert len(match["nonspecific_spike_times_ms"]) == 24
    assert not result["mismatch_run_authorized"]
    assert assessment["result_sha256"] == (
        "81dcea2a815cc1205cfc5e7a450a8e27512c9aaf2259de4f6618b5b51ace3e45"
    )
    assert assessment["assessment"]["both_official_axial_endpoints_closed"]
    assert not assessment["assessment"]["axial_interpolation_authorized"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_supplement_distal_gaba_factorial_is_preregistered_at_figure6() -> None:
    profile = yaml.safe_load(
        (
            ROOT
            / "configs/calibration/figure6_legacy_detector_modeldb_nonspecific_paper_axial_supplement_gaba_v1.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-axial-supplement-gaba-registration-498.yaml"
        ).read_text()
    )

    overrides = profile["runtime_overrides"]
    assert overrides["nonspecific_intrinsic_cell_convention"] == "modeldb_112923"
    assert overrides["nonspecific_axial_convention"] == "paper_literal"
    assert overrides["nonspecific_distal_gaba_source_convention"] == ("paper_supplement_1p5_1_7")
    assert overrides["spike_event_rule"] == "falling_threshold_crossing"
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "figure7_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")


def test_supplement_distal_gaba_passes_figure6_and_registers_match() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-axial-supplement-gaba-499.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-axial-supplement-gaba-assessment-500.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-axial-supplement-gaba-match-registration-501.yaml"
        ).read_text()
    )

    assert result["runtime_fingerprint"] == (
        "a4e88775f99579f2147fa298b44fc3f8b6d51d7eb06f5d222dc7d48f8db09496"
    )
    assert result["population_spikes"]["thalamic_relay"] == 20
    assert result["population_spikes"]["thalamic_nonspecific"] == 22
    assert assessment["result_sha256"] == (
        "92649496d6d7c38779bc341f9f21edf2e15d661af78009976f5bdaf7838dc93b"
    )
    assert all(assessment["figure6_control_identity"].values())
    assert assessment["assessment"]["preregistered_figure6_contract_passed"]
    assert registration["execution"] == {
        "figure6_repeat_runs": 1,
        "match_runs": 1,
        "mismatch_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Repeat Artifact 499 exactly")


def test_supplement_gaba_match_is_identical_and_closed() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-axial-supplement-gaba-match-502.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-axial-supplement-gaba-match-assessment-503.yaml"
        ).read_text()
    )

    match = result["match_result"]
    assert len(match["relay_spike_times_ms"]) == 20
    assert len(match["trn_spike_times_ms"]) == 550
    assert len(match["nonspecific_spike_times_ms"]) == 24
    assert assessment["result_sha256"] == (
        "42435916e7b9f39a7e965410d2c6881087ae660549b709f870dfb69a788a4cc5"
    )
    assert all(assessment["control_identity"].values())
    assert not assessment["assessment"]["official_distal_gaba_conflict_resolves_rate"]
    assert not assessment["assessment"]["gaba_interpolation_authorized"]
    assert not assessment["assessment"]["mismatch_run_authorized"]


def test_paper_nonspecific_calcium_kinetics_factorial_is_preregistered() -> None:
    profile = yaml.safe_load(
        (
            ROOT
            / "configs/calibration/figure6_legacy_detector_modeldb_nonspecific_paper_kinetics_v1.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-kinetics-registration-504.yaml"
        ).read_text()
    )

    overrides = profile["runtime_overrides"]
    assert overrides["nonspecific_intrinsic_cell_convention"] == "modeldb_112923"
    assert overrides["nonspecific_axial_convention"] == "paper_literal"
    assert overrides["nonspecific_calcium_kinetics_convention"] == "paper_2008"
    assert overrides["nonspecific_distal_gaba_source_convention"] == ("paper_supplement_1p5_1_7")
    assert profile["factorial_boundary"]["sole_change"] == (
        "nonspecific calcium kinetics from modeldb_112923 to paper_2008"
    )
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "figure7_runs": 0,
        "parameter_search": False,
    }


def test_paper_calcium_kinetics_passes_figure6_and_registers_match() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-kinetics-505.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-paper-kinetics-assessment-506.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-kinetics-match-registration-507.yaml"
        ).read_text()
    )

    assert result["runtime_fingerprint"] == (
        "2d0584e8b06185fb78ec9ef28c15716ef41d0b71f9caf3690a28e9fae9307c72"
    )
    assert result["population_spikes"]["thalamic_relay"] == 20
    assert result["population_spikes"]["thalamic_nonspecific"] == 16
    assert result["recruitment"]["feedforward_chain_complete"]
    assert result["top_down_timing"]["causal_pair_in_learning_window"]
    assert assessment["result_sha256"] == (
        "1e8c88a5d741d28a0047f5753ed13424b5f65d4417dad861bdca510931ca5912"
    )
    assert assessment["assessment"]["preregistered_figure6_contract_passed"]
    assert not assessment["assessment"]["calcium_parameter_fitted"]
    assert registration["execution"] == {
        "figure6_repeat_runs": 1,
        "match_runs": 1,
        "mismatch_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Repeat Artifact 505 exactly")


def test_paper_calcium_match_fails_without_kinetics_interpolation() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-kinetics-match-508.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-paper-kinetics-match-assessment-509.yaml"
        ).read_text()
    )

    match = result["match_result"]
    assert len(match["relay_spike_times_ms"]) == 20
    assert len(match["trn_spike_times_ms"]) == 549
    assert len(match["nonspecific_spike_times_ms"]) == 18
    assert assessment["result_sha256"] == (
        "18adfc7efacd3ce00668932e30ed4d242dc01552c532157240288e04215d0c59"
    )
    assert not assessment["assessment"]["calcium_kinetics_interpolation_authorized"]
    assert not assessment["assessment"]["mismatch_run_authorized"]


def test_kinness_axial_paper_kinetics_factorial_is_preregistered() -> None:
    profile = yaml.safe_load(
        (
            ROOT
            / "configs/calibration/figure6_legacy_detector_modeldb_nonspecific_kinness_axial_paper_kinetics_v1.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-kinness-axial-paper-kinetics-registration-510.yaml"
        ).read_text()
    )

    overrides = profile["runtime_overrides"]
    assert overrides["nonspecific_intrinsic_cell_convention"] == "modeldb_112923"
    assert overrides["nonspecific_axial_convention"] == "kinness_serialized_edge"
    assert overrides["nonspecific_calcium_kinetics_convention"] == "paper_2008"
    assert overrides["nonspecific_distal_gaba_source_convention"] == (
        "modeldb_serialized_1p461_1_4"
    )
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "figure7_runs": 0,
        "parameter_search": False,
    }


def test_kinness_axial_paper_kinetics_passes_figure6_and_registers_match() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-kinness-axial-paper-kinetics-511.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-modeldb-nonspecific-kinness-axial-paper-kinetics-assessment-512.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-kinness-axial-paper-kinetics-match-registration-513.yaml"
        ).read_text()
    )

    assert result["runtime_fingerprint"] == (
        "d6ecb849964c78cfbfe64a61db5aac10f32e9e0d99b1dfca26ef1b3991568933"
    )
    assert result["population_spikes"]["thalamic_relay"] == 20
    assert result["population_spikes"]["thalamic_nonspecific"] == 1
    assert result["recruitment"]["feedforward_chain_complete"]
    assert result["top_down_timing"]["causal_pair_in_learning_window"]
    assert assessment["result_sha256"] == (
        "24bdc580ae393a6693a2d23437b0a29ef61a4abbdd19e20e7a5d506be91a1eaf"
    )
    assert assessment["assessment"]["preregistered_figure6_contract_passed"]
    assert assessment["assessment"]["source_factorial_complete_after_match"]
    assert registration["execution"] == {
        "figure6_repeat_runs": 1,
        "match_runs": 1,
        "mismatch_runs": 0,
        "parameter_search": False,
    }


def test_final_executable_cell_factorial_closes_without_interpolation() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-kinness-axial-paper-kinetics-match-514.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-modeldb-nonspecific-kinness-axial-paper-kinetics-match-assessment-515.yaml"
        ).read_text()
    )

    match = result["match_result"]
    assert len(match["relay_spike_times_ms"]) == 20
    assert len(match["trn_spike_times_ms"]) == 549
    assert match["nonspecific_spike_times_ms"] == [2.72]
    assert assessment["result_sha256"] == (
        "0b5f5e1fe79f92b483e4dffd80d39c03d03aed54e35c668ab108193c7b7bc529"
    )
    assert [
        endpoint["nonspecific_events"]
        for endpoint in assessment["completed_executable_cell_factorial"].values()
    ] == [2, 24, 18, 1]
    assert assessment["assessment"]["executable_cell_axial_kinetics_factorial_closed"]
    assert not assessment["assessment"]["continuous_interpolation_authorized"]
    assert not assessment["assessment"]["mismatch_run_authorized"]


def test_paper_cell_paper_kinetics_cross_is_preregistered_at_figure6() -> None:
    profile = yaml.safe_load(
        (
            ROOT
            / "configs/calibration/figure6_legacy_detector_paper_nonspecific_paper_kinetics_v1.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-paper-nonspecific-paper-kinetics-registration-516.yaml"
        ).read_text()
    )

    overrides = profile["runtime_overrides"]
    assert overrides["nonspecific_intrinsic_cell_convention"] == "paper_table3"
    assert overrides["nonspecific_axial_convention"] == "paper_literal"
    assert overrides["nonspecific_calcium_kinetics_convention"] == "paper_2008"
    assert overrides["spike_event_rule"] == "falling_threshold_crossing"
    assert registration["execution"] == {
        "figure6_learning_runs": 1,
        "figure7_runs": 0,
        "parameter_search": False,
    }
    assert registration["stopping_rule"].startswith("Run one fresh complete Figure 6")


def test_paper_coherent_cell_passes_figure6_and_registers_match() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-paper-nonspecific-paper-kinetics-517.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-legacy-detector-paper-nonspecific-paper-kinetics-assessment-518.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-paper-nonspecific-paper-kinetics-match-registration-519.yaml"
        ).read_text()
    )

    assert result["runtime_fingerprint"] == (
        "37cf90faf61129d5619e90c02374201db01f48fd7dfdf50c2a19d6989db2aa24"
    )
    assert result["population_spikes"]["thalamic_relay"] == 20
    assert result["population_spikes"]["thalamic_nonspecific"] == 23
    assert result["recruitment"]["feedforward_chain_complete"]
    assert result["top_down_timing"]["causal_pair_in_learning_window"]
    assert assessment["result_sha256"] == (
        "5d6d66da5ca9b9bee1ab3bb64cf903233101243a61fa23f4a539bd0275370052"
    )
    assert assessment["assessment"]["preregistered_figure6_contract_passed"]
    assert not assessment["assessment"]["figure6_nonspecific_count_used_for_selection"]
    assert registration["execution"] == {
        "figure6_repeat_runs": 1,
        "match_runs": 1,
        "mismatch_runs": 0,
        "parameter_search": False,
    }


def test_paper_coherent_match_fails_and_requires_source_recovery() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-paper-nonspecific-paper-kinetics-match-520.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-legacy-detector-paper-nonspecific-paper-kinetics-match-assessment-521.yaml"
        ).read_text()
    )

    match = result["match_result"]
    assert len(match["relay_spike_times_ms"]) == 20
    assert sorted(set(match["relay_spike_indices"])) == [38, 39, 40, 41, 42]
    assert len(match["trn_spike_times_ms"]) == 550
    assert len(match["nonspecific_spike_times_ms"]) == 24
    assert assessment["result_sha256"] == (
        "855cc4144557a5ebb347ca1e1d63f23cc7bb5fbf68d98d36fff7599bd6acba39"
    )
    assert not assessment["assessment"]["paper_cellular_source_bundle_survives_match"]
    assert assessment["assessment"]["source_recovery_required_before_next_behavioral_run"]
    assert not assessment["assessment"]["parameter_interpolation_authorized"]
    assert not assessment["assessment"]["mismatch_run_authorized"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_smart_era_event_source_audit_forbids_population_threshold_fit() -> None:
    audit = yaml.safe_load(
        (ROOT / "docs/validation-results/kinness-smart-era-event-source-audit-522.yaml").read_text()
    )

    assert audit["smart_era_release"]["kinness_release_name"] == "KInNeSS 0.3.4 RC2"
    assert audit["smart_era_release"]["kinness_cvs_tag"] == "KINNESS_0_3_4_RC2"
    assert not audit["smart_era_release"]["archive_availability"]["kinness_rc2_snapshot_available"]
    assert audit["sanndra_history"]["spikeevents_file"] == "spikeevents.h"
    assert not audit["sanndra_history"]["source_body_available_in_preserved_doxygen"]
    assert audit["inference"]["universal_axon_conversion_supported"]
    assert not audit["inference"]["exact_crossing_algorithm_recovered_for_nonspecific"]
    assert not audit["assessment"]["population_specific_event_tuning_authorized"]
    assert not audit["assessment"]["original_smart_reproduced"]


def test_scheduler_audit_closes_extra_global_delay() -> None:
    result = yaml.safe_load(
        (ROOT / "docs/validation-results/kinness-brian-scheduler-audit-524.yaml").read_text()
    )
    assessment = yaml.safe_load(
        (ROOT / "docs/validation-results/kinness-brian-scheduler-assessment-525.yaml").read_text()
    )

    assert result["runtime"]["network_schedule"] == [
        "start",
        "groups",
        "thresholds",
        "synapses",
        "resets",
        "end",
    ]
    assert result["delivery_microprobe"]["gate_at_serialized_arrival"] == 0.0
    assert result["delivery_microprobe"]["first_positive_gate_time_ms"] == 0.11
    assert all(result["gates"].values())
    assert assessment["result_sha256"] == (
        "df78bb5523c90890b8d4771a4265caab0796657038628379db12d996e50810ff"
    )
    assert not assessment["assessment"]["add_global_one_step_delay_authorized"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_nonspecific_pathway_ablation_localizes_late_events_to_trn_gaba() -> None:
    result = yaml.safe_load(
        (ROOT / "docs/validation-results/figure7-nonspecific-pathway-ablation-527.yaml").read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-nonspecific-pathway-ablation-assessment-528.yaml"
        ).read_text()
    )

    outcomes = {item["label"]: item["summary"] for item in result["outcomes"]}
    without_gaba = outcomes["without_all_trn_to_nonspecific_gaba"]
    without_cortical = outcomes["without_all_layer6ii_to_nonspecific_excitation"]
    assert without_gaba["pre_upstream_nonspecific_event_count"] == 3
    assert without_gaba["post_upstream_nonspecific_event_count"] == 0
    assert without_cortical["pre_upstream_nonspecific_event_count"] == 3
    assert without_cortical["post_upstream_nonspecific_event_count"] == 21
    assert assessment["result_sha256"] == (
        "37c5fa55bccf5ea79bd680d2dec67bd00a3c9baac54fe86cf4029967027ca666"
    )
    assert assessment["assessment"]["all_late_events_require_trn_gaba_pathway"]
    assert not assessment["assessment"]["late_events_require_layer6ii_direct_excitation"]
    assert not assessment["assessment"]["t_type_calcium_necessity_proven"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_nonspecific_calcium_ablation_requires_isolated_replay() -> None:
    result = yaml.safe_load(
        (ROOT / "docs/validation-results/figure7-nonspecific-calcium-ablation-530.yaml").read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-nonspecific-calcium-ablation-assessment-531.yaml"
        ).read_text()
    )

    assert result["intervention"] == {
        "nonspecific_calcium_ablated_at_stimulus": True,
        "scope": "dendrites_only",
    }
    assert result["control"]["nonspecific_event_count"] == 24
    assert result["ablation"]["nonspecific_event_count"] == 1
    assert result["upstream_identity"]["relay_exact_sequence"]
    assert not result["upstream_identity"]["trn_exact_sequence"]
    assert assessment["result_sha256"] == (
        "8935f775ac94a9a619477784448cb2c4b647950bc8384019ef9044f6dd47d814"
    )
    assert assessment["assessment"]["nonspecific_dendritic_t_current_necessary_in_connected_model"]
    assert not assessment["assessment"]["cell_autonomous_rebound_proven"]
    assert assessment["assessment"]["isolated_replay_required"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_nonspecific_isolated_replay_proves_t_current_necessity() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-nonspecific-isolated-replay-registration-532.yaml"
        ).read_text()
    )
    result = yaml.safe_load(
        (ROOT / "docs/validation-results/figure7-nonspecific-isolated-replay-533.yaml").read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-nonspecific-isolated-replay-assessment-534.yaml"
        ).read_text()
    )

    assert registration["trace"]["incoming_projection_ids"] == [
        "modeldb112923.projection.047",
        "modeldb112923.projection.048",
        "modeldb112923.projection.049",
        "modeldb112923.projection.050",
        "modeldb112923.projection.051",
    ]
    assert not registration["trace"]["precomputed_synaptic_currents_replayed"]
    assert all(result["connected_source_repeat"].values())
    assert result["intact_replay"]["exact_spike_train"]
    assert result["intact_replay"]["maximum_error_by_state"] == 0.0
    assert result["source_late_event_count"] == 21
    assert result["ablated_late_event_count"] == 0
    assert result["cell_autonomous_t_current_support"]
    assert assessment["result_sha256"] == (
        "6d32095641e0661ca02906a3bad2f86fb584a50737a343b490939846f8e7c0c6"
    )
    assert not assessment["original_smart_reproduced"]
    assert not assessment["baseline_promoted"]


def test_effective_nonspecific_t_grid_selects_only_registered_match_survivor() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-nonspecific-effective-t-calibration-registration-535.yaml"
        ).read_text()
    )
    result = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-nonspecific-effective-t-calibration-536.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-nonspecific-effective-t-calibration-assessment-537.yaml"
        ).read_text()
    )

    assert len(registration["common_proximal_distal_scale_grid"]) == 17
    assert len(result["outcomes"]) == 17
    assert result["survivor_scales"] == [0.1875]
    assert result["selected"]["event_count"] == 4
    assert result["selected"]["effective_density_mS_cm2"] == 46.875
    assert assessment["result_sha256"] == (
        "c87144e2ba847da9cb31392b1d33e36c0e5f03566d63b93ef076b7644a69ec70"
    )
    assert not assessment["interpolation_or_extension_used"]
    assert not assessment["mismatch_authorized"]
    assert not assessment["original_smart_reproduced"]


def test_effective_nonspecific_t_connected_match_passes_before_mismatch() -> None:
    result = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-nonspecific-effective-t-connected-match-539.yaml"
        ).read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-nonspecific-effective-t-connected-match-assessment-540.yaml"
        ).read_text()
    )

    assert result["runtime_fingerprint"] == (
        "fa4ab9f0bf2bec4d6ad53cb6a91620047689b6839b146ed7d777d42350c2cdf5"
    )
    assert result["figure6"]["population_spikes"]["thalamic_relay"] == 20
    assert all(result["figure6"]["gates"].values())
    assert result["match"]["relay_active_indices"] == [38, 39, 40, 41, 42]
    assert result["match"]["relay_event_count"] == 20
    assert result["match"]["nonspecific_event_count"] == 4
    assert all(result["match"]["gates"].values())
    assert not result["mismatch_consulted"]
    assert assessment["result_sha256"] == (
        "bc53521a1728e207b5c1addf0982297016ced3a146d96d6442901180c8015bd4"
    )
    assert assessment["assessment"]["mismatch_authorized"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_effective_nonspecific_t_pair_fails_upstream_comparison() -> None:
    result = yaml.safe_load(
        (ROOT / "docs/validation-results/figure7-nonspecific-effective-t-pair-542.yaml").read_text()
    )
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-nonspecific-effective-t-pair-assessment-543.yaml"
        ).read_text()
    )

    assert all(result["figure6_gates"].values())
    assert result["match"]["relay_active_indices"] == [38, 39, 40, 41, 42]
    assert result["mismatch"]["relay_active_indices"] == [22, 31, 40, 49, 58]
    assert result["match"]["trn_event_count"] == 549
    assert result["mismatch"]["trn_event_count"] == 608
    assert result["match"]["nonspecific_event_count"] == 4
    assert result["mismatch"]["nonspecific_event_count"] == 4
    assert not result["gates"]["mismatch_relay_overlap_only"]
    assert not result["gates"]["match_more_trn_events"]
    assert not result["gates"]["mismatch_nonspecific_70_hz"]
    assert not result["reproduced"]
    assert assessment["result_sha256"] == (
        "ea796987c3eb0e30e34eca6ff1f8aca5f6e247dc998fa9434a4502129c01bdb5"
    )
    assert assessment["assessment"]["candidate_closed"]
    assert not assessment["assessment"]["mismatch_driven_retuning_authorized"]


def test_learned_coincidence_effective_t_cross_is_fixed_before_execution() -> None:
    profile_path = ROOT / "configs/calibration/figure7_learned_coincidence_effective_t_pair_v1.yaml"
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-learned-coincidence-effective-t-pair-registration-544.yaml"
        ).read_text()
    )
    profile = yaml.safe_load(profile_path.read_text())

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert registration["runtime_fingerprint"] == (
        "fa4ab9f0bf2bec4d6ad53cb6a91620047689b6839b146ed7d777d42350c2cdf5"
    )
    assert profile["runtime_overrides"] == {"nonspecific_dendritic_calcium_density_scale": 0.1875}
    assert profile["comparator"] == {
        "transform": "top_k_binary",
        "source_index": 40,
        "target_count": 5,
        "justification": profile["comparator"]["justification"],
    }
    assert profile["protocol"]["top_down_cue_lead_ms"] == 0.0
    assert profile["figure7_gates"]["match_nonspecific_events"] == 4
    assert profile["figure7_gates"]["mismatch_nonspecific_events"] == 7
    assert "not a recovered SMART source" in profile["source_audit"]["translation_boundary"]
    assert "not recovered original" in registration["classification_boundary"]


def test_learned_coincidence_effective_t_cross_closes_on_trn_order() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-learned-coincidence-effective-t-pair-545.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-learned-coincidence-effective-t-pair-assessment-546.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    assert result["match"]["relay_active_indices"] == [38, 39, 40, 41, 42]
    assert result["mismatch"]["relay_active_indices"] == [40]
    assert result["match"]["nonspecific_event_count"] == 4
    assert result["mismatch"]["nonspecific_event_count"] == 7
    assert result["match"]["trn_event_count"] == 549
    assert result["mismatch"]["trn_event_count"] == 584
    assert not result["gates"]["match_more_trn_events"]
    assert not result["behavioral_targets_pass"]
    assert assessment["assessment"]["all_gates_except_trn_order_pass"]
    assert assessment["assessment"]["candidate_closed"]
    assert not assessment["assessment"]["original_smart_reproduced"]


def test_trn_drive_decomposition_is_hash_pinned_and_read_only() -> None:
    registration = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-trn-drive-decomposition-registration-547.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["runtime_fingerprint"] == (
        "fa4ab9f0bf2bec4d6ad53cb6a91620047689b6839b146ed7d777d42350c2cdf5"
    )
    assert registration["fixed_candidate"]["comparator_target_count"] == 5
    assert registration["fixed_candidate"]["nonspecific_dendritic_calcium_density_scale"] == 0.1875
    assert "No weight" in registration["constraints"]
    assert "complete TRN event indices and times" in registration["added_readouts_only"]


def test_trn_drive_decomposition_localizes_peripheral_recurrent_inversion() -> None:
    result_path = ROOT / "docs/validation-results/figure7-trn-drive-decomposition-548.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-trn-drive-decomposition-assessment-549.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert result["repeat_identity"]["match_trn_events"] == 549
    assert result["repeat_identity"]["mismatch_trn_events"] == 584
    assert result["trn_event_counts"]["diagnostic_indices"]["match_total"] == 79
    assert result["trn_event_counts"]["diagnostic_indices"]["mismatch_total"] == 72
    assert result["trn_event_counts"]["remaining_72_indices"]["match_total"] == 470
    assert result["trn_event_counts"]["remaining_72_indices"]["mismatch_total"] == 512
    for projection in result["source_gate_integrals_over_100ms_for_diagnostic_indices"].values():
        assert projection["match_sum"] > projection["mismatch_sum"]
    assert assessment["assessment"]["global_event_order_inverted_inside_trn_network"]
    assert not assessment["assessment"]["candidate_reopened"]


def test_trn_recurrent_gaba_ablation_is_recognition_only_and_nonpromotable() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-trn-recurrent-gaba-ablation-registration-550.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["recognition_only_ablation"]["projection_ids"] == [
        "modeldb112923.projection.008",
        "modeldb112923.projection.011",
    ]
    assert "projection 013 within-TRN distal gap junctions" in registration["retained"]
    assert "cannot be promoted" in registration["boundary"]


def test_trn_recurrent_gaba_ablation_preserves_failed_inversion() -> None:
    result_path = ROOT / "docs/validation-results/figure7-trn-recurrent-gaba-ablation-551.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-trn-recurrent-gaba-ablation-assessment-552.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert result["figure6_gates"] == {
        "relay_active_indices": True,
        "relay_events_per_active_index": True,
        "relay_events": True,
        "top_down_horizontal_contrast": True,
    }
    assert result["match"]["trn_event_count"] == 2008
    assert result["mismatch"]["trn_event_count"] == 2028
    assert result["match"]["nonspecific_event_count"] == 22
    assert result["mismatch"]["nonspecific_event_count"] == 22
    assert not assessment["assessment"]["match_greater_than_mismatch_trn_order_restored"]
    assert assessment["assessment"]["recurrent_gaba_required_for_trn_stability"]
    assert not assessment["assessment"]["recurrent_gaba_is_sole_cause_of_global_inversion"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_trn_gap_junction_ablation_is_recognition_only_and_nonpromotable() -> None:
    registration = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-trn-gap-junction-ablation-registration-553.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["recognition_only_ablation"]["projection_ids"] == [
        "modeldb112923.projection.013"
    ]
    assert "projections 008 and 011 recurrent TRN chemical GABA" in registration["retained"]
    assert "cannot be promoted" in registration["boundary"]


def test_trn_gap_junction_ablation_worsens_failed_inversion() -> None:
    result_path = ROOT / "docs/validation-results/figure7-trn-gap-junction-ablation-554.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-trn-gap-junction-ablation-assessment-555.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    assert result["match"]["trn_event_count"] == 551
    assert result["mismatch"]["trn_event_count"] == 608
    assert result["match"]["trn_diagnostic_index_event_count"] == 79
    assert result["mismatch"]["trn_diagnostic_index_event_count"] == 80
    assert result["match"]["trn_remaining_index_event_count"] == 472
    assert result["mismatch"]["trn_remaining_index_event_count"] == 528
    assert result["match"]["nonspecific_event_count"] == 4
    assert result["mismatch"]["nonspecific_event_count"] == 8
    assert assessment["assessment"]["mismatch_excess_increased_relative_to_control"]
    assert assessment["assessment"]["electrical_coupling_counteracts_global_inversion"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_trn_recurrent_gaba_path_decomposition_is_fixed_and_nonpromotable() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-trn-recurrent-gaba-path-decomposition-registration-556.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    arms = registration["recognition_only_ablation_arms"]
    assert [arm["name"] for arm in arms] == [
        "soma_gaba_removed",
        "proximal_gaba_removed",
    ]
    assert [arm["projection_ids"] for arm in arms] == [
        ["modeldb112923.projection.008"],
        ["modeldb112923.projection.011"],
    ]
    assert registration["fixed_control_trn_events_match_mismatch"] == [549, 584]
    assert "Neither arm can be promoted" in registration["boundary"]


def test_trn_recurrent_gaba_path_decomposition_localizes_somatic_path() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-trn-recurrent-gaba-path-decomposition-557.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-trn-recurrent-gaba-path-decomposition-assessment-558.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    outcomes = {outcome["arm"]: outcome for outcome in result["outcomes"]}
    soma = outcomes["soma_gaba_removed"]
    proximal = outcomes["proximal_gaba_removed"]
    assert [soma[key]["trn_event_count"] for key in ("match", "mismatch")] == [
        1055,
        818,
    ]
    assert [proximal[key]["trn_event_count"] for key in ("match", "mismatch")] == [566, 571]
    assert [soma[key]["nonspecific_event_count"] for key in ("match", "mismatch")] == [5, 7]
    assert [proximal[key]["nonspecific_event_count"] for key in ("match", "mismatch")] == [22, 22]
    assert assessment["assessment"]["soma_path_removal_reverses_trn_order"]
    assert not assessment["assessment"]["proximal_path_removal_reverses_trn_order"]
    assert not assessment["assessment"]["either_arm_is_behavioral_candidate"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_trn_somatic_ring_audit_preserves_source_limit() -> None:
    audit = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-trn-somatic-ring-semantics-audit-559.yaml"
        ).read_text()
    )

    assert audit["source_record_008"]["ring"]
    assert audit["source_record_008"]["border_effect"] == "wrap"
    assert audit["current_executable_translation"]["targets_per_source"] == 80
    assert audit["current_executable_translation"]["total_connections"] == 6480
    assert not audit["current_executable_translation"]["self_connections"]
    assert not audit["assessment"]["new_source_justified_geometry_found"]
    assert not audit["assessment"]["current_center_excluded_geometry_officially_verified"]
    assert not audit["assessment"]["baseline_promoted"]


def test_trn_somatic_gaba_match_screen_is_bounded_and_mismatch_locked() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-trn-somatic-gaba-match-screen-registration-560.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["calibrated_projection_id"] == ("modeldb112923.projection.008")
    assert registration["effective_scale_grid"] == [
        0.75,
        0.875,
        0.9375,
        0.96875,
        1.0,
    ]
    assert registration["fixed_match_gates"] == {
        "relay_active_indices": [38, 39, 40, 41, 42],
        "relay_events": 20,
        "nonspecific_events": 4,
    }
    assert "mismatch outcomes at screened scales" in registration["locked_until_assessment"]
    assert "Do not interpolate" in registration["selection_rule"]


def test_trn_somatic_gaba_match_screen_selects_only_exact_gate_survivors() -> None:
    result_path = ROOT / "docs/validation-results/figure7-trn-somatic-gaba-match-screen-561.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-trn-somatic-gaba-match-screen-assessment-562.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    assert not result["mismatch_consulted"]
    survivors = [
        outcome["effective_scale"]
        for outcome in result["outcomes"]
        if all(outcome["match"]["match_gates"].values())
    ]
    assert survivors == [0.875, 1.0]
    assert assessment["assessment"]["match_survivor_scales"] == survivors
    assert not assessment["assessment"]["response_monotonic"]
    assert not assessment["assessment"]["interpolation_authorized"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_trn_somatic_gaba_mismatch_is_limited_to_match_survivors() -> None:
    registration = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-trn-somatic-gaba-mismatch-registration-563.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]
    match_path = ROOT / registration["match_result"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert (
        hashlib.sha256(match_path.read_bytes()).hexdigest() == registration["match_result_sha256"]
    )
    assert registration["survivor_scales"] == [0.875, 1.0]
    assert registration["fixed_match_trn_events_by_scale"] == {
        "0.875": 576,
        "1.0": 549,
    }
    assert registration["fixed_mismatch_gates"] == {
        "relay_active_indices": [40],
        "match_more_active_relay_cells": True,
        "match_more_trn_events": True,
        "nonspecific_events": 7,
    }
    assert "Do not interpolate" in registration["selection_rule"]


def test_trn_somatic_gaba_mismatch_closes_one_dimensional_family() -> None:
    result_path = ROOT / "docs/validation-results/figure7-trn-somatic-gaba-mismatch-564.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-trn-somatic-gaba-mismatch-assessment-565.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    outcomes = {item["effective_scale"]: item["mismatch"] for item in result["outcomes"]}
    assert outcomes[0.875]["relay_active_indices"] == [40]
    assert outcomes[1.0]["relay_active_indices"] == [40]
    assert outcomes[0.875]["nonspecific_event_count"] == 7
    assert outcomes[1.0]["nonspecific_event_count"] == 7
    assert outcomes[0.875]["trn_event_count"] == 595
    assert outcomes[1.0]["trn_event_count"] == 584
    assert not outcomes[0.875]["mismatch_gates"]["match_more_trn_events"]
    assert not outcomes[1.0]["mismatch_gates"]["match_more_trn_events"]
    assert assessment["assessment"]["projection008_one_dimensional_family_closed"]
    assert assessment["assessment"]["complete_survivor_scales"] == []
    assert not assessment["assessment"]["baseline_promoted"]


def test_persistent_gaba_t_cross_is_bounded_and_applies_before_learning() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gaba-t-match-cross-registration-566.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["persistent_projection_scale"] == {
        "projection_id": "modeldb112923.projection.008",
        "scale": 0.75,
        "effective_serialized_weight": 0.225,
    }
    assert registration["nonspecific_t_scale_grid"] == [0.125, 0.15625, 0.1875]
    assert "fresh Figure 6" in registration["persistence_rule"]
    assert "Do not rank by TRN count" in registration["selection_rule"]
    assert "make either parameter recognition-only" in registration["selection_rule"]


def test_persistent_gaba_t_cross_has_no_match_survivor() -> None:
    result_path = ROOT / "docs/validation-results/figure7-persistent-gaba-t-match-cross-567.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gaba-t-match-cross-assessment-568.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert not result["mismatch_consulted"]
    assert len(result["outcomes"]) == 3
    for outcome in result["outcomes"]:
        assert all(outcome["figure6_gates"].values())
        assert outcome["match"]["relay_event_count"] == 20
        assert outcome["match"]["trn_event_count"] == 621
        assert outcome["match"]["nonspecific_event_count"] == 3
        assert not all(outcome["match"]["match_gates"].values())
    assert assessment["assessment"]["survivor_t_scales"] == []
    assert not assessment["assessment"]["grid_extension_authorized"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_persistent_gaba0875_consistency_introduces_no_new_value() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gaba0875-consistency-registration-569.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["persistent_projection_scale"]["scale"] == 0.875
    assert registration["nonspecific_t_scale_grid"] == [0.1875]
    assert not registration["novel_parameter_values_introduced"]
    assert "one fixed consistency check" in registration["selection_rule"]
    assert "do not alter either value" in registration["selection_rule"]


def test_persistent_gaba0875_consistency_passes_learning_and_match() -> None:
    result_path = ROOT / "docs/validation-results/figure7-persistent-gaba0875-consistency-570.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gaba0875-consistency-assessment-571.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    outcome = result["outcomes"][0]
    assert all(outcome["figure6_gates"].values())
    assert all(outcome["match"]["match_gates"].values())
    assert outcome["match"]["relay_event_count"] == 20
    assert outcome["match"]["trn_event_count"] == 576
    assert outcome["match"]["nonspecific_event_count"] == 4
    assert not result["mismatch_consulted"]
    assert assessment["assessment"]["persistent_mismatch_authorized"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_persistent_gaba0875_mismatch_is_single_fixed_trial() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gaba0875-mismatch-registration-572.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]
    match_path = ROOT / registration["match_result"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert (
        hashlib.sha256(match_path.read_bytes()).hexdigest() == registration["match_result_sha256"]
    )
    assert registration["persistent_projection_scale"]["scale"] == 0.875
    assert registration["nonspecific_t_scale"] == 0.1875
    assert registration["fixed_match_trn_events"] == 576
    assert registration["fixed_mismatch_gates"] == {
        "relay_active_indices": [40],
        "match_more_active_relay_cells": True,
        "match_more_trn_events": True,
        "nonspecific_events": 7,
    }
    assert "one fresh Figure 6 and one mismatch" in registration["selection_rule"]


def test_persistent_gaba0875_mismatch_closes_on_trn_order_only() -> None:
    result_path = ROOT / "docs/validation-results/figure7-persistent-gaba0875-mismatch-573.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-gaba0875-mismatch-assessment-574.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    gates = result["mismatch"]["mismatch_gates"]
    assert gates == {
        "relay_active_indices": True,
        "match_more_active_relay_cells": True,
        "match_more_trn_events": False,
        "nonspecific_events": True,
    }
    assert result["mismatch"]["trn_event_count"] == 595
    assert result["mismatch"]["nonspecific_event_count"] == 7
    assert assessment["assessment"]["candidate_closed"]
    assert not assessment["assessment"]["repeat_or_adjustment_authorized"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_persistent_inhibitory_drive_audit_is_readout_only() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-inhibitory-drive-audit-registration-575.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["persistent_projection_scale"] == {
        "projection_id": "modeldb112923.projection.008",
        "scale": 0.875,
    }
    assert registration["nonspecific_t_scale"] == 0.1875
    assert len(registration["added_readouts_only"]) == 4
    assert "No model, protocol, parameter" in registration["boundary"]
    assert "cannot be reopened or promoted" in registration["boundary"]


def test_persistent_inhibitory_drive_audit_records_instrumentation_failure() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-persistent-inhibitory-drive-audit-576.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-inhibitory-drive-audit-assessment-577.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert result["match"]["trn_event_count"] == 576
    assert result["mismatch"]["trn_event_count"] == 595
    for condition in ("match", "mismatch"):
        assert result[condition]["nonspecific_trn_gaba_peak"] is None
        assert result[condition]["nonspecific_trn_gaba_integral_ms"] is None
        assert result[condition]["nonspecific_post_startup_trn_gaba_peak"] is None
        assert result[condition]["nonspecific_trn_current_range_pA"] is None
    assert not assessment["assessment"]["all_registered_added_readouts_present"]
    assert not assessment["assessment"]["effective_inhibition_order_identifiable"]
    assert not assessment["assessment"]["candidate_reopened"]


def test_recorded_inhibitory_drive_audit_changes_instrumentation_only() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-inhibitory-drive-recorded-registration-578.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["sole_execution_change"] == {
        "value": "record_relay_diagnostics=true",
        "purpose": "construct the existing nonspecific-pathway state monitor",
        "model_or_protocol_effect": "none",
    }
    assert registration["required_event_identity"]["match"] == {
        "relay_events": 20,
        "trn_events": 576,
        "nonspecific_events": 4,
    }
    assert registration["required_event_identity"]["mismatch"] == {
        "relay_events": 3,
        "trn_events": 595,
        "nonspecific_events": 7,
    }
    assert len(registration["required_finite_readouts"]) == 4
    assert "cannot reopen" in registration["boundary"]


def test_recorded_inhibitory_drive_audit_is_valid_and_confirms_order() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-persistent-inhibitory-drive-recorded-579.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-inhibitory-drive-recorded-assessment-580.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    assert (result["match"]["relay_event_count"], result["match"]["trn_event_count"]) == (
        20,
        576,
    )
    assert (
        result["mismatch"]["relay_event_count"],
        result["mismatch"]["trn_event_count"],
    ) == (3, 595)
    for condition in ("match", "mismatch"):
        values = [
            result[condition]["nonspecific_trn_gaba_peak"],
            result[condition]["nonspecific_trn_gaba_integral_ms"],
            result[condition]["nonspecific_post_startup_trn_gaba_peak"],
            *result[condition]["nonspecific_trn_current_range_pA"],
        ]
        assert all(isinstance(value, (int, float)) for value in values)
    assert (
        result["mismatch"]["nonspecific_trn_gaba_integral_ms"]
        > result["match"]["nonspecific_trn_gaba_integral_ms"]
    )
    assert assessment["assessment"]["effective_inhibition_order_identifiable"]
    assert assessment["assessment"]["raw_event_and_effective_drive_direction_agree"]
    assert assessment["assessment"]["candidate_closed"]
    assert not assessment["assessment"]["candidate_reopened"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_peripheral_trn_localization_is_full_sheet_and_read_only() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-peripheral-trn-localization-registration-581.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["diagnostic_scope"]["recorded_indices"] == "all 81 TRN cells"
    assert len(registration["diagnostic_scope"]["central_indices"]) == 9
    assert len(registration["added_readouts_only"]) == 6
    assert registration["required_event_identity"]["match"]["trn_events"] == 576
    assert registration["required_event_identity"]["mismatch"]["trn_events"] == 595
    assert "stronger than the direct evidence" in registration["evidence_correction"]
    assert "cannot" in registration["boundary"]


def test_peripheral_trn_full_output_failure_is_not_interpreted() -> None:
    result_path = ROOT / "docs/validation-results/figure7-peripheral-trn-localization-582.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-peripheral-trn-localization-assessment-583.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert result["execution"]["exit_code"] == 0
    assert result["execution"]["original_output_tokens"] == 30806
    assert not result["full_sheet_source_arrays_archived"]
    assert not result["regional_source_order_identifiable"]
    assert not assessment["assessment"]["all_registered_readouts_assessable"]
    assert not assessment["assessment"]["candidate_reopened"]


def test_peripheral_trn_summary_repeat_changes_serialization_only() -> None:
    registration = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-peripheral-trn-summary-registration-584.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert "Only stdout serialization changes" in registration["execution_identity"]
    assert len(registration["registered_summaries"]["regions"]["central"]) == 9
    assert registration["required_event_identity"]["match"]["trn_events"] == 576
    assert registration["required_event_identity"]["mismatch"]["trn_events"] == 595
    assert "cannot be promoted" in registration["boundary"]


def test_peripheral_trn_summary_confirms_recurrent_inversion() -> None:
    result_path = ROOT / "docs/validation-results/figure7-peripheral-trn-summary-585.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure7-peripheral-trn-summary-assessment-586.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    assert result["event_identity"] == {
        "match": {"relay_events": 20, "trn_events": 576, "nonspecific_events": 4},
        "mismatch": {"relay_events": 3, "trn_events": 595, "nonspecific_events": 7},
    }
    peripheral = result["region_summaries"]["peripheral"]
    assert peripheral["mismatch"]["trn_events"] > peripheral["match"]["trn_events"]
    for field in (
        "relay_ampa_integral_ms",
        "layer6ii_ampa_integral_ms",
        "layer6ii_nmda_integral_ms",
    ):
        assert peripheral["match"][field] > peripheral["mismatch"][field]
    for field in ("relay_ampa", "layer6ii_ampa", "layer6ii_nmda"):
        assert result["peripheral_per_cell_order_counts"][field]["mismatch_greater"] == 0
    assert assessment["assessment"]["recurrent_trn_inversion_confirmed"]
    assert assessment["assessment"]["broad_afferent_mismatch_excess_rejected"]
    assert assessment["assessment"]["candidate_closed"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_persistent_recurrent_gaba_match_cross_is_bounded_and_match_only() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-recurrent-gaba-match-cross-registration-587.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["persistent_somatic_projection_scale"] == {
        "projection_id": "modeldb112923.projection.008",
        "scale": 0.75,
        "effective_serialized_weight": 0.225,
    }
    assert registration["proximal_projection_id"] == "modeldb112923.projection.011"
    assert registration["persistent_proximal_projection_scale_grid"] == [
        0.875,
        0.9375,
        1.0,
    ]
    assert registration["nonspecific_t_scale"] == 0.1875
    assert "without consulting mismatch" in registration["selection_rule"]
    assert "Do not rank by TRN count" in registration["selection_rule"]
    assert "does not recover" in registration["boundary"]


def test_persistent_recurrent_gaba_match_cross_has_one_survivor() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-persistent-recurrent-gaba-match-cross-588.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-recurrent-gaba-match-cross-assessment-589.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert not result["mismatch_consulted"]
    assert len(result["outcomes"]) == 3
    outcomes = {outcome["proximal_projection_scale"]: outcome for outcome in result["outcomes"]}
    assert not all(outcomes[0.875]["figure6_gates"].values())
    assert all(outcomes[0.9375]["figure6_gates"].values())
    assert all(outcomes[0.9375]["match"]["match_gates"].values())
    assert outcomes[0.9375]["match"]["trn_event_count"] == 697
    assert outcomes[0.9375]["match"]["nonspecific_event_count"] == 4
    assert not all(outcomes[1.0]["match"]["match_gates"].values())
    assert assessment["assessment"]["exact_match_survivors"] == [0.9375]
    assert assessment["assessment"]["persistent_mismatch_authorized"]
    assert not assessment["assessment"]["grid_extension_authorized"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_persistent_recurrent_gaba_pair_is_single_fixed_verification() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-recurrent-gaba-pair-registration-590.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert registration["persistent_projection_scales"] == [
        {"projection_id": "modeldb112923.projection.008", "scale": 0.75},
        {"projection_id": "modeldb112923.projection.011", "scale": 0.9375},
    ]
    assert registration["fixed_match_control"]["trn_events"] == 697
    assert registration["fixed_mismatch_gates"] == {
        "relay_active_indices": [40],
        "match_more_active_relay_cells": True,
        "match_more_trn_events": True,
        "nonspecific_events": 7,
    }
    assert len(registration["registered_diagnostics"]) == 5
    assert "Do not adjust, repeat" in registration["selection_rule"]
    assert "not recover original" in registration["boundary"]


def test_persistent_recurrent_gaba_pair_closes_on_one_rate_gate() -> None:
    result_path = ROOT / "docs/validation-results/figure7-persistent-recurrent-gaba-pair-591.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-recurrent-gaba-pair-assessment-592.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    gates = result["fixed_figure7_gates"]
    assert sum(not passed for passed in gates.values()) == 1
    assert not gates["mismatch_nonspecific_events"]
    assert result["event_identity"]["match"] == {
        "relay_events": 20,
        "relay_active_indices": [38, 39, 40, 41, 42],
        "trn_events": 697,
        "nonspecific_events": 4,
    }
    assert result["event_identity"]["mismatch"] == {
        "relay_events": 3,
        "relay_active_indices": [40],
        "trn_events": 608,
        "nonspecific_events": 6,
    }
    assert (
        result["nonspecific_trn_gaba_integral_ms"]["match"]
        > result["nonspecific_trn_gaba_integral_ms"]["mismatch"]
    )
    assert assessment["assessment"]["failed_gate_count"] == 1
    assert assessment["assessment"]["candidate_closed"]
    assert not assessment["assessment"]["repeat_or_adjustment_authorized"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_mismatch_nonspecific_trace_compare_is_read_only_and_fixed() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-mismatch-nonspecific-trace-compare-registration-593.yaml"
        ).read_text()
    )
    profile_path = ROOT / registration["profile"]
    script_path = ROOT / registration["script"]

    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == registration["profile_sha256"]
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == registration["script_sha256"]
    assert len(registration["arms"]) == 2
    assert registration["arms"][0]["expected_mismatch"]["nonspecific_events"] == 7
    assert registration["arms"][1]["expected_mismatch"]["nonspecific_events"] == 6
    assert len(registration["added_readouts_only"]) == 6
    assert "cannot select a parameter" in registration["decision_rule"]
    assert "cannot reopen" in registration["boundary"]


def test_mismatch_nonspecific_trace_compare_localizes_missing_event() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure7-mismatch-nonspecific-trace-compare-594.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-mismatch-nonspecific-trace-compare-assessment-595.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert len(result["outcomes"]) == 2
    control, repaired = result["outcomes"]
    assert all(control["figure6_gates"].values())
    assert all(repaired["figure6_gates"].values())
    assert control["mismatch_identity"] == {
        "relay_events": 3,
        "relay_active_indices": [40],
        "trn_events": 595,
        "nonspecific_events": 7,
    }
    assert repaired["mismatch_identity"] == {
        "relay_events": 3,
        "relay_active_indices": [40],
        "trn_events": 608,
        "nonspecific_events": 6,
    }
    assert (
        repaired["nonspecific_trn_gaba"]["integral_ms"]
        > control["nonspecific_trn_gaba"]["integral_ms"]
    )
    assert len(control["nonspecific_positive_detector_local_maxima_ms_mV"]) == 7
    assert len(repaired["nonspecific_positive_detector_local_maxima_ms_mV"]) == 6
    assert repaired["fourth_late_detector_peak"] is None
    assert result["comparison"]["missing_event_localized_to_stronger_inhibitory_envelope"]
    assert result["comparison"]["intrinsic_t_recovery_rejected_as_primary_difference"]
    assert assessment["assessment"]["missing_event_localized_to_inhibitory_envelope"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not assessment["assessment"]["candidate_reopened"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_repaired_trn_output_gaba_interaction_is_narrow_and_preregistered() -> None:
    audit_path = ROOT / "docs/validation-results/figure7-trn-nonspecific-gaba-reopen-audit-596.yaml"
    audit = yaml.safe_load(audit_path.read_text())
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-output-gaba-match-registration-597.yaml"
        ).read_text()
    )

    assert (
        hashlib.sha256(audit_path.read_bytes()).hexdigest() == registration["authorization_sha256"]
    )
    assert (
        hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest()
        == registration["profile_sha256"]
    )
    assert (
        hashlib.sha256((ROOT / registration["script"]).read_bytes()).hexdigest()
        == registration["script_sha256"]
    )
    assert audit["admissibility"]["sole_reopened_scale"] == 0.75
    assert not audit["admissibility"]["new_interpolation_allowed"]
    assert not audit["admissibility"]["new_grid_extension_allowed"]
    assert registration["nonspecific_gaba_common_scale"] == 0.75
    assert registration["nonspecific_gaba_projection_ids"] == [
        "modeldb112923.projection.047",
        "modeldb112923.projection.048",
        "modeldb112923.projection.049",
    ]
    assert "consult mismatch" in registration["boundary"]
    assert "promote a baseline" in registration["boundary"]


def test_repaired_trn_output_gaba_interaction_closes_at_match() -> None:
    result_path = ROOT / "docs/validation-results/figure7-persistent-output-gaba-match-598.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-persistent-output-gaba-match-assessment-599.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    assert result["match"]["relay_active_indices"] == [38, 39, 40, 41, 42]
    assert result["match"]["relay_event_count"] == 20
    assert result["match"]["nonspecific_event_count"] == 5
    assert not result["match"]["gates"]["nonspecific_events"]
    assert not result["mismatch_consulted"]
    assert assessment["assessment"]["candidate_closed"]
    assert not assessment["assessment"]["interpolation_or_grid_extension_authorized"]
    assert not assessment["assessment"]["repeat_or_adjustment_authorized"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_calibrated_figure10_reset_pair_is_hash_pinned_and_causal() -> None:
    harness_evidence = yaml.safe_load(
        (ROOT / "docs/validation-results/figure10-calibrated-harness-600.yaml").read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure10-calibrated-reset-registration-601.yaml"
        ).read_text()
    )

    assert (
        hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest()
        == registration["profile_sha256"]
    )
    assert harness_evidence["implementation_sha256"] == registration["harness_sha256"]
    assert registration["script_sha256"] == (
        "bb1f306eb01b24178d3940beedca3c01a9bef30b362a0e60c6e6d287740d8b4d"
    )
    assert registration["negative_control"]["disabled_only_at_mismatch"] == [
        "modeldb112923.projection.017",
        "modeldb112923.projection.018",
    ]
    assert len(registration["fixed_gates"]) == 6
    assert "cannot erase" in registration["known_locked_discrepancy"]
    assert not registration["baseline_freeze_authorized"]


def test_calibrated_figure10_reset_localizes_failure_after_layer5() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure10-calibrated-reset-pair-602.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-calibrated-reset-assessment-603.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["intact"]["layer4_pre_active_indices"] == [38, 39, 40, 41, 42]
    assert result["intact"]["nonspecific_pre_events"] == 4
    assert result["intact"]["nonspecific_post_events"] == 7
    assert result["intact"]["layer5_post_events"] > result["disconnected_control"][
        "layer5_post_events"
    ]
    assert result["intact"]["layer6i_post_events"] == result[
        "disconnected_control"
    ]["layer6i_post_events"]
    assert result["intact"]["layer4_post_events"] == result["disconnected_control"][
        "layer4_post_events"
    ]
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": False,
        "alternative_release": False,
    }
    assert not result["reproduced_reset"]
    assert assessment["assessment"]["nonspecific_to_layer5_causal_effect_observed"]
    assert not assessment["assessment"]["layer5_to_layer6i_causal_effect_observed"]
    assert not assessment["assessment"]["official_figure10_reset_reproduced"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_figure10_layer6i_transfer_audit_is_hash_pinned_and_read_only() -> None:
    monitor = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-transfer-monitor-604.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-transfer-registration-605.yaml"
        ).read_text()
    )
    prior_path = ROOT / registration["prior_identity"]

    assert hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest() == registration[
        "profile_sha256"
    ]
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == registration[
        "prior_identity_sha256"
    ]
    assert monitor["harness_sha256"] == registration["harness_sha256"]
    assert monitor["runner_sha256"] == registration["script_sha256"]
    assert registration["record_layer6i_diagnostics"]
    assert registration["required_identity"]["intact_control_layer5_post"] == [70, 55]
    assert "cannot select a parameter" in registration["decision_rule"]
    assert "No projection-025 scaling" in registration["boundary"]


def test_figure10_layer6i_transfer_localizes_masking_after_input() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure10-layer6i-transfer-pair-606.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-transfer-assessment-607.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    intact = result["intact"]
    control = result["disconnected_control"]
    projection = "modeldb112923.projection.025"
    assert intact["layer6i_mismatch_gate_integral_ms_by_projection"][projection] > control[
        "layer6i_mismatch_gate_integral_ms_by_projection"
    ][projection]
    assert intact["layer6i_mismatch_current_integral_pA_ms_by_projection"][projection] > control[
        "layer6i_mismatch_current_integral_pA_ms_by_projection"
    ][projection]
    assert intact["layer6i_mismatch_current_peak_pA_by_projection"][projection] > control[
        "layer6i_mismatch_current_peak_pA_by_projection"
    ][projection]
    assert intact["layer6i_post_events"] == control["layer6i_post_events"] == 30
    assert intact["layer4_post_events"] == control["layer4_post_events"] == 43
    assert assessment["assessment"]["failure_localized_to_layer6i_masking_or_saturation"]
    assert assessment["assessment"]["layer5_transmitter_transfer_failure_rejected"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not assessment["assessment"]["baseline_promoted"]


def test_figure10_complete_reset_chain_audit_is_hash_pinned_and_read_only() -> None:
    monitor = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure10-reset-chain-monitor-608.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-reset-chain-registration-609.yaml"
        ).read_text()
    )
    prior_path = ROOT / registration["prior_identity"]

    assert hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest() == registration[
        "profile_sha256"
    ]
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == registration[
        "prior_identity_sha256"
    ]
    assert monitor["harness_sha256"] == registration["harness_sha256"]
    assert monitor["runner_sha256"] == registration["script_sha256"]
    assert registration["record_layer6i_diagnostics"]
    assert registration["record_reset_chain_diagnostics"]
    assert [item["projection_id"] for item in monitor["source_path"]] == [
        "modeldb112923.projection.017",
        "modeldb112923.projection.018",
        "modeldb112923.projection.025",
        "modeldb112923.projection.026",
        "modeldb112923.projection.036",
    ]
    assert "cannot select or alter a parameter" in registration["decision_rule"]
    assert "Neither Figure 7 nor Figure 10 is reopened" in registration["boundary"]


def test_figure10_complete_reset_chain_localizes_layer6i_spike_masking() -> None:
    result_path = ROOT / "docs/validation-results/figure10-reset-chain-pair-610.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-reset-chain-assessment-611.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    intact = result["intact"]
    control = result["disconnected_control"]
    projection025 = "modeldb112923.projection.025"
    assert intact["layer6i_mismatch_current_integral_pA_ms_by_projection"][projection025] > (
        2.5
        * control["layer6i_mismatch_current_integral_pA_ms_by_projection"][projection025]
    )
    assert intact["layer6i_mismatch_current_peak_pA_by_projection"][projection025] > (
        8.0 * control["layer6i_mismatch_current_peak_pA_by_projection"][projection025]
    )
    assert intact["layer6i_post_events"] == control["layer6i_post_events"] == 30
    assert [event[0] for event in intact["layer6i_mismatch_events"]] == [
        event[0] for event in control["layer6i_mismatch_events"]
    ]
    assert intact["layer4_inhibitory_post_events"] == (
        control["layer4_inhibitory_post_events"]
    ) == 124
    assert intact["layer4_inhibitory_post_active_indices"] == control[
        "layer4_inhibitory_post_active_indices"
    ]
    assert intact["layer4i_mismatch_projection026_current_peak_pA"] == control[
        "layer4i_mismatch_projection026_current_peak_pA"
    ]
    assert intact["layer4e_mismatch_projection036_current_trough_pA"] == control[
        "layer4e_mismatch_projection036_current_trough_pA"
    ]
    assert assessment["assessment"]["failure_localized_to_layer6i_spike_generation_masking"]
    assert assessment["assessment"]["downstream_synaptic_transfer_failure_rejected"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_figure10_layer23_layer6i_candidate_is_one_source_discrete_endpoint() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer23-layer6i-source-audit-612.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer23-layer6i-registration-613.yaml"
        ).read_text()
    )

    assert audit["sources"]["released_network"]["modifiable"] is False
    assert audit["sources"]["released_network"]["weight"] == 4.0
    assert audit["sources"]["released_network"]["asymptotic_weight"] == 2.0
    assert audit["sources"]["paper_supplement"]["weight_density_1e6_cm2"] == 1.0
    assert audit["resolution"]["rejected_interpretation"]["value"] == 2.0
    assert audit["resolution"]["selected_candidate"]["value"] == 1.0
    assert "only" in audit["stopping_rule"].lower()
    scales = {
        item["projection_id"]: item["scale"]
        for item in registration["persistent_projection_scales"]
    }
    assert scales["modeldb112923.projection.024"] == 0.25
    assert "only endpoint" in registration["boundary"]
    assert hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest() == (
        registration["profile_sha256"]
    )
    assert registration["harness_sha256"] == (
        "baa4c943652b752f04d1a451c3edb0ea068acf967f25b8f52f81b4a8f72900fb"
    )
    assert registration["script_sha256"] == (
        "74fe4b138a698e87976dcb0ed57faf2eea971d8cdd6dc9fb18586cffd09f598a"
    )
    assert hashlib.sha256((ROOT / registration["source_control"]).read_bytes()).hexdigest() == (
        registration["source_control_sha256"]
    )


def test_figure10_layer23_layer6i_endpoint_fails_and_closes_family() -> None:
    result_path = ROOT / "docs/validation-results/figure10-layer23-layer6i-pair-614.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer23-layer6i-assessment-615.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    assert result["persistent_projection_weight_scales"]["modeldb112923.projection.024"] == 0.25
    intact = result["intact"]
    control = result["disconnected_control"]
    projection025 = "modeldb112923.projection.025"
    assert intact["layer5_post_events"] > control["layer5_post_events"]
    assert intact["layer6i_mismatch_current_integral_pA_ms_by_projection"][projection025] > (
        10.0
        * control["layer6i_mismatch_current_integral_pA_ms_by_projection"][projection025]
    )
    assert intact["layer6i_post_events"] == control["layer6i_post_events"] == 5
    assert {event[0] for event in intact["layer6i_mismatch_events"]} == {40}
    assert [event[0] for event in intact["layer6i_mismatch_events"]] == [
        event[0] for event in control["layer6i_mismatch_events"]
    ]
    assert intact["layer4_inhibitory_post_events"] == (
        control["layer4_inhibitory_post_events"]
    ) == 124
    assert intact["layer4_post_events"] == control["layer4_post_events"] == 43
    assert intact["nonspecific_post_events"] == 9
    assert not result["reset_gates"]["winner_suppression"]
    assert not result["reset_gates"]["alternative_release"]
    assert assessment["assessment"]["family_closed"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_figure10_spatial_reset_audit_is_hash_pinned_and_read_only() -> None:
    monitor = yaml.safe_load(
        (ROOT / "docs/validation-results/figure10-spatial-reset-monitor-616.yaml").read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-spatial-reset-registration-617.yaml"
        ).read_text()
    )

    assert hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest() == (
        registration["profile_sha256"]
    )
    assert hashlib.sha256((ROOT / registration["prior_identity"]).read_bytes()).hexdigest() == (
        registration["prior_identity_sha256"]
    )
    assert monitor["harness_sha256"] == registration["harness_sha256"]
    assert monitor["runner_sha256"] == registration["script_sha256"]
    assert len(monitor["added_readouts"]) == 5
    assert "cannot select or alter a parameter" in registration["decision_rule"]
    assert "No model or score changes" in registration["boundary"]
    assert all(
        item["projection_id"] != "modeldb112923.projection.024"
        for item in registration["persistent_projection_scales"]
    )


def test_figure10_spatial_reset_localizes_layer5_wave_coverage_failure() -> None:
    result_path = ROOT / "docs/validation-results/figure10-spatial-reset-pair-618.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-spatial-reset-assessment-619.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment["result_sha256"]
    assert all(result["figure6_gates"].values())
    intact = result["intact"]
    control = result["disconnected_control"]
    assert intact["nonspecific_post_events"] == control["nonspecific_post_events"] == 7
    assert len(intact["layer5_post_active_indices"]) == 17
    assert len(control["layer5_post_active_indices"]) == 5
    added = set(intact["layer5_post_active_indices"]) - set(
        control["layer5_post_active_indices"]
    )
    assert len(added) == 12
    assert {intact["layer5_post_events_by_index"][index] for index in added} == {1}
    assert set(intact["projection025_nonzero_target_indices"]) == set(
        intact["layer5_post_active_indices"]
    )
    assert set(control["projection025_nonzero_target_indices"]) == set(
        control["layer5_post_active_indices"]
    )
    assert set(intact["layer6i_post_active_indices"]) == set(
        control["layer6i_post_active_indices"]
    ) == {38, 39, 40, 41, 42}
    assert max(
        intact["layer6i_soma_voltage_peak_mV_by_index"][index] for index in added
    ) < -60.0
    assert assessment["assessment"]["earliest_spatial_failure_is_layer5_wave_coverage"]
    assert assessment["assessment"]["secondary_failure_is_weak_peripheral_layer6i_recruitment"]
    assert assessment["assessment"]["event_detector_miss_rejected"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_figure10_duration_candidate_is_one_source_visible_endpoint() -> None:
    audit = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure10-duration-source-audit-620.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure10-duration-registration-621.yaml"
        ).read_text()
    )

    assert audit["paper_source"]["visible_time_axis_ms"] == [0, 300]
    assert audit["current_protocol"]["total_duration_ms"] == 200.0
    assert audit["selected_endpoint"]["total_duration_ms"] == 300.0
    assert "only" in audit["stopping_rule"].lower()
    assert registration["protocol"]["pre_match_duration_ms"] == 100.0
    assert registration["protocol"]["mismatch_duration_ms"] == 200.0
    assert registration["protocol"]["total_duration_ms"] == 300.0
    assert registration["record_layer6i_diagnostics"]
    assert not registration["record_reset_chain_diagnostics"]
    assert hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest() == (
        registration["profile_sha256"]
    )
    assert audit["instrumentation"]["runner_sha256"] == registration["script_sha256"]
    assert hashlib.sha256((ROOT / registration["prior_identity"]).read_bytes()).hexdigest() == (
        registration["prior_identity_sha256"]
    )
    assert "sole duration endpoint" in registration["boundary"]


def test_figure10_duration_restores_layer5_wave_but_not_reset() -> None:
    result_path = ROOT / "docs/validation-results/figure10-duration-pair-622.yaml"
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT / "docs/validation-results/figure10-duration-assessment-623.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    intact = result["intact"]
    control = result["disconnected_control"]
    assert intact["layer4_first_100ms_mismatch_events"] == (
        control["layer4_first_100ms_mismatch_events"]
    ) == 43
    assert intact["nonspecific_first_100ms_mismatch_events"] == (
        control["nonspecific_first_100ms_mismatch_events"]
    ) == 7
    assert [
        intact["layer5_first_100ms_mismatch_events"],
        control["layer5_first_100ms_mismatch_events"],
    ] == [70, 55]
    assert intact["layer5_late_active_indices"] == "all-0-through-80"
    assert len(control["layer5_late_active_indices"]) == 11
    assert intact["projection025_nonzero_target_indices"] == "all-0-through-80"
    projection025 = "modeldb112923.projection.025"
    assert intact["layer6i_mismatch_current_integral_pA_ms_by_projection"][
        projection025
    ] > 4.5 * control["layer6i_mismatch_current_integral_pA_ms_by_projection"][
        projection025
    ]
    assert intact["layer6i_late_mismatch_events"] == (
        control["layer6i_late_mismatch_events"]
    ) == 42
    assert intact["layer6i_late_active_indices"] == control[
        "layer6i_late_active_indices"
    ]
    assert intact["layer4_late_mismatch_events"] == (
        control["layer4_late_mismatch_events"]
    ) == 49
    assert assessment["assessment"]["duration_family_closed"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["reproduced_reset"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_projection025_spread_candidate_is_one_bounded_mixed_source_endpoint() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection025-spread-audit-624.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection025-spread-registration-625.yaml"
        ).read_text()
    )

    assert audit["executable_topology"]["standard_deviation"]["connections"] == 81
    assert audit["executable_topology"]["variance"]["connections"] == 729
    assert audit["executable_topology"]["variance"]["inputs_per_target"] == 9
    assert audit["selected_endpoint"] == {
        "global_gaussian_spread_convention": "standard_deviation",
        "projection025_gaussian_spread_convention": "variance",
        "endpoint_count": 1,
        "justification": audit["selected_endpoint"]["justification"],
    }
    assert "not recovery" in audit["claims_boundary"]
    assert hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest() == (
        registration["profile_sha256"]
    )
    assert hashlib.sha256((ROOT / registration["prior_identity"]).read_bytes()).hexdigest() == (
        registration["prior_identity_sha256"]
    )
    assert registration["runtime_overrides"] == {
        "projection025_gaussian_spread_convention": "variance"
    }
    assert registration["required_prerequisites"]["projection025_connections"] == 729
    assert "cannot alone promote" in registration["boundary"]


def test_projection025_variance_increases_current_but_not_reset() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection025-spread-pair-626.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection025-spread-assessment-627.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["projection025_topology"] == {
        "connection_count": 729,
        "inputs_per_target": 9,
    }
    intact = result["intact"]
    control = result["disconnected_control"]
    projection025 = "modeldb112923.projection.025"
    assert intact["projection025_nonzero_target_indices"] == "all-0-through-80"
    assert control["projection025_nonzero_target_count"] == 31
    assert intact["layer6i_mismatch_current_integral_pA_ms_by_projection"][
        projection025
    ] > 4.5 * control["layer6i_mismatch_current_integral_pA_ms_by_projection"][
        projection025
    ]
    assert intact["layer6i_post_events"] == control["layer6i_post_events"] == 72
    assert intact["layer6i_mismatch_active_indices"] == control[
        "layer6i_mismatch_active_indices"
    ]
    assert intact["layer4_post_events"] == control["layer4_post_events"] == 92
    assert result["reset_assessment"]["intact_winner_post_spikes"] == (
        result["reset_assessment"]["control_winner_post_spikes"]
    ) == 76
    assert result["reset_assessment"]["intact_released_alternatives"] == (
        result["reset_assessment"]["control_released_alternatives"]
    ) == 2
    assert assessment["assessment"]["family_closed"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["reproduced_reset"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer6i_axial_candidate_is_one_source_discrete_endpoint() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-axial-source-audit-628.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-axial-registration-629.yaml"
        ).read_text()
    )

    assert audit["layer6i_intrinsic_cross_check"]["source_agreement"]
    assert audit["projection025_target_cross_check"]["target_compartment"] == (
        "proximal_dendrite"
    )
    paper = audit["axial_equation_conflict"]["active_paper_literal"]
    kinness = audit["axial_equation_conflict"]["kinness_serialized_edge"]
    assert paper["conductance_into_soma_nS"] == pytest.approx(62.831853071795855)
    assert paper["conductance_into_proximal_nS"] == pytest.approx(24.543692606170264)
    assert kinness["conductance_into_soma_nS"] == pytest.approx(35.298793860559485)
    assert kinness["conductance_into_proximal_nS"] == pytest.approx(35.29879386055947)
    assert audit["selected_endpoint"]["endpoint_count"] == 1
    assert hashlib.sha256((ROOT / registration["profile"]).read_bytes()).hexdigest() == (
        registration["profile_sha256"]
    )
    assert registration["runtime_sha256"] == (
        "3e9bedd47822a785c0735afb4ab25bfa03e099b3258875fd434964255d102a85"
    )
    assert hashlib.sha256((ROOT / registration["source_control"]).read_bytes()).hexdigest() == (
        registration["source_control_sha256"]
    )
    assert registration["runtime_overrides"] == {
        "layer6i_axial_convention": "kinness_serialized_edge"
    }
    assert registration["required_prerequisites"]["projection025_connections"] == 81
    assert "No resistance or conductance scale" in registration["boundary"]


def test_layer6i_kinness_axial_reduces_output_but_not_reset() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure10-layer6i-axial-pair-630.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-axial-assessment-631.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["runtime_overrides"] == {
        "layer6i_axial_convention": "kinness_serialized_edge"
    }
    intact = result["intact"]
    control = result["disconnected_control"]
    assert intact["layer6i_post_events"] == control["layer6i_post_events"] == 40
    assert intact["layer6i_late_mismatch_events"] == (
        control["layer6i_late_mismatch_events"]
    ) == 24
    assert intact["layer6i_mismatch_active_indices"] == control[
        "layer6i_mismatch_active_indices"
    ]
    assert intact["layer6i_mismatch_events_by_index"] == control[
        "layer6i_mismatch_events_by_index"
    ]
    assert intact["layer4_post_events"] == control["layer4_post_events"] == 98
    projection025 = "modeldb112923.projection.025"
    assert intact["layer6i_mismatch_current_integral_pA_ms_by_projection"][
        projection025
    ] > 4.6 * control["layer6i_mismatch_current_integral_pA_ms_by_projection"][
        projection025
    ]
    assert result["reset_assessment"]["intact_winner_post_spikes"] == (
        result["reset_assessment"]["control_winner_post_spikes"]
    ) == 80
    assert result["reset_assessment"]["intact_released_alternatives"] == (
        result["reset_assessment"]["control_released_alternatives"]
    ) == 2
    assert assessment["assessment"]["layer6i_output_reduced_from_source_control"]
    assert assessment["assessment"]["family_closed"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["reproduced_reset"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer6i_timing_diagnostic_is_hash_pinned_and_read_only() -> None:
    monitor = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-timing-monitor-632.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-timing-registration-633.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("source_control", "source_control_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["script_sha256"] == (
        "27829c7fdfef9aa23995790eb465d4604a1bacff6b5d358f54f0d73bb8ab026e"
    )
    assert monitor["harness_sha256"] == registration["harness_sha256"]
    assert monitor["runner_sha256"] == registration["script_sha256"]
    assert monitor["runtime_sha256"] == registration["runtime_sha256"]
    assert registration["runtime_sha256"] == (
        "3e9bedd47822a785c0735afb4ab25bfa03e099b3258875fd434964255d102a85"
    )
    assert registration["runtime_fingerprint"] == (
        "fa4ab9f0bf2bec4d6ad53cb6a91620047689b6839b146ed7d777d42350c2cdf5"
    )
    assert registration["record_layer6i_diagnostics"]
    assert registration["record_layer6i_trace_indices"] == [0, 31, 40]
    assert not registration["record_reset_chain_diagnostics"]
    assert registration.get("runtime_overrides") is None
    assert monitor["detector_interpretation"] == {
        "coordinate": "absolute_physical",
        "event_threshold_mV": -20.0,
        "event_rule": "falling_threshold_crossing",
        "signed_gap": "event_threshold_mV-minus-detector_peak_mV",
        "positive_gap_meaning": "voltage peak remained below event threshold",
    }
    assert not monitor["serialization"]["full_time_series_persisted"]
    assert "No parameter can be selected" in registration["boundary"]


def test_layer6i_timing_localizes_isolated_subthreshold_peripheral_drive() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure10-layer6i-timing-pair-634.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-timing-assessment-635.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    intact_identity = result["intact"]["population_identity"]
    control_identity = result["disconnected_control"]["population_identity"]
    assert intact_identity["layer4_pre_post_events"] == [39, 92]
    assert control_identity["layer4_pre_post_events"] == [39, 92]
    assert intact_identity["nonspecific_pre_post_events"] == [4, 13]
    assert control_identity["nonspecific_pre_post_events"] == [4, 13]
    assert intact_identity["layer5_pre_post_events"] == [41, 231]
    assert control_identity["layer5_pre_post_events"] == [41, 140]
    assert intact_identity["layer6i_pre_post_events"] == [36, 72]
    assert control_identity["layer6i_pre_post_events"] == [36, 72]
    assert set(result["intact"]["selected_cells"]) == {0, 31, 40}
    intact_cell0 = result["intact"]["selected_cells"][0]
    control_cell0 = result["disconnected_control"]["selected_cells"][0]
    assert intact_cell0["mismatch_event_times_ms"] == []
    assert control_cell0["mismatch_event_times_ms"] == []
    projection023 = "modeldb112923.projection.023"
    projection024 = "modeldb112923.projection.024"
    projection025 = "modeldb112923.projection.025"
    assert intact_cell0["projections"][projection023]["current_integral_pA_ms"] == 0
    assert intact_cell0["projections"][projection024]["current_integral_pA_ms"] == 0
    assert intact_cell0["projections"][projection025][
        "current_integral_pA_ms"
    ] == pytest.approx(3013.832334921532)
    assert control_cell0["projections"][projection025][
        "current_integral_pA_ms"
    ] == 0
    assert intact_cell0["detector_threshold_peak_gap_mV"][2] == pytest.approx(
        48.519120517055725
    )
    cell31 = result["intact"]["selected_cells"][31]
    cell40 = result["intact"]["selected_cells"][40]
    assert cell31["projections"][projection024]["current_integral_pA_ms"] > (
        12.0 * cell31["projections"][projection025]["current_integral_pA_ms"]
    )
    assert cell40["projections"][projection023]["current_integral_pA_ms"] > (
        57.0 * cell40["projections"][projection025]["current_integral_pA_ms"]
    )
    assert assessment["assessment"]["amplitude_limitation_localized"]
    assert not assessment["assessment"]["timing_only_explanation_supported"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["reproduced_reset"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer6i_lossless_replay_is_hash_pinned_without_scale_search() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-replay-implementation-636.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-replay-registration-637.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("script", "script_sha256"),
        ("source_control", "source_control_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["harness_sha256"] == (
        "e280bfdc864c63630ac9e07a8ee980aa768015dce736bf04e5c98df9e0ef5a0f"
    )
    assert implementation["harness_sha256"] == registration["harness_sha256"]
    assert implementation["replay_module_sha256"] == registration[
        "replay_module_sha256"
    ]
    assert implementation["runner_sha256"] == registration["script_sha256"]
    assert registration["runtime_sha256"] == (
        "3e9bedd47822a785c0735afb4ab25bfa03e099b3258875fd434964255d102a85"
    )
    assert registration["trace"]["cell_index"] == 0
    assert registration["trace"]["replay_quantity"] == "receptor_gate_waveforms"
    assert not registration["trace"]["precomputed_projection_currents_replayed"]
    assert registration["intact_replay_gate"] == {
        "source_and_replay_spike_times_exact": True,
        "maximum_voltage_error_mV": 1.0e-12,
        "maximum_dimensionless_state_error": 1.0e-12,
    }
    assert "Do not run the disconnected arm" in registration["boundary"]
    assert "authorizes no scale search" in implementation["boundary"]


def test_layer6i_lossless_replay_passes_exact_identity_gate() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure10-layer6i-replay-638.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-replay-assessment-639.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert result["trace"]["path"] == (
        "results/figure10-layer6i-cell0-replay-638.npz"
    )
    assert result["trace"]["sha256"] == assessment["trace_sha256"]
    assert all(result["fresh_figure6_gates"].values())
    assert result["connected_source_identity_pass"]
    assert result["intact_replay"]["exact_spike_train"]
    assert result["trace"]["source_event_times_from_mismatch_ms"] == []
    assert result["intact_replay"]["source_spike_times_ms"] == []
    assert result["intact_replay"]["replay_spike_times_ms"] == []
    assert result["intact_replay_max_state_error"] == 0.0
    assert all(
        error == 0.0
        for _, error in result["intact_replay"]["max_abs_error_by_variable"]
    )
    assert assessment["assessment"][
        "replay_system_validated_for_isolated_conductance_sensitivity"
    ]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["parameter_search_performed"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer6i_conductance_bracket_is_hash_pinned_and_isolated() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-conductance-implementation-640.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-conductance-registration-641.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("replay_module", "replay_module_sha256"),
        ("script", "script_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert implementation["replay_module_sha256"] == registration[
        "replay_module_sha256"
    ]
    assert implementation["runner_sha256"] == registration["script_sha256"]
    assert registration["runtime_sha256"] == (
        "3e9bedd47822a785c0735afb4ab25bfa03e099b3258875fd434964255d102a85"
    )
    assert registration["projection025_conductance_scales"] == [
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
    ]
    assert implementation["implementation"]["maximal_conductance_scaled"]
    assert not implementation["implementation"]["captured_gate_scaled"]
    assert not implementation["implementation"]["event_multiplicity_changed"]
    assert not implementation["implementation"]["connected_network_changed"]
    assert "Do not interpolate" in registration["decision_rule"]


def test_layer6i_conductance_bracket_locates_first_event_without_selection() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer6i-conductance-bracket-642.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-conductance-assessment-643.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert result["scale_one_identity_gate_pass"]
    assert result["all_trials_finite"]
    assert [item["scale"] for item in result["scale_results"]] == [
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
    ]
    assert [item["event_count"] for item in result["scale_results"]] == [
        0,
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert result["first_event_scale"] == 8.0
    assert assessment["assessment"]["lower_endpoint_scale"] == 4.0
    assert assessment["assessment"]["upper_endpoint_scale"] == 8.0
    assert assessment["assessment"]["connected_network_test_authorized"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["connected_network_changed"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer6i_scale8_connected_pair_is_hash_pinned_and_single_endpoint() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-scale8-registration-644.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["harness_sha256"] == (
        "e280bfdc864c63630ac9e07a8ee980aa768015dce736bf04e5c98df9e0ef5a0f"
    )
    assert registration["script_sha256"] == (
        "27829c7fdfef9aa23995790eb465d4604a1bacff6b5d358f54f0d73bb8ab026e"
    )
    assert registration["runtime_sha256"] == (
        "3e9bedd47822a785c0735afb4ab25bfa03e099b3258875fd434964255d102a85"
    )
    assert registration["persistent_projection_scales"][-1] == {
        "projection_id": "modeldb112923.projection.025",
        "scale": 8.0,
    }
    assert registration["required_prerequisites"]["figure6_all_gates"]
    assert registration["record_layer6i_diagnostics"]
    assert registration["record_reset_chain_diagnostics"]
    assert registration["record_layer6i_trace_indices"] == [0, 31, 40]
    assert "Execute one connected intact/control pair" in registration["boundary"]
    assert "No baseline promotion" in registration["boundary"]


def test_layer6i_scale8_restores_broad_wave_but_fails_reset_sign() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure10-layer6i-scale8-pair-645.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-scale8-assessment-646.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["intact"]["layer6i_mismatch_active_count"] == 81
    assert result["disconnected_control"]["layer6i_mismatch_active_count"] == 11
    assert result["intact"]["layer6i_post_events"] == 153
    assert result["disconnected_control"]["layer6i_post_events"] == 76
    assert result["intact"]["layer4_inhibitory_post_events"] == 218
    assert result["disconnected_control"]["layer4_inhibitory_post_events"] == 232
    assert abs(
        result["intact"]["layer4e_mismatch_projection036_current_integral_pA_ms"]
    ) < abs(
        result["disconnected_control"][
            "layer4e_mismatch_projection036_current_integral_pA_ms"
        ]
    )
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": False,
        "alternative_release": False,
    }
    assert result["reset_assessment"]["intact_winner_post_spikes"] == 78
    assert result["reset_assessment"]["control_winner_post_spikes"] == 76
    assert result["reset_assessment"]["intact_released_alternatives"] == 0
    assert result["reset_assessment"]["control_released_alternatives"] == 2
    assert assessment["assessment"]["scale8_endpoint_closed"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["reproduced_reset"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer4_balance_audit_is_hash_pinned_and_read_only() -> None:
    monitor = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-balance-monitor-647.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-balance-registration-648.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert monitor["harness_sha256"] == registration["harness_sha256"]
    assert monitor["runner_sha256"] == registration["script_sha256"]
    assert registration["harness_sha256"] == (
        "1e061546e8d928cd7c7d959cb6fe11218c7c9165d7d3ce2d8536898fac1bb367"
    )
    assert registration["script_sha256"] == (
        "71bc3db5bbc989baf1e1157791b57f6dbc18008f4eaf19950868870eff775390"
    )
    assert registration["runtime_sha256"] == (
        "3e9bedd47822a785c0735afb4ab25bfa03e099b3258875fd434964255d102a85"
    )
    assert [item["projection_id"] for item in monitor["audited_paths"]] == [
        "modeldb112923.projection.026",
        "modeldb112923.projection.036",
        "modeldb112923.projection.038",
    ]
    assert registration["balance_bin_width_ms"] == 10.0
    assert registration["focal_windows_from_mismatch_ms"] == [
        [100.0, 110.0],
        [110.0, 120.0],
    ]
    assert registration["record_reset_chain_diagnostics"]
    assert registration["record_layer4_balance_diagnostics"]
    assert registration["output_mode"] == "layer4_balance_bounded"
    assert "No parameter" in registration["boundary"]


def test_layer4_balance_localizes_first_wrong_sign_at_projection036() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure10-layer4-balance-pair-649.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-balance-assessment-650.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer6i_post_events_intact_control"] == [
        153,
        76,
    ]
    assert len(result["intact_balance_bins"]) == 20
    assert len(result["control_balance_bins"]) == 20
    intact_focal = result["intact_balance_bins"][10]
    control_focal = result["control_balance_bins"][10]
    assert intact_focal[:2] == [100.0, 110.0]
    assert control_focal[:2] == [100.0, 110.0]
    assert intact_focal[2] > control_focal[2]
    assert intact_focal[3] > control_focal[3]
    assert abs(intact_focal[4]) < abs(control_focal[4])
    assert intact_focal[5] > control_focal[5]
    assert intact_focal[6] > control_focal[6]
    assert intact_focal[8] < control_focal[8]
    assert assessment["assessment"]["first_wrong_sign_stage"] == (
        "projection036-output"
    )
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["reproduced_reset"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


@pytest.mark.parametrize(
    "registration_name",
    [
        "figure10-projection036-ring-registration-653.yaml",
        "figure10-layer4-target-balance-registration-657.yaml",
        "figure10-layer4-complete-input-registration-661.yaml",
        "figure10-layer4-subbin-timing-registration-665.yaml",
        "figure10-layer4-gate-timing-registration-669.yaml",
        "figure10-layer4-native-gate-onset-registration-673.yaml",
        "figure10-projection036-delay0p2-registration-677.yaml",
        "figure10-projection026-supplement-delay-registration-681.yaml",
        "figure10-projection036-source-arrival-registration-685.yaml",
        "figure10-layer4i-source-phase-registration-689.yaml",
        "figure10-layer4i-source-phase-file-registration-693.yaml",
        "figure10-layer4i-source-phase-wide-registration-696.yaml",
        "figure10-layer4i-source-phase-from-onset-registration-699.yaml",
        "figure10-projection026-source-resource-registration-710.yaml",
        "figure10-projection026-focal-resource-registration-714.yaml",
        "figure10-layer6i-output-depletion-registration-718.yaml",
    ],
)
def test_historical_figure10_runtime_digest_remains_pinned(
    registration_name: str,
) -> None:
    registration = yaml.safe_load(
        (ROOT / "docs/validation-results" / registration_name).read_text()
    )
    assert registration["runtime_sha256"] == HISTORICAL_FIGURE10_RUNTIME_SHA256


def test_projection036_ring_audit_preserves_source_boundary() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-ring-source-audit-651.yaml"
        ).read_text()
    )

    assert audit["projection"]["id"] == "modeldb112923.projection.036"
    assert audit["projection"]["ring"]
    assert audit["projection"]["sigma_x_y"] == [1.5, 1.5]
    assert not audit["source_evidence"]["exact_legacy_ring_geometry_recovered"]
    active = audit["active_center_excluded_gaussian"]
    annulus = audit["parameter_free_radial_annulus"]
    assert active["incoming_edges_per_target"] == 80
    assert annulus["incoming_edges_per_target"] == 80
    assert not active["center_edge_included"]
    assert not annulus["center_edge_included"]
    assert annulus["radius_scale"] == 1.0
    assert annulus["factor_sum_ratio_to_active"] == pytest.approx(
        2.8567374243691
    )
    assert "projection-036-only" in audit["decision"]
    assert "No weight/radius tuning" in (
        ROOT / "docs/parameter-provenance.yaml"
    ).read_text()


def test_projection036_ring_endpoint_is_hash_pinned_and_bounded() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-ring-implementation-652.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-ring-registration-653.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert implementation["runtime_sha256"] == registration["runtime_sha256"]
    assert registration["harness_sha256"] == (
        "1e061546e8d928cd7c7d959cb6fe11218c7c9165d7d3ce2d8536898fac1bb367"
    )
    assert registration["script_sha256"] == (
        "71bc3db5bbc989baf1e1157791b57f6dbc18008f4eaf19950868870eff775390"
    )
    assert registration["runtime_overrides"] == {
        "projection036_ring_kernel_convention": "radial_annulus"
    }
    assert registration["persistent_projection_scales"][-1] == {
        "projection_id": "modeldb112923.projection.025",
        "scale": 8.0,
    }
    assert registration["record_reset_chain_diagnostics"]
    assert registration["record_layer4_balance_diagnostics"]
    assert registration["output_mode"] == "layer4_balance_bounded"
    assert "all four preregistered gates" in registration["decision_rule"]
    assert "Do not tune or refine radius" in registration["boundary"]
    assert "baseline promotion" in registration["boundary"]


def test_projection036_radial_annulus_fails_reset_and_closes_endpoint() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection036-ring-pair-654.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-ring-assessment-655.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer6i_active_cells_intact_control"] == [
        81,
        17,
    ]
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": False,
        "alternative_release": False,
    }
    assert result["reset_assessment"]["intact_winner_post_spikes"] == 72
    assert result["reset_assessment"]["control_winner_post_spikes"] == 64
    assert result["reset_assessment"]["intact_released_alternatives"] == 2
    assert result["reset_assessment"]["control_released_alternatives"] == 2
    assert len(result["intact_balance_bins"]) == 20
    assert len(result["control_balance_bins"]) == 20
    aggregate = assessment["aggregate_intact_control"]
    assert aggregate["projection036_inhibitory_magnitude_ratio"] == pytest.approx(
        1.0002094662547423
    )
    assert aggregate["projection038_ratio"] == pytest.approx(2.1091324757775265)
    assert assessment["assessment"][
        "projection036_radial_annulus_endpoint_closed"
    ]
    assert not assessment["assessment"]["radius_or_weight_refinement_authorized"]
    assert not assessment["assessment"]["official_figure10_reset_reproduced"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer4_target_balance_audit_is_hash_pinned_and_read_only() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-target-balance-monitor-656.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-target-balance-registration-657.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert implementation["harness_sha256"] == registration["harness_sha256"]
    assert implementation["runner_sha256"] == registration["script_sha256"]
    assert registration["harness_sha256"] == (
        "927dd94a24a5d6a8f25aa1d04160f35ffcbb341bd16a94547675949b8fb19aca"
    )
    assert registration["script_sha256"] == (
        "1d837d92be8824256dbc3f99e5ee12acacd406fe121ea994fa492960145abea1"
    )
    assert registration["record_layer4_target_balance_indices"] == [
        31,
        38,
        39,
        40,
        41,
        42,
        49,
    ]
    assert registration["target_groups"] == {
        "pre_reset_winner": [38, 39, 40, 41, 42],
        "prior_control_alternatives": [31, 49],
    }
    assert registration.get("runtime_overrides") is None
    assert registration["record_layer4_balance_diagnostics"]
    assert "unchanged source-control" in registration["boundary"]
    assert "No parameter selection" in registration["boundary"]


def test_layer4_target_balance_localizes_complete_input_gap() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4-target-balance-pair-658.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-target-balance-assessment-659.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer4_post_events"] == [78, 92]
    assert result["source_identity"]["layer6i_post_events"] == [153, 76]
    assert result["reset_gates"]["winner_suppression"] is False
    assert result["reset_gates"]["alternative_release"] is False
    for arm in ("intact", "disconnected_control"):
        assert set(result["target_series"][arm]) == {
            "31",
            "38",
            "39",
            "40",
            "41",
            "42",
            "49",
        }
        for series in result["target_series"][arm].values():
            assert all(len(values) == 20 for values in series)
    winner = assessment["winner_group_38_42"]["late_100_200_ms"]
    assert winner["projection036_ratio"] == pytest.approx(0.9510632777019089)
    assert winner["projection038_ratio"] == pytest.approx(1.1074690716568674)
    alternatives = assessment["alternative_group_31_49"]["full_mismatch"]
    assert alternatives["projection036_ratio"] == pytest.approx(0.5524496270415391)
    assert alternatives["events_intact_control"] == [0, 16]
    assert assessment["assessment"]["complete_layer4_input_balance_required"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not assessment["assessment"]["official_figure10_reset_reproduced"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer4_complete_input_audit_is_hash_pinned_and_read_only() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-complete-input-monitor-660.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-complete-input-registration-661.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert implementation["harness_sha256"] == registration["harness_sha256"]
    assert implementation["runner_sha256"] == registration["script_sha256"]
    assert registration["harness_sha256"] == (
        "eaba419ebab808f8540a17928882cd070eca376e8664ca75b7a28716cd3eb093"
    )
    assert registration["script_sha256"] == (
        "f59d35c19f56cba6df148f4f2cb35a5b071e6daea3ab7c216bf4895da0218e52"
    )
    assert [item["projection_id"] for item in implementation["target_inputs"]] == [
        "modeldb112923.projection.035",
        "modeldb112923.projection.036",
        "modeldb112923.projection.037",
        "modeldb112923.projection.038",
    ]
    assert registration["output_mode"] == "layer4_complete_target_bounded"
    assert registration["record_layer4_target_balance_indices"] == [
        31,
        38,
        39,
        40,
        41,
        42,
        49,
    ]
    assert registration.get("runtime_overrides") is None
    assert "No parameter selection" in registration["boundary"]


def test_layer4_complete_input_result_localizes_first_divergence() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4-complete-input-pair-662.yaml"
    )
    prior_path = (
        ROOT
        / "docs/validation-results/figure10-layer4-target-balance-pair-658.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-complete-input-assessment-663.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "new_input_result_sha256"
    ]
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == assessment[
        "prior_target_result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer4_post_events"] == [78, 92]
    assert result["source_identity"]["layer6i_post_events"] == [153, 76]
    for arm in ("intact", "disconnected_control"):
        assert set(result["new_input_series"][arm]) == {
            "31",
            "38",
            "39",
            "40",
            "41",
            "42",
            "49",
        }
        for series in result["new_input_series"][arm].values():
            assert len(series) == 2
            assert all(len(values) == 20 for values in series)
    first = assessment["alternative_group_31_49"]["first_divergence_70_80_ms"]
    assert first["events_intact_control"] == [0, 2]
    assert first["projection037_excitation_pA_ms_intact_control"] == [0.0, 290.147379]
    assert assessment["assessment"]["projection037_is_initial_release_drive"] == (
        "unresolved-with-ten-ms-bins"
    )
    assert assessment["assessment"]["projection037_is_positive_feedback_after_release"] == (
        "supported"
    )
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer4_subbin_timing_audit_is_hash_pinned_and_read_only() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-subbin-timing-monitor-664.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-subbin-timing-registration-665.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert implementation["harness_sha256"] == registration["harness_sha256"]
    assert implementation["runner_sha256"] == registration["script_sha256"]
    assert registration["harness_sha256"] == (
        "ccd45605d8406b2eb7f5f41f63860b21f465283f10ee058895887186cfb0eb7f"
    )
    assert registration["script_sha256"] == (
        "836b050b9c3ecfd1e8c27d719bb093c0b8ea10966ed6a76355db37719c4d10f9"
    )
    assert registration["record_layer4_target_timing_indices"] == [31, 49]
    assert registration["layer4_target_timing_window_ms"] == [65.0, 85.0]
    assert registration["layer4_target_timing_bin_width_ms"] == 1.0
    assert registration["layer4_target_timing_current_threshold_pA"] == 1e-9
    assert registration["output_mode"] == "layer4_target_timing_bounded"
    assert registration.get("runtime_overrides") is None
    assert "Do not select a parameter" in registration["decision_rule"]
    assert "No parameter selection" in registration["boundary"]


def test_layer4_subbin_timing_classifies_recurrence_as_feedback() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4-subbin-timing-pair-666.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-subbin-timing-assessment-667.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer4_post_events"] == [78, 92]
    assert result["source_identity"]["layer6i_post_events"] == [153, 76]
    for index in ("31", "49"):
        intact = result["causal_order_by_target"]["intact"][index]
        control = result["causal_order_by_target"]["disconnected_control"][index]
        assert intact["spike_times_from_mismatch_ms"] == []
        assert intact["projection037_first_active_time_from_mismatch_ms"] is None
        assert control["spike_times_from_mismatch_ms"] == [77.33000000000001]
        assert control["projection037_first_active_time_from_mismatch_ms"] == (
            77.45000000000002
        )
    assert assessment["assessment"]["projection037_is_initial_release_drive"] is False
    assert assessment["assessment"]["projection037_is_post_spike_feedback_amplifier"]
    assert assessment["causal_order"]["projection037_lag_after_spike_ms"] == pytest.approx(
        0.12
    )
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer4_gate_timing_audit_is_hash_pinned_and_read_only() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-gate-timing-monitor-668.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-gate-timing-registration-669.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert implementation["harness_sha256"] == registration["harness_sha256"]
    assert implementation["runner_sha256"] == registration["script_sha256"]
    assert registration["harness_sha256"] == (
        "2ad42f286a91adbb556cdddca8f7fb88b16118bcfa443e498fff81f890483746"
    )
    assert registration["script_sha256"] == (
        "836b050b9c3ecfd1e8c27d719bb093c0b8ea10966ed6a76355db37719c4d10f9"
    )
    assert registration["record_layer4_target_timing_indices"] == [31]
    assert registration["layer4_target_timing_window_ms"] == [74.0, 78.0]
    assert registration["layer4_target_timing_bin_width_ms"] == 0.1
    assert registration["voltage_divergence_threshold_mV"] == 0.1
    assert registration["output_mode"] == "layer4_target_timing_bounded"
    assert registration.get("runtime_overrides") is None
    assert "Do not select or fit a parameter" in registration["decision_rule"]
    assert "No parameter selection" in registration["boundary"]


def test_layer4_gate_timing_localizes_inhibitory_arrival_candidate() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4-gate-timing-pair-670.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-gate-timing-assessment-671.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer4_post_events"] == [78, 92]
    assert result["source_identity"]["layer6i_post_events"] == [153, 76]
    endpoint = result["preregistered_first_voltage_divergence"]
    assert endpoint["preceding_bin_available"] is False
    assert endpoint["classification"] == "left-censored-by-registered-window"
    sequence = assessment["localized_sequence"]
    assert sequence["projection036_intact_surge_bin_ms"] == [75.4, 75.5]
    assert sequence["projection036_control_surge_bin_ms"] == [75.5, 75.6]
    assert sequence["inhibitory_arrival_lead_intact_ms"] == 0.1
    assert assessment["assessment"]["inhibitory_arrival_is_candidate_release_block"]
    assert not assessment["assessment"]["preregistered_endpoint_decisive"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer4_native_gate_onset_confirmation_is_hash_pinned() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-native-gate-onset-monitor-672.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-native-gate-onset-registration-673.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert implementation["harness_sha256"] == registration["harness_sha256"]
    assert implementation["runner_sha256"] == registration["script_sha256"]
    assert registration["harness_sha256"] == (
        "a985727d1a1b87e676452d8878b639b6a53250244508c08bcb677663cd7f2446"
    )
    assert registration["script_sha256"] == (
        "94fcd7d827cb024964ba5f0b461e265b0f5e52da5bf34e0470a04db70287199e"
    )
    assert registration["record_layer4_target_timing_indices"] == [31]
    assert registration["layer4_target_timing_window_ms"] == [73.0, 77.0]
    assert registration["layer4_target_timing_gate_threshold"] == 0.1
    assert registration["protocol"]["dt_ms"] == 0.01
    assert registration.get("runtime_overrides") is None
    assert "do not execute or select" in registration["decision_rule"]


def test_layer4_native_gate_onset_authorizes_single_delay_endpoint() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4-native-gate-onset-pair-674.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4-native-gate-onset-assessment-675.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer4_post_events"] == [78, 92]
    inhibition = result["native_gate_onsets_from_mismatch_ms"][
        "projection036_inhibition"
    ]
    assert inhibition["intact_strictly_precedes_control"]
    assert inhibition["minimum_supported_lead_ms"] > 0.05
    assert assessment["assessment"]["inhibitory_arrival_candidate_confirmed"]
    assert assessment["assessment"]["projection036_delay_family_authorized"]
    assert not assessment["assessment"]["projection036_delay_family_executed"]
    assert "projection-036 delay 0.2 ms" in assessment["authorization"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_projection036_delay0p2_endpoint_is_hash_pinned_and_bounded() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-delay0p2-implementation-676.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-delay0p2-registration-677.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert implementation["harness_sha256"] == registration["harness_sha256"]
    assert implementation["runner_sha256"] == registration["script_sha256"]
    assert registration["harness_sha256"] == (
        "82af92d2da5c5ccdba24ee049a1ea07794c3726a167d2fca9b5f8d389b640b26"
    )
    assert registration["script_sha256"] == (
        "1df663f30a4d16e6a860d795a266a68482ee01fbb371038ce243b81ed79195c1"
    )
    assert registration["persistent_projection_delays"] == [
        {"projection_id": "modeldb112923.projection.036", "delay_ms": 0.2}
    ]
    assert registration["output_mode"] == "layer4_balance_bounded"
    assert "Do not refine" in registration["decision_rule"]
    assert "No original-SMART claim" in registration["boundary"]


def test_projection036_delay0p2_fails_reset_and_closes_endpoint() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection036-delay0p2-pair-678.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-delay0p2-assessment-679.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["persistent_projection_delays_ms"] == {
        "modeldb112923.projection.036": 0.2
    }
    assert result["source_identity"]["layer4_post_events_intact_control"] == [79, 76]
    assert result["source_identity"]["layer6i_active_cells_intact_control"] == [81, 5]
    assert result["source_identity"]["layer4_inhibitory_post_events_intact_control"] == [
        218,
        218,
    ]
    assert result["baseline_comparison"]["endpoint_intact_minus_control_winner_events"] == 3
    assert result["baseline_comparison"]["endpoint_released_alternatives_intact_control"] == [
        0,
        0,
    ]
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": False,
        "alternative_release": False,
    }
    verdict = assessment["assessment"]
    assert not verdict["winner_difference_not_worsened"]
    assert not verdict["control_alternative_release_preserved"]
    assert not verdict["projection036_delay0p2_endpoint_selected"]
    assert verdict["projection036_delay0p2_endpoint_closed"]
    assert not verdict["delay_refinement_or_interpolation_authorized"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_projection026_supplement_delay_endpoint_is_source_pinned_and_bounded() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-supplement-delay-implementation-680.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-supplement-delay-registration-681.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("supplementary_catalog", "supplementary_catalog_sha256"),
        ("executable_catalog", "executable_catalog_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["harness_sha256"] == (
        "4c4e7effe3157a25e24dac9724054a346c91b79eac325ff28eb70d7e12fcbb9a"
    )
    assert registration["script_sha256"] == (
        "1df663f30a4d16e6a860d795a266a68482ee01fbb371038ce243b81ed79195c1"
    )
    conflict = implementation["source_conflict"]
    assert conflict["supplementary_delay_ms"] == 0.1
    assert conflict["executable_delay_ms"] == 1.0
    assert conflict["comparison_projection_delay_ms_both_sources"] == 1.0
    assert registration["persistent_projection_delays"] == [
        {"projection_id": "modeldb112923.projection.026", "delay_ms": 0.1}
    ]
    assert registration["source_cross"]["projection036_delay_ms"] == 0.1
    assert registration["source_cross"]["projection038_delay_ms"] == 1.0
    assert registration["record_layer4_target_timing_indices"] == [31, 40]
    assert registration["layer4_target_timing_window_ms"] == [73.0, 78.0]
    assert registration["output_mode"] == "layer4_target_timing_bounded"
    assert "without an intermediate value" in registration["decision_rule"]
    assert "cannot establish a global" in registration["boundary"]
    assert "until the same choice passes fresh learning" in registration["boundary"]


def test_projection026_supplement_delay_fails_reset_and_closes_conflict() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection026-supplement-delay-pair-682.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-supplement-delay-assessment-683.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["persistent_projection_delays_ms"] == {
        "modeldb112923.projection.026": 0.1
    }
    assert result["reset_assessment"]["intact_winner_post_spikes"] == 81
    assert result["reset_assessment"]["control_winner_post_spikes"] == 80
    assert result["reset_assessment"]["intact_released_alternatives"] == 0
    assert result["reset_assessment"]["control_released_alternatives"] == 2
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": False,
        "alternative_release": False,
    }
    timing = result["retained_control_target_timing"]
    assert timing["cell31"]["projection038_first_gate_threshold_ms"] is None
    assert timing["cell40"]["projection038_first_gate_threshold_ms"] is None
    verdict = assessment["assessment"]
    assert not verdict["projection026_supplement_delay_selected"]
    assert verdict["projection026_delay_conflict_closed"]
    assert not verdict["delay_grid_or_interpolation_authorized"]
    assert not verdict["global_classic_convention_authorized"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_projection036_source_arrival_audit_is_hash_pinned_and_passive() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-source-arrival-monitor-684.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-source-arrival-registration-685.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["harness_sha256"] == (
        "03e5d2a256ba55f450a1a21a87d991e65d634a199bec63d6b16c4395a36f9cee"
    )
    assert registration["script_sha256"] == (
        "757beafe9436d6fe714a173d57b28a0e828f89d0979704a82a092820e2b75ac9"
    )
    assert implementation["implementation"]["behavior_change"] == "none"
    assert registration["persistent_projection_delays"] == []
    assert registration["record_projection036_arrival_target_indices"] == [31]
    assert registration["projection036_arrival_window_ms"] == [75.2, 75.8]
    assert registration["required_source_identity"][
        "layer4_post_events_intact_control"
    ] == [78, 92]
    assert "Do not select a parameter" in registration["decision_rule"]
    assert "No original-SMART claim" in registration["boundary"]


def test_layer4i_source_phase_audit_is_hash_pinned_complete_and_passive() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-monitor-688.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-registration-689.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["harness_sha256"] == (
        "f9e0ce7d772881f776475184bab7fa440537c7c62ce9dec05614ae12fff24fbf"
    )
    assert registration["script_sha256"] == (
        "03089bf8c8f12fed34bf499a68864b509b072279a7d0947eff9dc4077d24a19f"
    )
    assert implementation["implementation"]["behavior_change"] == "none"
    assert set(implementation["port_identity"].values()) == {
        f"modeldb112923.projection.{index:03d}" for index in range(26, 31)
    }
    assert registration["persistent_projection_delays"] == []
    assert registration["record_layer4_inhibitory_trace_indices"] == [38, 40, 42]
    assert registration["layer4_inhibitory_trace_window_ms"] == [75.0, 75.6]
    assert registration["required_arrival_identity"]["source_set_both_arms"] == [
        38,
        39,
        40,
        41,
        42,
    ]
    assert "Do not select a parameter" in registration["decision_rule"]
    assert "No original-SMART claim" in registration["boundary"]


def test_layer4i_source_phase_compacted_run_is_not_interpreted() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4i-source-phase-pair-690.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-assessment-691.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["execution_exit_code"] == 0
    assert not result["required_control_source_identity_preserved"]
    assert not result["complete_native_pair_trace_preserved"]
    assert not assessment["assessment"]["mechanistic_interpretation_authorized"]
    assert not assessment["assessment"]["parameter_selected"]


def test_layer4i_source_phase_file_rerun_is_hash_pinned_and_passive() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-file-monitor-692.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-file-registration-693.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["harness_sha256"] == (
        "788fe02a64f8c524c5df8e23ef1c54dbbf2593abab57ae1ef87999b0ba52d1cb"
    )
    assert registration["script_sha256"] == (
        "d288c991c6f2daf34eed452be48a0156c1934265b9c5e755887dce94c9e05f8e"
    )
    assert implementation["implementation"]["behavior_change"] == "none"
    assert implementation["implementation"]["state_monitor_rows"] == [38, 40, 42]
    assert registration["output_mode"] == "layer4_source_trace_files"
    assert registration["record_layer4_inhibitory_trace_indices"] == [38, 40, 42]
    assert registration["intact_layer4_inhibitory_trace_output"].endswith(".npz")
    assert registration["control_layer4_inhibitory_trace_output"].endswith(".npz")
    assert "left-censored" in registration["decision_rule"]
    assert "No original-SMART claim" in registration["boundary"]


def test_layer4i_source_phase_file_pair_is_exact_and_left_censored() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4i-source-phase-file-pair-694.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-file-assessment-695.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer4_post_events"] == [78, 92]
    assert result["arrival_identity"]["intact_first_source_set"] == [38, 42]
    assert result["arrival_identity"]["control_first_source_set"] == [40]
    assert result["trace_contract"]["samples_per_source"] == 61
    assert result["trace_contract"]["intact"]["sha256"] == (
        "79e24d69d2aeefb76cd53bdd05c662a0060073537c40ae02c0d39eca6f927024"
    )
    assert result["trace_contract"]["disconnected_control"]["sha256"] == (
        "12f0bf5e1bc2ccbac5aab4b030a4ab71ab3166572eec94d0699212a19cb753bf"
    )
    endpoint = result["left_endpoint_evidence"]["soma_voltage_mV"]
    assert min(endpoint["intact_cells_38_40_42"]) > 0
    assert min(endpoint["control_cells_38_40_42"]) > 0
    verdict = assessment["assessment"]
    assert verdict["exact_registered_population_identity"]
    assert verdict["exact_registered_arrival_identity"]
    assert verdict["window_left_censored"]
    assert not verdict["input_path_selected_as_initiator"]
    assert not verdict["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer4i_source_phase_wide_trace_is_hash_pinned_and_passive() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-wide-registration-696.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["harness_sha256"] == (
        "788fe02a64f8c524c5df8e23ef1c54dbbf2593abab57ae1ef87999b0ba52d1cb"
    )
    assert registration["script_sha256"] == (
        "d288c991c6f2daf34eed452be48a0156c1934265b9c5e755887dce94c9e05f8e"
    )
    assert registration["record_layer4_inhibitory_trace_indices"] == [38, 40, 42]
    assert registration["layer4_inhibitory_trace_window_ms"] == [73.5, 75.6]
    assert registration["persistent_projection_delays"] == []
    assert registration["output_mode"] == "layer4_source_trace_files"
    assert "voltage-dependent" in registration["decision_rule"]
    assert "Do not fit or select a parameter" in registration["decision_rule"]
    assert "No original-SMART claim" in registration["boundary"]


def test_layer4i_source_phase_from_onset_is_hash_pinned_and_passive() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-from-onset-registration-699.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["harness_sha256"] == (
        "788fe02a64f8c524c5df8e23ef1c54dbbf2593abab57ae1ef87999b0ba52d1cb"
    )
    assert registration["script_sha256"] == (
        "d288c991c6f2daf34eed452be48a0156c1934265b9c5e755887dce94c9e05f8e"
    )
    assert registration["record_layer4_inhibitory_trace_indices"] == [38, 40, 42]
    assert registration["layer4_inhibitory_trace_window_ms"] == [0.0, 75.6]
    assert registration["persistent_projection_delays"] == []
    assert registration["output_mode"] == "layer4_source_trace_files"
    assert "1e-12" in registration["decision_rule"]
    assert "not as a calibrated or recovered parameter" in registration[
        "decision_rule"
    ]
    assert "No original-SMART claim" in registration["boundary"]


def test_layer4i_source_phase_wide_pair_is_exact_and_gate_left_censored() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4i-source-phase-wide-pair-697.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-wide-assessment-698.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer4_post_events"] == [78, 92]
    assert result["trace_contract"]["samples_per_source"] == 211
    assert result["left_endpoint_evidence"]["projection027_gate"][
        "identical_at_start"
    ]
    assert result["left_endpoint_evidence"]["projection028_gate"][
        "identical_at_start"
    ]
    assert assessment["assessment"]["action_potential_rise_captured"]
    assert assessment["assessment"][
        "projection026_gate_difference_present_at_window_start"
    ]
    assert assessment["assessment"][
        "projection030_gate_difference_present_at_window_start"
    ]
    assert assessment["assessment"]["causal_gate_onset_left_censored"]
    assert not assessment["assessment"]["initiating_gate_identified"]
    assert not assessment["assessment"]["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_layer4i_source_phase_from_onset_localizes_projection026() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer4i-source-phase-from-onset-pair-700.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer4i-source-phase-from-onset-assessment-701.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["trace_contract"][
        "all_retained_variables_equal_at_mismatch_onset_within_1e_minus_12"
    ]
    assert result["trace_contract"]["samples_per_source"] == 7561
    first = result["first_arm_difference_by_variable_ms"]
    assert first["projection026_gate"] == pytest.approx(23.45)
    assert first["projection030_gate"] == pytest.approx(72.88)
    assert first["projection027_gate"] == pytest.approx(73.85)
    assert first["projection028_gate"] == pytest.approx(75.40)
    verdict = assessment["assessment"]
    assert verdict["projection026_is_first_measured_presynaptic_gate_difference"]
    assert verdict["projection026_diagnostic_target_localized"]
    assert not verdict["initiating_projection_parameter_identified"]
    assert not verdict["parameter_selected"]
    assert not verdict["official_figure10_reset_reproduced"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_projection026_source_arrival_audit_is_hash_pinned_and_passive() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-source-arrival-monitor-702.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-source-arrival-registration-703.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["script_sha256"] == (
        "df9fa6ccbae6ef996eccc6b62e3caba32f2a7559489e0300a0e0d1542764a28d"
    )
    assert implementation["implementation"]["behavior_change"] == "none"
    assert implementation["implementation"]["source_population"] == (
        "layer6i_excitatory_v1"
    )
    assert registration["persistent_projection_delays"] == []
    assert registration["record_projection026_arrival_target_indices"] == [
        38,
        40,
        42,
    ]
    assert registration["projection026_arrival_window_ms"] == [23.3, 23.6]
    assert registration["required_prior_trace_identity"]["samples_per_source"] == 7561
    assert "Do not fit or select" in registration["decision_rule"]
    assert "No original-SMART claim" in registration["boundary"]


def test_projection026_source_arrival_pair_is_exact_and_left_censored() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection026-source-arrival-pair-704.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-source-arrival-assessment-705.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["source_identity"]["layer6i_post_events"] == [153, 76]
    contract = result["projection026_arrival_contract"]
    assert contract["effective_delay_ms"] == pytest.approx(1.0)
    assert contract["total_edge_weight_each_target"] == pytest.approx(
        3.2292073770027754
    )
    assert contract["connected_source_and_weight_sets_identical_between_arms"]
    assert contract["no_arrivals_in_registered_window_both_arms"]
    verdict = assessment["assessment"]
    assert verdict["identical_projection026_connected_source_sets"]
    assert verdict["identical_projection026_compiled_edge_weights"]
    assert not verdict["projection026_arrival_present_in_registered_window"]
    assert verdict["causal_source_history_left_censored"]
    assert not verdict["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_projection026_source_history_registration_is_hash_pinned() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-source-arrival-from-onset-registration-706.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["script_sha256"] == (
        "df9fa6ccbae6ef996eccc6b62e3caba32f2a7559489e0300a0e0d1542764a28d"
    )
    assert registration["persistent_projection_delays"] == []
    assert registration["record_projection026_arrival_target_indices"] == [
        38,
        40,
        42,
    ]
    assert registration["projection026_arrival_window_ms"] == [0.0, 23.6]
    assert registration["required_prior_trace_identity"][
        "first_projection026_gate_difference_ms"
    ] == pytest.approx(23.45)
    assert "Only the passive" in registration["boundary"]
    assert "No original-SMART claim" in registration["boundary"]


def test_projection026_source_history_pair_is_exact_and_resource_candidate() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection026-source-arrival-from-onset-pair-707.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-source-arrival-from-onset-assessment-708.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    contract = result["projection026_arrival_contract"]
    assert contract["arrival_histories_identical_between_arms"]
    assert contract["last_arrival_before_gate_difference_ms"][40] == pytest.approx(
        19.86
    )
    verdict = assessment["assessment"]
    assert verdict["identical_projection026_arrival_histories_through_23p6_ms"]
    assert not verdict["delayed_ligand_arrival_explains_23p45_gate_difference"]
    assert verdict["source_transmitter_event_semantics_remain_candidate"]
    assert not verdict["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_projection026_source_resource_audit_is_hash_pinned_and_passive() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-source-resource-monitor-709.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-source-resource-registration-710.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["script_sha256"] == (
        "b8c10ed056900959ee61bb9d88c8733ed59cd7804152acc1c27ae671c5b0a08d"
    )
    assert implementation["implementation"]["behavior_change"] == "none"
    assert registration["output_mode"] == "projection026_source_resource_bounded"
    assert registration["projection026_arrival_window_ms"] == [0.0, 24.6]
    assert registration["persistent_projection_delays"] == []
    assert "Only output retention" in registration["boundary"]
    assert "No original-SMART claim" in registration["boundary"]


def test_projection026_source_resource_pair_localizes_timing_only() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection026-source-resource-pair-711.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-source-resource-assessment-712.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    timing = result["projection026_source_timing"]
    assert timing["focal_sources"] == [39, 41]
    assert timing["intact_source_spike_time_from_mismatch_ms"] == pytest.approx(
        23.43
    )
    assert timing[
        "disconnected_control_source_spike_time_from_mismatch_ms"
    ] == pytest.approx(23.44)
    assert timing["source_phase_lead_intact_ms"] == pytest.approx(0.01)
    verdict = assessment["assessment"]
    assert verdict["source_phase_difference_localized"]
    assert verdict["timing_consistent_with_first_projection026_gate_difference"]
    assert not verdict["complete_paired_focal_transmitter_values_preserved"]
    assert not verdict["transmitter_magnitude_interpretation_authorized"]
    assert not verdict["parameter_selected"]


def test_projection026_focal_resource_recovery_is_hash_pinned_and_passive() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-focal-resource-output-713.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-focal-resource-registration-714.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["script_sha256"] == (
        "f137a92900487e4f5c954ff044e0e5086e4304fa1d0fdcf7516a7fa3fa1c322b"
    )
    assert registration["runtime_sha256"] == (
        "ebdf48f0138803ab50b1dec2ef87817b4c537e4da490a088b9df6bdcb9d119a1"
    )
    assert implementation["implementation"]["behavior_change"] == "none"
    assert registration["layer6i_source_resource_indices"] == [39, 41]
    assert registration["layer6i_source_resource_window_ms"] == [23.3, 23.6]
    assert registration["projection026_arrival_window_ms"] == [0.0, 24.6]
    assert "after simulation" in registration["boundary"]
    assert "No original-SMART claim" in registration["boundary"]


def test_projection026_focal_resource_pair_confirms_onset_without_parameter() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection026-focal-resource-pair-715.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection026-focal-resource-assessment-716.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["persistent_projection_delays_ms"] == {}
    focal = result["focal_source_events"]
    assert focal["source_indices"] == [39, 41]
    assert focal["intact"]["time_from_mismatch_ms"] == pytest.approx(23.43)
    assert focal["disconnected_control"][
        "time_from_mismatch_ms"
    ] == pytest.approx(23.44)
    assert focal["intact"][
        "threshold_order0_transmitter_pre_reset"
    ] == pytest.approx(0.046008006379109824)
    assert focal["disconnected_control"][
        "threshold_order0_transmitter_pre_reset"
    ] == pytest.approx(0.046031855880830336)
    assert focal["depletion_epsilon"] == pytest.approx(1.0)
    assert focal["derived_transmitter_immediately_after_reset_both_arms"] == 0.0
    delivery = result["focal_projection026_delivery"]
    assert delivery["intact_arrival_from_mismatch_ms"] == pytest.approx(24.43)
    assert delivery[
        "disconnected_control_arrival_from_mismatch_ms"
    ] == pytest.approx(24.44)
    verdict = assessment["assessment"]
    assert verdict["complete_paired_focal_transmitter_values_preserved"]
    assert verdict["source_phase_resource_onset_mechanism_confirmed"]
    assert not verdict["projection026_parameter_fault_detected"]
    assert not verdict["parameter_selected"]
    assert not verdict["official_figure10_reset_reproduced"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_promoted"]


def test_layer6i_output_depletion_audit_is_source_corrected_and_registered() -> None:
    audit_path = (
        ROOT
        / "docs/validation-results/figure10-layer6i-output-depletion-source-audit-717.yaml"
    )
    registration_path = (
        ROOT
        / "docs/validation-results/figure10-layer6i-output-depletion-registration-718.yaml"
    )
    audit = yaml.safe_load(audit_path.read_text())
    registration = yaml.safe_load(registration_path.read_text())

    for path_key, hash_key in (
        ("source_audit", "source_audit_sha256"),
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("script", "script_sha256"),
        ("source_result", "source_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["runtime_sha256"] == (
        "ebdf48f0138803ab50b1dec2ef87817b4c537e4da490a088b9df6bdcb9d119a1"
    )
    executable = audit["executable_source"]
    assert executable["layer6i"]["depletion_enabled"]
    assert executable["layer6i"]["shared_source_resource_projections"] == [
        "modeldb112923.projection.026",
        "modeldb112923.projection.038",
    ]
    assert not executable["layer4_inhibitory"]["depletion_enabled"]
    assert executable["projection036"]["transmitter_multiplier"] == 1
    assert not audit["correction"]["model_or_parameter_changed"]
    assert registration["output_mode"] == "projection026_source_resource_bounded"
    assert registration["layer6i_source_resource_indices"] == [
        31, 38, 39, 40, 41, 42, 49
    ]
    assert registration["layer6i_source_resource_window_ms"] == [0.0, 80.0]
    assert registration["record_projection036_arrival_target_indices"] == [31]
    assert registration["projection036_arrival_window_ms"] == [75.2, 77.4]
    assert "No transmitter alternative" in registration["boundary"]


def test_layer6i_output_depletion_pair_localizes_delivery_schedule_conflict() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer6i-output-depletion-pair-719.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-output-depletion-assessment-720.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["persistent_projection_delays_ms"] == {}
    assert result["exact_source_identity"]["layer6i_post_events_intact_control"] == [
        153,
        76,
    ]
    intact = result["first_source_events"]["intact"]
    alternatives = intact["previously_inactive_alternatives"]
    winners = intact["winner_aligned"]
    assert [row["source_index"] for row in alternatives] == [31, 49]
    assert [row["transmitter_pre_reset"] for row in alternatives] == [1.0, 1.0]
    assert max(row["transmitter_pre_reset"] for row in winners) < 0.015
    assert result["first_source_events"]["disconnected_control"][
        "previously_inactive_alternatives"
    ] == []
    schedule = result["derived_resource_schedule"]
    assert schedule["immediate_post_event_transmitter_epsilon1"] == 0.0
    assert schedule["no_intervening_event_recovery_at_1ms_delivery"] == pytest.approx(
        0.0024968776025399153
    )
    assert schedule["recovery_at_projection026_waveform_peak"] == pytest.approx(
        0.0033739155249272734
    )
    assert schedule["recovery_at_projection038_waveform_peak"] == pytest.approx(
        0.007471945180861583
    )
    verdict = assessment["assessment"]
    assert verdict["previously_inactive_pre_event_transmitter_advantage"]
    assert not verdict["pre_event_advantage_preserved_at_delayed_delivery"]
    assert verdict["continuous_schedule_erases_emitted_event_advantage"]
    assert verdict["layer6i_habituation_mechanism_conflict_localized"]
    assert not verdict["parameter_selected"]
    assert not verdict["official_figure10_reset_reproduced"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_promoted"]


def test_layer6i_emission_resource_cross_is_bounded_and_preregistered() -> None:
    audit_path = (
        ROOT
        / "docs/validation-results/figure10-transmitter-scheduling-source-audit-721.yaml"
    )
    implementation_path = (
        ROOT
        / "docs/validation-results/figure10-layer6i-emission-resource-implementation-722.yaml"
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-emission-resource-registration-723.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("source_audit", "source_audit_sha256"),
        ("implementation", "implementation_sha256"),
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("script", "script_sha256"),
        ("prior_result", "prior_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    audit = yaml.safe_load(audit_path.read_text())
    implementation = yaml.safe_load(implementation_path.read_text())
    assert not audit["unresolved_ordering"]["exact_legacy_source_body_available"]
    assert audit["authorized_discriminator"]["affected_projections"] == [
        "modeldb112923.projection.026",
        "modeldb112923.projection.038",
    ]
    assert implementation["implementation"]["default"] == (
        "continuous_current_resource"
    )
    assert implementation["implementation"]["candidate"] == (
        "pre_depletion_emission_snapshot"
    )
    assert registration["runtime_overrides"] == {
        "layer6i_output_transmitter_gate_convention": (
            "pre_depletion_emission_snapshot"
        )
    }
    assert registration["runtime_fingerprint"] == (
        "fbcc1dc2ac7db8442ce1ff17b71a1d04582c32e817370c700c69ec3caa9fbe4e"
    )
    assert registration["candidate_scope"][
        "continuous_default_retained_for_all_other_projections"
    ]
    assert "one intact/control pair" in registration["decision_rule"]
    assert "would not" in registration["boundary"]


def test_layer6i_emission_resource_pair_restores_only_winner_suppression() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer6i-emission-resource-pair-724.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-emission-resource-assessment-725.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["persistent_projection_delays_ms"] == {}
    assert result["intact"]["layer6i_mismatch_active_count"] == 81
    assert result["intact"]["layer4_inhibitory_active_count"] == 81
    assert result["reset_assessment"]["intact_winner_post_spikes"] == 69
    assert result["reset_assessment"]["control_winner_post_spikes"] == 78
    assert result["reset_assessment"]["intact_released_alternatives"] == 0
    assert result["reset_assessment"]["control_released_alternatives"] == 0
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": True,
        "alternative_release": False,
    }
    verdict = assessment["assessment"]
    assert verdict["scheduling_convention_causally_improves_winner_suppression"]
    assert not verdict["scheduling_convention_sufficient_for_reset"]
    assert not verdict["scheduling_convention_selected"]
    assert verdict["scheduling_convention_rejected_as_standalone_endpoint"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_promoted"]


def test_layer6i_emission_target_balance_audit_is_passive_and_pinned() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-emission-target-balance-registration-726.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("implementation", "implementation_sha256"),
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("script", "script_sha256"),
        ("source_registration", "source_registration_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["runtime_overrides"] == {
        "layer6i_output_transmitter_gate_convention": (
            "pre_depletion_emission_snapshot"
        )
    }
    assert registration["record_layer4_target_balance_indices"] == [
        31,
        38,
        39,
        40,
        41,
        42,
        49,
    ]
    assert registration["output_mode"] == "layer4_complete_target_bounded"
    assert "cannot reopen or select" in registration["boundary"]


def test_layer6i_emission_target_balance_localizes_but_does_not_select() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer6i-emission-target-balance-pair-727.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-emission-target-balance-assessment-728.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": True,
        "alternative_release": False,
    }
    for arm in ("intact", "disconnected_control"):
        assert result[arm]["layer4_target_balance_bin_edges_ms"] == [
            float(value) for value in range(0, 201, 10)
        ]
        assert set(result[arm]["layer4_target_balance_series"]) == {
            "31",
            "38",
            "39",
            "40",
            "41",
            "42",
            "49",
        }
    verdict = assessment["assessment"]
    assert verdict["exact_source_identity_reproduced"]
    assert verdict["direct_projection038_reaches_alternatives"]
    assert verdict["first_direct_reset_pulse_opposed_by_extra_projection036_inhibition"]
    assert not verdict["registered_voltage_readout_present"]
    assert not verdict["complete_causal_order_resolved"]
    assert not verdict["parameter_selected"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_promoted"]


def test_layer6i_emission_target_voltage_correction_is_preregistered() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-emission-target-voltage-registration-729.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("source_result", "source_result_sha256"),
        ("implementation", "implementation_sha256"),
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("script", "script_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["record_layer4_target_timing_indices"] == [31, 49]
    assert registration["layer4_target_timing_window_ms"] == [50.0, 90.0]
    assert registration["layer4_target_timing_bin_width_ms"] == 1.0
    assert registration["output_mode"] == "layer4_target_timing_bounded"
    assert "changes no executable model quantity" in registration[
        "instrumentation_correction"
    ]
    assert "remains rejected and default-off" in registration["boundary"]


def test_layer6i_emission_target_voltage_confirms_membrane_state_deficit() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-layer6i-emission-target-voltage-pair-730.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-layer6i-emission-target-voltage-assessment-731.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": True,
        "alternative_release": False,
    }
    verdict = assessment["assessment"]
    assert verdict["exact_source_identity_reproduced"]
    assert verdict["direct_projection038_gate_onset_ms"] == pytest.approx(61.88)
    assert verdict["first_direct_pulse_creates_persistent_voltage_deficit"]
    assert verdict["intact_voltage_overtakes_control_after_deficit"]
    assert verdict["intact_voltage_overtake_first_bin_ms"] == [74.0, 75.0]
    assert verdict["intact_peak_soma_voltage_mV_in_window"] == pytest.approx(
        -58.703029330588265
    )
    assert verdict["intact_alternative_spikes_in_window"] == 0
    assert not verdict["intact_projection037_active_in_window"]
    assert not verdict["parameter_selected"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_promoted"]


def test_figure10_static_comparator_lifecycle_audit_is_source_pinned() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-comparator-lifecycle-source-audit-732.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("source_result", "source_result_sha256"),
    ):
        assert hashlib.sha256((ROOT / audit[path_key]).read_bytes()).hexdigest() == audit[
            hash_key
        ]
    recovered = audit["recovered_executable"]
    assert hashlib.sha256((ROOT / recovered["path"]).read_bytes()).hexdigest() == recovered[
        "sha256"
    ]
    comparator = audit["reconstructed_comparator"]
    assert hashlib.sha256(
        (ROOT / comparator["registration"]).read_bytes()
    ).hexdigest() == comparator["registration_sha256"]
    harness = audit["current_figure10_harness"]
    assert hashlib.sha256((ROOT / harness["path"]).read_bytes()).hexdigest() == harness[
        "sha256"
    ]
    assert hashlib.sha256(
        (ROOT / harness["protocol_path"]).read_bytes()
    ).hexdigest() == harness["protocol_sha256"]
    assert harness["lifecycle"] == {
        "gain_vector_computed_once": True,
        "same_gain_vector_applied_to_match_and_mismatch": True,
        "gain_vector_released_after_reset": False,
        "gain_vector_depends_on_reset_pathway": False,
    }
    consequence = harness["sensory_consequence"]
    assert consequence["vertical_mismatch_active_pixels"] == [22, 31, 40, 49, 58]
    assert consequence["vertical_mismatch_directly_driven_with_top5"] == [40]
    assert consequence["vertical_pixels_held_at_zero_for_entire_mismatch"] == [
        22,
        31,
        49,
        58,
    ]
    decision = audit["source_coherence_decision"]
    assert not decision["static_top5_figure10_use_source_coherent"]
    assert not decision["fit_dynamic_release_time_authorized"]
    assert decision["parameter_free_endpoint"] == "comparator-free Figure-10 input"


def test_figure10_full_bottom_up_profile_is_parameter_free_identity_input() -> None:
    profile = yaml.safe_load(
        (
            ROOT / "configs/calibration/figure10_full_bottom_up_emission_v1.yaml"
        ).read_text()
    )
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-full-bottom-up-emission-implementation-733.yaml"
        ).read_text()
    )

    assert profile["comparator"] == {
        "transform": "identity_all_ones_via_top_k_all",
        "source_index": 40,
        "target_count": 81,
        "implementation_note": profile["comparator"]["implementation_note"],
    }
    assert implementation["implementation"]["code_change_required"] is False
    assert implementation["implementation"]["expected_vector_unique_values"] == [1.0]
    assert implementation["implementation"]["equations_changed"] is False
    assert implementation["implementation"]["model_parameter_changed"] is False
    assert "not a new comparator" in implementation["boundary"]


def test_figure10_full_bottom_up_emission_cross_is_preregistered() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-full-bottom-up-emission-registration-734.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("source_audit", "source_audit_sha256"),
        ("implementation", "implementation_sha256"),
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("script", "script_sha256"),
        ("prior_result", "prior_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["runtime_overrides"] == {
        "layer6i_output_transmitter_gate_convention": (
            "pre_depletion_emission_snapshot"
        )
    }
    assert registration["record_layer4_target_balance_indices"] == [
        22,
        31,
        40,
        49,
        58,
    ]
    assert registration["protocol"]["comparator"]["target_count"] == 81
    assert registration["required_source_identity"][
        "mismatch_event_counts_pinned_to_prior_result"
    ] is False
    assert "projection 035" in registration["input_delivery_gate"]
    assert "more of" in registration["decision_rule"]
    assert "one parameter-free source-coherence protocol cross" in registration[
        "boundary"
    ]


def test_figure10_full_bottom_up_cross_delivers_input_but_fails_release() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-full-bottom-up-emission-pair-735.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-full-bottom-up-emission-assessment-736.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["intact"]["layer4_pre_events"] == 43
    assert result["disconnected_control"]["layer4_pre_events"] == 43
    assert result["reset_assessment"] == {
        "pre_reset_winner_index": 38,
        "pre_reset_winner_indices": [38, 39, 40, 41, 42],
        "pre_reset_winner_spikes": 43,
        "intact_winner_post_spikes": 42,
        "control_winner_post_spikes": 45,
        "intact_released_alternatives": 4,
        "control_released_alternatives": 4,
        "intact_nonspecific_spikes": 13,
        "intact_layer5_spikes": 235,
        "intact_layer6i_spikes": 199,
    }
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "winner_suppression": True,
        "alternative_release": False,
    }
    assert all(
        value > 0
        for arm in assessment["input_delivery"][
            "projection035_mismatch_integral_pA_ms"
        ].values()
        for value in arm.values()
    )
    assert assessment["causal_output"]["intact"]["alternative_events"] == 60
    assert assessment["causal_output"]["disconnected_control"][
        "alternative_events"
    ] == 64
    verdict = assessment["assessment"]
    assert verdict["full_five_pixel_vertical_input_delivered_both_arms"]
    assert verdict["winner_suppression_pass"]
    assert not verdict["alternative_release_pass"]
    assert not verdict["source_coherence_endpoint_pass"]
    assert not verdict["emission_snapshot_scheduling_selected"]
    assert not verdict["parameter_selected"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_promoted"]


def test_figure10_search_cycle_source_audit_pins_missing_scheduler_boundary() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-search-cycle-lifecycle-source-audit-737.yaml"
        ).read_text()
    )

    assert hashlib.sha256(
        (ROOT / audit["authorization"]).read_bytes()
    ).hexdigest() == audit["authorization_sha256"]
    assert hashlib.sha256(
        (ROOT / audit["primary_paper"]["path"]).read_bytes()
    ).hexdigest() == audit["primary_paper"]["sha256"]
    for item in audit["kinness_reports"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]
    recovered = audit["recovered_executable"]
    assert hashlib.sha256((ROOT / recovered["path"]).read_bytes()).hexdigest() == recovered[
        "sha256"
    ]
    for endpoint in audit["completed_bracketing_endpoints"].values():
        if isinstance(endpoint, dict) and "result" in endpoint:
            assert hashlib.sha256(
                (ROOT / endpoint["result"]).read_bytes()
            ).hexdigest() == endpoint["result_sha256"]
    marker = audit["independent_reset_marker"]
    assert marker["direct_projection038_gate_threshold"] == pytest.approx(0.1)
    assert marker["direct_projection038_first_threshold_crossing_ms_from_mismatch"] == pytest.approx(
        61.88
    )
    reconstruction = audit["authorized_reconstruction"]
    assert reconstruction["yoking"]["identical_transition_time_both_arms"]
    assert reconstruction["at_marker"]["action"] == (
        "replace relay gain vector with identity all ones"
    )
    assert "ties or silence fail" in audit["decision_rule"]


def test_figure10_search_cycle_pair_is_preregistered_and_hash_pinned() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-search-cycle-lifecycle-registration-739.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("source_audit", "source_audit_sha256"),
        ("implementation", "implementation_sha256"),
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("script", "script_sha256"),
        ("prior_result", "prior_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["runtime_fingerprint"] == (
        "fbcc1dc2ac7db8442ce1ff17b71a1d04582c32e817370c700c69ec3caa9fbe4e"
    )
    assert registration["protocol"]["release_after_mismatch_ms"] == pytest.approx(
        61.88
    )
    assert registration["protocol"]["release_yoked_across_arms"]
    assert registration["required_source_identity"][
        "alternatives_before_release_intact_control"
    ] == [0, 0]
    assert "strictly before" in registration["decision_rule"]
    assert "no second marker" in registration["boundary"]


def test_figure10_search_cycle_result_closes_wrong_sign_latency_endpoint() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-search-cycle-lifecycle-pair-740.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-search-cycle-lifecycle-assessment-741.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["reset_gates"] == {
        "pre_reset_winner": True,
        "reset_chain": True,
        "alternatives_quiet_before_release": True,
        "full_input_delivery": True,
        "winner_suppression": True,
        "alternative_latency": False,
    }
    assert result["intact"]["alternative_events_before_release"] == 0
    assert result["disconnected_control"]["alternative_events_before_release"] == 0
    assert result["intact"]["first_alternative_event_ms"] == pytest.approx(98.38)
    assert result["disconnected_control"][
        "first_alternative_event_ms"
    ] == pytest.approx(85.91)
    assert result["intact"]["winner_post_events"] == 49
    assert result["disconnected_control"]["winner_post_events"] == 59
    assert result["intact"]["alternative_events"] == 36
    assert result["disconnected_control"]["alternative_events"] == 36
    verdict = assessment["assessment"]
    assert verdict["winner_suppression_pass"]
    assert not verdict["alternative_latency_pass"]
    assert verdict["event_locked_lifecycle_endpoint_closed"]
    assert not verdict["another_marker_or_score_authorized"]
    assert not verdict["emission_snapshot_scheduling_selected"]
    assert not verdict["parameter_selected"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_promoted"]


def test_figure10_search_cycle_balance_audit_is_preregistered_and_passive() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-search-cycle-balance-registration-743.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("implementation", "implementation_sha256"),
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("script", "script_sha256"),
        ("prior_result", "prior_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["runtime_fingerprint"] == (
        "fbcc1dc2ac7db8442ce1ff17b71a1d04582c32e817370c700c69ec3caa9fbe4e"
    )
    assert registration["audit_window_ms"] == [61.0, 110.0]
    assert registration["audit_bin_width_ms"] == pytest.approx(1.0)
    assert registration["monitored_layer4_targets"] == [
        22,
        31,
        38,
        39,
        40,
        41,
        42,
        49,
        58,
    ]
    assert registration["source_resolved_arrival_targets"] == [31, 40]
    assert registration["required_source_identity"][
        "intact_control_first_alternative_event_ms"
    ] == pytest.approx([98.38, 85.91])
    assert "localization only" in registration["decision_rule"]
    assert "No model quantity" in registration["boundary"]


def test_figure10_search_cycle_balance_localizes_indirect_inhibition() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure10-search-cycle-balance-pair-744.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-search-cycle-balance-assessment-745.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["intact"]["pre_layer4_events"] == 43
    assert result["disconnected_control"]["pre_layer4_events"] == 43
    assert result["intact"]["winner_post_events"] == 49
    assert result["disconnected_control"]["winner_post_events"] == 59
    assert result["intact"]["alternative_events"] == 36
    assert result["disconnected_control"]["alternative_events"] == 36
    assert result["intact"]["first_alternative_event_ms"] == pytest.approx(98.38)
    assert result["disconnected_control"][
        "first_alternative_event_ms"
    ] == pytest.approx(85.91)
    direct = assessment["direct_projection038_spatial_bias_61_69_ms"]
    assert direct["alternative_to_winner_ratio"] == pytest.approx(38.2597961)
    balance = assessment["early_61_69_ms_balance_at_alternative_31"]
    assert balance["total_extra_excitation_intact_minus_control_pA_ms"] > 0
    assert balance[
        "extra_projection036_inhibitory_current_intact_minus_control_pA_ms"
    ] < 0
    assert balance["net_recorded_current_difference_intact_minus_control_pA_ms"] < 0
    focal = assessment["focal_alternative_31"]
    assert focal["projection036_intact_lead_ms"] == pytest.approx(5.55)
    assert focal["recurrent_excitation_delay_ms"] == pytest.approx(14.44)
    verdict = assessment["assessment"]
    assert verdict["direct_projection038_spatial_bias_present"]
    assert verdict["indirect_off_surround_balance_failure_localized"]
    assert not verdict["parameter_selected"]
    assert not verdict["official_figure10_reset_reproduced"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_frozen"]


def test_projection036_spread_audit_finds_one_bounded_source_ambiguity() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-spread-source-audit-746.yaml"
        ).read_text()
    )

    assert hashlib.sha256(
        (ROOT / audit["authorization"]).read_bytes()
    ).hexdigest() == audit["authorization_sha256"]
    for source in ("primary_paper", "kinness_framework", "modeldb_executable", "supplement"):
        assert hashlib.sha256((ROOT / audit[source]["path"]).read_bytes()).hexdigest() == audit[
            source
        ]["sha256"]
    parity = audit["compiled_brian2_parity"]
    assert parity["source_density_weight_area_translation_exact"]
    assert parity["source_transmitter_shared_by_026_and_038"]
    assert not parity["current_or_area_transcription_error_found"]
    ambiguity = audit["spread_ambiguity"]
    assert ambiguity["active_standard_deviation"]["incoming_edges_per_target"] == 80
    assert ambiguity["bounded_variance_interpretation"][
        "incoming_edges_per_target"
    ] == 68
    assert ambiguity["bounded_variance_interpretation"][
        "summed_incoming_weight_reduction_fraction"
    ] == pytest.approx(0.3559821946)
    assert "exactly one" in audit["decision"]
    assert "failure closes" in audit["stopping_rule"].lower()
    assert "ring shape or radius is not reopened" in audit["prior_closed_families"][
        "projection036_ring_geometry"
    ]


def test_projection036_spread_cross_is_preregistered_and_hash_pinned() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-spread-registration-748.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("source_audit", "source_audit_sha256"),
        ("implementation", "implementation_sha256"),
        ("profile", "profile_sha256"),
        ("harness", "harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("script", "script_sha256"),
        ("prior_result", "prior_result_sha256"),
        ("prior_assessment", "prior_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["runtime_fingerprint"] == (
        "fbcc1dc2ac7db8442ce1ff17b71a1d04582c32e817370c700c69ec3caa9fbe4e"
    )
    cross = registration["projection036_cross"]
    assert cross["candidate_interpretation"] == "variance"
    assert cross["ring_convention"] == "center_excluded_gaussian"
    assert cross["nonzero_edges"] == 5508
    assert cross["incoming_edges_per_target"] == 68
    assert cross["incoming_weight_sum_per_target"] == pytest.approx(
        16.83946769281913
    )
    assert registration["protocol"]["release_after_mismatch_ms"] == pytest.approx(
        61.88
    )
    assert "Ties or silence fail" in registration["decision_rule"]
    assert "failure closes" in registration["stopping_rule"].lower()


def test_projection036_spread_cross_reproduces_registered_search_cycle() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection036-spread-pair-749.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-spread-assessment-750.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    for arm in (result["intact"], result["disconnected_control"]):
        assert arm["projection036_spread_convention"] == "variance"
        assert arm["projection036_nonzero_edges"] == 5508
        assert arm["projection036_incoming_weight_sum"] == pytest.approx(
            16.83946769281913
        )
        assert arm["pre_layer4_events"] == 43
        assert arm["pre_layer4_active_indices"] == [38, 39, 40, 41, 42]
        assert arm["alternative_events_before_release"] == 0
        assert arm["alternative_active_indices"] == [22, 31, 49, 58]
    assert result["intact"]["winner_post_events"] == 47
    assert result["disconnected_control"]["winner_post_events"] == 54
    assert result["intact"]["first_alternative_event_ms"] == pytest.approx(79.25)
    assert result["disconnected_control"][
        "first_alternative_event_ms"
    ] == pytest.approx(87.11)
    assert all(result["reset_gates"].values())
    assert result["reproduced_search_cycle"]
    verdict = assessment["assessment"]
    assert verdict["all_preregistered_search_cycle_gates_pass"]
    assert verdict["projection036_variance_endpoint_selected"]
    assert verdict["mixed_source_calibrated_endpoint"]
    assert verdict["official_figure10_first_order_search_cycle_reproduced"]
    assert not verdict["exact_kinness_recovery_proven"]
    assert not verdict["figure7_exact_rate_reproduced"]
    assert not verdict["official_gamma_beta_reproduced"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_frozen"]


def test_projection036_figure7_consistency_is_preregistered_and_hash_pinned() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-projection036-spread-consistency-registration-752.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("profile", "profile_sha256"),
        ("implementation", "implementation_sha256"),
        ("script", "script_sha256"),
        ("projection036_harness", "projection036_harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("training_profile", "training_profile_sha256"),
        ("selected_figure10_result", "selected_figure10_result_sha256"),
        ("selected_figure10_assessment", "selected_figure10_assessment_sha256"),
        ("prior_figure7_result", "prior_figure7_result_sha256"),
        ("prior_figure7_assessment", "prior_figure7_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["runtime_fingerprint"] == (
        "fbcc1dc2ac7db8442ce1ff17b71a1d04582c32e817370c700c69ec3caa9fbe4e"
    )
    endpoint = registration["fixed_endpoint"]
    assert endpoint["projection036_spread_convention"] == "variance"
    assert endpoint["projection036_nonzero_edges"] == 5508
    assert endpoint["projection036_incoming_weight_sum_per_target"] == pytest.approx(
        16.83946769281913
    )
    assert registration["required_gates"]["match_nonspecific_events"] == 4
    assert registration["required_gates"]["mismatch_nonspecific_events"] == 7
    assert "exactly one" in registration["decision_rule"]
    assert "Do not repeat or adjust" in registration["stopping_rule"]
    assert "not a calibration screen" in registration["boundary"]


def test_projection036_figure7_consistency_fails_only_exact_rate_gates() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure7-projection036-spread-consistency-753.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure7-projection036-spread-consistency-assessment-754.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["projection036_topology_gates"].values())
    assert all(result["figure6_gates"].values())
    gates = result["figure7_gates"]
    assert gates["match_relay_active_indices"]
    assert gates["match_relay_events"]
    assert gates["mismatch_relay_allowed_indices"]
    assert gates["match_more_active_relay_cells"]
    assert gates["match_more_trn_events"]
    assert gates["match_more_trn_to_nonspecific_gaba"]
    assert not gates["match_nonspecific_events"]
    assert not gates["mismatch_nonspecific_events"]
    assert result["event_identity"]["match"]["nonspecific_events"] == 3
    assert result["event_identity"]["mismatch"]["nonspecific_events"] == 6
    assert not result["figure7_consistency_reproduced"]
    verdict = assessment["assessment"]
    assert verdict["failed_gate_count"] == 2
    assert not verdict["figure14_holdout_authorized"]
    assert not verdict["repeat_or_rate_tuning_authorized"]
    assert not verdict["joint_figure6_figure7_figure10_endpoint_exists"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_frozen"]


def test_projection025_source_control_audit_closes_scale8_as_classic_value() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-7-10-projection025-source-control-audit-755.yaml"
        ).read_text()
    )

    for section in (
        "modeldb_source",
        "diagnostic_bracket",
        "connected_scale8_endpoint",
        "selected_figure10_endpoint",
        "figure7_consistency_failure",
    ):
        item = audit[section]
        if "path" in item:
            path_key = "path"
        elif "result" in item and "result_sha256" in item:
            assert hashlib.sha256((ROOT / item["result"]).read_bytes()).hexdigest() == item[
                "result_sha256"
            ]
            path_key = "assessment"
        else:
            path_key = "assessment"
        assert hashlib.sha256((ROOT / item[path_key]).read_bytes()).hexdigest() == item[
            f"{path_key}_sha256" if path_key != "path" else "sha256"
        ]
    conclusion = audit["provenance_conclusion"]
    assert not conclusion["projection025_scale8_admissible_for_classic_baseline"]
    assert conclusion["projection025_scale1_is_released_source_value"]
    assert not conclusion["projection025_continuous_interpolation_authorized"]
    assert conclusion["joint_source_control_cross_authorized"]
    assert "exactly one" in audit["decision"]
    assert "do not interpolate" in audit["stopping_rule"].lower()


def test_projection025_joint_source_control_is_preregistered_and_hash_pinned() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-7-10-projection025-source-control-registration-757.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("implementation", "implementation_sha256"),
        ("script", "script_sha256"),
        ("projection036_harness", "projection036_harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
        ("training_profile", "training_profile_sha256"),
        ("prior_figure7_assessment", "prior_figure7_assessment_sha256"),
        ("selected_figure10_assessment", "selected_figure10_assessment_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["fixed_change"] == {
        "projection_id": "modeldb112923.projection.025",
        "prior_diagnostic_scale": 8.0,
        "candidate_source_scale": 1.0,
        "implementation": "absence from persistent projection scale map",
    }
    assert registration["required_gates"] == {
        "projection025_source_scale": True,
        "projection036_topology_identity_all_five_builds": True,
        "all_fresh_figure6_gates": True,
        "all_fixed_figure7_gates": True,
        "all_fixed_figure10_search_cycle_gates": True,
    }
    assert "exactly one" in registration["decision_rule"]
    assert "Do not repeat" in registration["stopping_rule"]


def test_projection025_source_control_recovery_changes_serialization_only() -> None:
    failure = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-7-10-projection025-source-control-758.yaml"
        ).read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-7-10-projection025-source-control-recovery-registration-760.yaml"
        ).read_text()
    )

    assert failure["execution"]["stdout_bytes"] == 0
    assert failure["execution"]["stdout_sha256"] == hashlib.sha256(b"").hexdigest()
    assert failure["failure"]["stage"].startswith("yaml.safe_dump")
    assert not failure["scientific_assessment"]["complete_structured_result_emitted"]
    assert failure["scientific_assessment"]["candidate_passed"] is None
    assert failure["scientific_assessment"]["candidate_failed"] is None
    for path_key, hash_key in (
        ("original_registration", "original_registration_sha256"),
        ("failed_execution", "failed_execution_sha256"),
        ("recovery_implementation", "recovery_implementation_sha256"),
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("script", "script_sha256"),
        ("projection036_harness", "projection036_harness_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    recovery = registration["recovery_contract"]
    assert recovery["only_change"] == "numpy.bool_ to bool serialization cast"
    assert recovery["model_protocol_seed_and_gates_identical"]
    assert recovery["execute_complete_simulation_again"]
    assert not recovery["failed_execution_metrics_reused"]
    assert "complete joint source-control simulation once" in registration[
        "decision_rule"
    ]


def test_projection025_source_control_has_no_joint_first_order_survivor() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure6-7-10-projection025-source-control-recovery-761.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-7-10-projection025-source-control-assessment-762.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert result["projection025_runtime_scale"] == 1.0
    assert all(result["topology_and_source_gates"].values())
    assert all(result["figure6_gates"].values())
    assert result["figure7_event_identity"]["match"]["nonspecific_events"] == 3
    assert result["figure7_event_identity"]["mismatch"]["nonspecific_events"] == 6
    assert not result["figure7_gates"]["match_nonspecific_events"]
    assert not result["figure7_gates"]["mismatch_nonspecific_events"]
    assert result["figure10_intact"]["winner_post_events"] == 59
    assert result["figure10_disconnected_control"]["winner_post_events"] == 59
    assert not result["figure10_gates"]["winner_suppression"]
    assert result["figure10_gates"]["alternative_latency"]
    assert not result["joint_first_order_gates_pass"]
    verdict = assessment["assessment"]
    assert not verdict["current_constrained_family_has_joint_survivor"]
    assert not verdict["source_control_endpoint_selected"]
    assert not verdict["projection025_scale8_selected_or_recovered"]
    assert not verdict["figure14_holdout_authorized"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_frozen"]


def test_projection025_source_to_brian2_conductance_parity_is_exact_and_closed() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-7-10-projection025-conductance-parity-audit-763.yaml"
        ).read_text()
    )

    for source in (
        audit["authorization"],
        audit["primary_sources"]["smart_nml"],
        audit["primary_sources"]["kinness_framework"],
        audit["primary_sources"]["derived_catalog"],
        *audit["implementation_sources"].values(),
    ):
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source[
            "sha256"
        ]
    compiled = audit["read_only_compilation"]
    assert compiled["lateral_area_cm2"] == pytest.approx(
        math.pi * compiled["target_diameter_cm"] * compiled["target_length_cm"]
    )
    assert compiled["expected_maximal_conductance_nS"] == pytest.approx(
        compiled["density_mS_cm2"] * compiled["lateral_area_cm2"] * 1e6
    )
    assert compiled["compiled_maximal_conductance_nS"] == pytest.approx(
        compiled["expected_maximal_conductance_nS"], rel=0.0, abs=0.0
    )
    assert compiled["compiled_edges"] == 81
    assert compiled["inputs_per_target"] == 1
    assert compiled["edge_weight_min"] == compiled["edge_weight_max"] == 1.0
    assert compiled["incoming_weight_sum_target40"] == 1.0
    assert not audit["parity_gates"]["unaccounted_conductance_factor_detected"]
    assert all(
        value
        for key, value in audit["parity_gates"].items()
        if key != "unaccounted_conductance_factor_detected"
    )
    verdict = audit["assessment"]
    assert verdict["projection025_source_translation_exact"]
    assert not verdict["projection025_source_ambiguity_remaining"]
    assert verdict["projection025_family_closed"]
    assert not verdict["projection025_scale8_source_supported"]
    assert not verdict["projection025_interpolation_authorized"]
    assert not verdict["current_constrained_family_has_joint_survivor"]
    assert not verdict["source_level_correction_available_from_projection025"]
    assert not verdict["parameter_selected"]
    assert not verdict["figure14_holdout_authorized"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_frozen"]


def test_sanndra_predecessor_integration_cross_is_bounded_and_rejected() -> None:
    audit_path = (
        ROOT
        / "docs/validation-results/sanndra-predecessor-integration-audit-764.yaml"
    )
    audit = yaml.safe_load(audit_path.read_text())
    registration_path = (
        ROOT
        / "docs/validation-results/figure6-7-10-sanndra-scalar-rk4-registration-765.yaml"
    )
    registration = yaml.safe_load(registration_path.read_text())
    result_path = (
        ROOT
        / "docs/validation-results/figure6-7-10-sanndra-scalar-rk4-766.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure6-7-10-sanndra-scalar-rk4-assessment-767.yaml"
        ).read_text()
    )

    assert audit["primary_predecessor_source"]["archive_sha256"] == (
        "ee6f0700280ea3df8ab1be0f2b45a03a351842ba4390c275df0ec45f64f466a3"
    )
    assert audit["primary_predecessor_source"]["release_scope"].startswith(
        "preserved 2001 predecessor"
    )
    assert not audit["inference_boundary"]["exact_smart_era_sanndra_source_recovered"]
    assert not audit["inference_boundary"]["exact_smart_era_scalar_update_proven"]
    assert audit["inference_boundary"]["source_adjacent_candidate_justified"]
    assert not audit["bounded_comparator"]["default_changed"]
    assert audit["unit_gates"]["scalar_linear_equation_matches_brian_rk4"]
    assert audit["unit_gates"]["coupled_two_state_probe_freezes_other_state"]
    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("profile", "profile_sha256"),
        ("script", "script_sha256"),
        ("runtime", "runtime_sha256"),
        ("synapses", "synapses_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    refactor = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/sanndra-scalar-rk4-registration-refactor-768.yaml"
        ).read_text()
    )
    assert refactor["executed_implementation"]["cell_runtime_sha256"] == registration[
        "cell_runtime_sha256"
    ]
    assert refactor["executed_implementation"][
        "integration_runtime_sha256"
    ] == registration["integration_runtime_sha256"]
    for item in refactor["current_equivalent_implementation"].values():
        if isinstance(item, dict) and "path" in item:
            assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item[
                "sha256"
            ]
    equivalence = refactor["equivalence_gates"]
    assert equivalence["scalar_update_function_unchanged"]
    assert equivalence["wrapper_delegates_directly_to_scalar_update_function"]
    assert equivalence[
        "registered_wrapper_and_executed_callable_generate_identical_abstract_code"
    ]
    assert equivalence["method_name_unchanged"] == "sanndra_scalar_rk4"
    assert equivalence["runtime_fingerprint_unchanged"] == registration[
        "runtime_fingerprint"
    ]
    assert not equivalence["model_parameters_protocols_and_random_seed_changed"]
    assert not equivalence["simulation_repeated"]
    assert not refactor["scientific_assessment"]["prior_result_changed"]
    assert not refactor["scientific_assessment"]["integration_endpoint_selected"]
    assert registration["fixed_difference_from_source_control_761"][
        "all_model_parameters_unchanged"
    ]
    assert registration["fixed_difference_from_source_control_761"][
        "projection025_scale"
    ] == 1.0
    assert hashlib.sha256(registration_path.read_bytes()).hexdigest() == assessment[
        "registration_sha256"
    ]
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["figure7"]["match"]["nonspecific_events"] == 5
    assert result["figure7"]["mismatch"]["nonspecific_events"] == 6
    assert not result["figure7"]["gates"]["match_more_trn_to_nonspecific_gaba"]
    assert result["figure10"]["intact"]["winner_post_events"] == 50
    assert result["figure10"]["disconnected_control"]["winner_post_events"] == 48
    assert not result["figure10"]["gates"]["winner_suppression"]
    assert not result["figure10"]["gates"]["alternative_latency"]
    assert not result["joint_first_order_gates_pass"]
    verdict = assessment["assessment"]
    assert not verdict["integration_endpoint_selected"]
    assert not verdict["integration_interpolation_authorized"]
    assert not verdict["original_smart_reproduced"]
    assert not verdict["baseline_frozen"]


def test_projection036_source_arrival_result_localizes_phase_not_topology() -> None:
    result_path = (
        ROOT
        / "docs/validation-results/figure10-projection036-source-arrival-pair-686.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure10-projection036-source-arrival-assessment-687.yaml"
        ).read_text()
    )

    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["persistent_projection_delays_ms"] == {}
    assert result["arrival_contract"]["target_index"] == 31
    assert result["arrival_contract"]["connected_source_count"] == 80
    assert result["arrival_contract"]["excluded_self_source_index"] == 31
    derived = result["derived_arrival_identity"]
    assert derived["source_set_both_arms"] == [38, 39, 40, 41, 42]
    assert derived["total_edge_weight_both_arms"] == pytest.approx(
        5.482968310784057
    )
    assert derived["intact_first_source_set"] == [38, 42]
    assert derived["control_first_source_set"] == [40]
    assert derived["intact_lead_ms"] == pytest.approx(0.15)
    assert {row[0] for row in result["intact_arrivals"]} == {
        row[0] for row in result["control_arrivals"]
    }
    assert {row[0]: row[3] for row in result["intact_arrivals"]} == {
        row[0]: row[3] for row in result["control_arrivals"]
    }
    verdict = assessment["assessment"]
    assert verdict["exact_registered_source_identity"]
    assert verdict["identical_connected_source_set"]
    assert verdict["identical_compiled_edge_weights"]
    assert verdict["temporal_source_phase_difference_localized"]
    assert not verdict["topology_or_weight_arm_difference_detected"]
    assert not verdict["parameter_selected"]
    assert not result["original_smart_reproduced"]
    assert not result["baseline_promoted"]


def test_figure15_projection037_direction_diagnostic_is_preregistered() -> None:
    audit_path = (
        ROOT
        / "docs/validation-results/figure15-post-holdout-layer4-circuit-audit-823.yaml"
    )
    audit = yaml.safe_load(audit_path.read_text())
    registration_path = (
        ROOT
        / "docs/validation-results/figure15-projection037-direction-registration-824.yaml"
    )
    registration = yaml.safe_load(registration_path.read_text())

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("source_audit", "source_audit_sha256"),
        ("original_holdout_registration", "original_holdout_registration_sha256"),
        ("script", "script_sha256"),
        ("analysis", "analysis_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["source_audit"] == str(audit_path.relative_to(ROOT))
    assert audit["dimension_ranking"]["first_diagnostic"].endswith(
        "projection.037 effective weight scale"
    )
    assert not audit["parameter_selected"]
    assert registration["diagnostic"]["projection037_scales"] == [0.5, 1.0, 1.5]
    assert registration["diagnostic"]["duration_ms"] == 200.0
    assert registration["diagnostic"]["pair"] == [39, 40]
    assert registration["diagnostic"]["frequency_resolution_hz"] == 5.0
    assert "Select no scale" in registration["selection_rule"]
    assert not registration["generation_two_opened"]
    assert not registration["baseline_frozen"]


def test_figure15_projection037_direction_is_null_and_closed() -> None:
    result_path = (
        ROOT / "docs/validation-results/figure15-projection037-direction-825.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure15-projection037-direction-assessment-826.yaml"
        ).read_text()
    )

    registration_path = (
        ROOT
        / "docs/validation-results/figure15-projection037-direction-registration-824.yaml"
    )
    assert hashlib.sha256(registration_path.read_bytes()).hexdigest() == assessment[
        "registration_sha256"
    ]
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["learned_state_reused_across_arms"]
    assert [item["projection037_scale"] for item in result["outcomes"]] == [
        0.5,
        1.0,
        1.5,
    ]
    assert {item["first_cell_spikes"] for item in result["outcomes"]} == {19}
    assert {item["second_cell_spikes"] for item in result["outcomes"]} == {26}
    assert {item["all_layer4_spikes"] for item in result["outcomes"]} == {96}
    assert {
        item["direct_cross_spectrum"]["gamma_peak_hz"]
        for item in result["outcomes"]
    } == {70.0}
    assert all(
        item["direct_cross_spectrum"]["frequency_resolution_hz"] == 5.0
        for item in result["outcomes"]
    )
    interpretation = assessment["interpretation"]
    assert not interpretation["frequency_causal_in_registered_range"]
    assert not interpretation["rate_only_in_registered_range"]
    assert interpretation["noncausal_for_registered_frequency_and_rate_readouts"]
    decision = assessment["decision"]
    assert not decision["projection037_scale_selected"]
    assert decision["projection037_closed_in_scale_range_0p5_to_1p5"]
    assert not assessment["scientific_status"]["generation_two_opened"]
    assert not assessment["scientific_status"]["original_smart_reproduced"]
    assert not assessment["scientific_status"]["baseline_frozen"]


def test_figure15_projection030_direction_diagnostic_is_preregistered() -> None:
    audit_path = (
        ROOT / "docs/validation-results/figure15-projection030-source-audit-827.yaml"
    )
    audit = yaml.safe_load(audit_path.read_text())
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure15-projection030-direction-registration-828.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("source_audit", "source_audit_sha256"),
        ("original_holdout_registration", "original_holdout_registration_sha256"),
        ("script", "script_sha256"),
        ("analysis", "analysis_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["source_audit"] == str(audit_path.relative_to(ROOT))
    assert audit["source_record"]["projection_id"] == "modeldb112923.projection.030"
    assert audit["source_record"]["source_weight"] == 5.0
    assert not audit["parameter_selected"]
    assert registration["diagnostic"]["projection030_scales"] == [0.5, 1.0, 1.5]
    assert registration["diagnostic"]["duration_ms"] == 200.0
    assert registration["diagnostic"]["pair"] == [39, 40]
    assert registration["diagnostic"]["frequency_resolution_hz"] == 5.0
    assert registration["gates"]["inhibitory_target_engagement_recorded"]
    assert "Select no scale" in registration["selection_rule"]
    assert not registration["generation_two_opened"]
    assert not registration["baseline_frozen"]


def test_figure15_projection030_controls_rate_not_frequency_direction() -> None:
    registration_path = (
        ROOT
        / "docs/validation-results/figure15-projection030-direction-registration-828.yaml"
    )
    result_path = (
        ROOT / "docs/validation-results/figure15-projection030-direction-829.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure15-projection030-direction-assessment-830.yaml"
        ).read_text()
    )

    assert hashlib.sha256(registration_path.read_bytes()).hexdigest() == assessment[
        "registration_sha256"
    ]
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["learned_state_reused_across_arms"]
    assert [item["projection030_scale"] for item in result["outcomes"]] == [
        0.5,
        1.0,
        1.5,
    ]
    assert [
        item["all_layer4_excitatory_spikes"] for item in result["outcomes"]
    ] == [134, 96, 79]
    assert [item["layer4_inhibitory_spikes"] for item in result["outcomes"]] == [
        5,
        5,
        5,
    ]
    assert [
        item["direct_cross_spectrum"]["gamma_peak_hz"]
        for item in result["outcomes"]
    ] == [55.0, 70.0, 55.0]
    assert assessment["interpretation"]["excitatory_rate_causal_in_registered_range"]
    assert not assessment["interpretation"][
        "frequency_causal_direction_identified"
    ]
    assert not assessment["decision"]["projection030_scale_selected"]
    assert assessment["decision"][
        "projection030_closed_as_isolated_frequency_calibration"
    ]
    assert not assessment["scientific_status"]["original_smart_reproduced"]
    assert not assessment["scientific_status"]["baseline_frozen"]


def test_figure15_projection036_weight_direction_is_preregistered() -> None:
    audit_path = (
        ROOT
        / "docs/validation-results/figure15-projection036-weight-source-audit-831.yaml"
    )
    audit = yaml.safe_load(audit_path.read_text())
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure15-projection036-weight-direction-registration-832.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("prior_topology_constraint", "prior_topology_constraint_sha256"),
        ("source_audit", "source_audit_sha256"),
        ("original_holdout_registration", "original_holdout_registration_sha256"),
        ("script", "script_sha256"),
        ("analysis", "analysis_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["source_audit"] == str(audit_path.relative_to(ROOT))
    assert audit["source_record"]["projection_id"] == "modeldb112923.projection.036"
    assert audit["fixed_topology_interpretation"]["convention"] == "variance"
    assert not audit["fixed_topology_interpretation"]["change_in_this_diagnostic"]
    assert registration["diagnostic"]["projection036_scales"] == [0.5, 1.0, 1.5]
    assert registration["diagnostic"]["fixed_projection036_topology"] == "variance"
    assert registration["diagnostic"]["duration_ms"] == 200.0
    assert registration["diagnostic"]["pair"] == [39, 40]
    assert "Select no scale" in registration["selection_rule"]
    assert not registration["generation_two_opened"]
    assert not registration["baseline_frozen"]


def test_figure15_projection036_weight_controls_rate_not_frequency_direction() -> None:
    registration_path = (
        ROOT
        / "docs/validation-results/figure15-projection036-weight-direction-registration-832.yaml"
    )
    result_path = (
        ROOT
        / "docs/validation-results/figure15-projection036-weight-direction-833.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure15-projection036-weight-direction-assessment-834.yaml"
        ).read_text()
    )

    assert hashlib.sha256(registration_path.read_bytes()).hexdigest() == assessment[
        "registration_sha256"
    ]
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["projection036_topology"] == "variance"
    assert result["learned_state_reused_across_arms"]
    assert [item["projection036_scale"] for item in result["outcomes"]] == [
        0.5,
        1.0,
        1.5,
    ]
    assert [
        item["all_layer4_excitatory_spikes"] for item in result["outcomes"]
    ] == [139, 96, 71]
    assert [
        item["direct_cross_spectrum"]["gamma_peak_hz"]
        for item in result["outcomes"]
    ] == [20.0, 70.0, 50.0]
    assert assessment["interpretation"]["excitatory_rate_causal_in_registered_range"]
    assert not assessment["interpretation"][
        "frequency_causal_direction_identified"
    ]
    assert not assessment["decision"]["projection036_weight_scale_selected"]
    assert assessment["decision"][
        "projection036_closed_as_isolated_frequency_calibration"
    ]
    assert assessment["decision"][
        "projection036_source_weight_and_variance_topology_preserved"
    ]
    assert not assessment["scientific_status"]["original_smart_reproduced"]
    assert not assessment["scientific_status"]["baseline_frozen"]


def test_figure15_projection028_direction_is_preregistered() -> None:
    audit_path = (
        ROOT / "docs/validation-results/figure15-projection028-source-audit-835.yaml"
    )
    audit = yaml.safe_load(audit_path.read_text())
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure15-projection028-direction-registration-836.yaml"
        ).read_text()
    )

    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("source_audit", "source_audit_sha256"),
        ("original_holdout_registration", "original_holdout_registration_sha256"),
        ("script", "script_sha256"),
        ("analysis", "analysis_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / registration[path_key]).read_bytes()
        ).hexdigest() == registration[hash_key]
    assert registration["source_audit"] == str(audit_path.relative_to(ROOT))
    assert audit["source_record"]["projection_id"] == "modeldb112923.projection.028"
    assert audit["source_record"]["source_weight"] == 0.11
    assert registration["diagnostic"]["projection028_scales"] == [0.5, 1.0, 1.5]
    assert registration["diagnostic"]["duration_ms"] == 200.0
    assert registration["diagnostic"]["pair"] == [39, 40]
    assert registration["gates"]["inhibitory_event_identity_and_timing_recorded"]
    assert "select no scale" in registration["selection_rule"]
    assert not registration["generation_two_opened"]
    assert not registration["baseline_frozen"]


def test_figure15_projection028_changes_rate_not_inhibitory_spike_timing() -> None:
    registration_path = (
        ROOT
        / "docs/validation-results/figure15-projection028-direction-registration-836.yaml"
    )
    result_path = (
        ROOT / "docs/validation-results/figure15-projection028-direction-837.yaml"
    )
    result = yaml.safe_load(result_path.read_text())
    assessment = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/figure15-projection028-direction-assessment-838.yaml"
        ).read_text()
    )

    assert hashlib.sha256(registration_path.read_bytes()).hexdigest() == assessment[
        "registration_sha256"
    ]
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == assessment[
        "result_sha256"
    ]
    assert all(result["figure6_gates"].values())
    assert result["learned_state_reused_across_arms"]
    assert [item["projection028_scale"] for item in result["outcomes"]] == [
        0.5,
        1.0,
        1.5,
    ]
    assert [
        item["all_layer4_excitatory_spikes"] for item in result["outcomes"]
    ] == [52, 96, 116]
    assert all(
        item["layer4_inhibitory_events"]
        == [
            [38, 3.0900000000000003],
            [39, 3.0900000000000003],
            [40, 3.0900000000000003],
            [41, 3.0900000000000003],
            [42, 3.0900000000000003],
        ]
        for item in result["outcomes"]
    )
    assert [
        item["direct_cross_spectrum"]["gamma_peak_hz"]
        for item in result["outcomes"]
    ] == [50.0, 70.0, 55.0]
    assert assessment["interpretation"]["excitatory_rate_causal_in_registered_range"]
    assert not assessment["interpretation"][
        "registered_inhibitory_spike_timing_mechanism_supported"
    ]
    assert not assessment["interpretation"][
        "frequency_causal_direction_identified"
    ]
    assert not assessment["decision"]["projection028_scale_selected"]
    assert assessment["decision"][
        "projection028_closed_as_isolated_frequency_calibration"
    ]
    assert not assessment["scientific_status"]["original_smart_reproduced"]
    assert not assessment["scientific_status"]["baseline_frozen"]


def test_projection029_conductance_scaling_is_typed_and_default_preserving() -> None:
    implementation = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/projection029-conductance-scaling-implementation-839.yaml"
        ).read_text()
    )

    for source in ("authorization", "implementation", "tests"):
        item = implementation[source]
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]
    contract = implementation["contract"]
    assert contract["electrical_state_variable"] == "g"
    assert contract["chemical_state_variable"] == "w"
    assert contract["symbolic_assignment"] == "g*(scale)"
    assert contract["chemical_projection_rejected"]
    assert contract["weight_and_conductance_scale_overlap_rejected"]
    assert contract["built_projection029_exact_half_scale_test"]
    assert contract["no_scale_means_no_change"]
    boundary = implementation["scientific_boundary"]
    assert not boundary["projection029_scale_selected"]
    assert not boundary["projection029_executed"]
    assert not boundary["original_smart_reproduced"]
    assert not boundary["baseline_frozen"]
