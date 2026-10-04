"""Reject contaminated admissions and incomplete native evaluation rosters."""
import pytest
from adapters import evaluate_real_per_cgra_2x2_collection as evaluation


def admission(**changes):
    value = {'admission_frozen_before_mapping': True, 'split': 'test_only',
             'native_labels_present': False, 'native_mapping_invoked': False}
    value.update(changes)
    value['manifest_sha256'] = evaluation.ref.canonical_json_sha256(value)
    return value


def test_label_free_admission():
    evaluation.verify_admission(admission())


@pytest.mark.parametrize('field', ['native_labels_present', 'native_mapping_invoked'])
@pytest.mark.parametrize('value', [True, None, 0])
def test_reject_admission_that_does_not_explicitly_exclude_labels(field, value):
    with pytest.raises(ValueError, match='label-free'):
        evaluation.verify_admission(admission(**{field: value}))


def test_roster_cannot_silently_drop_a_fully_censored_source():
    bindings = {'success': {}, 'all-failed': {}}
    queries = [{'mapper_input_identity': identity, 'rows': r, 'cols': c}
               for identity in bindings for r, c in evaluation.ref.MAPPER_SHAPES]
    frozen = {'projected_native_query_count': 16,
              'admitted_unique_mapper_input_identity_count': 2}
    evaluation.verify_roster(queries, bindings, frozen)
    with pytest.raises(ValueError, match='full frozen admission roster'):
        evaluation.verify_roster(queries[:8], bindings, frozen)
    with pytest.raises(ValueError, match='full frozen admission roster'):
        evaluation.verify_roster(queries[:-1] + queries[:1], bindings, frozen)
