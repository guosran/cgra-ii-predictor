"""The evaluation-only extractor must not weaken production body validation."""
import hashlib
import pytest
from adapters import prepare_orbit_per_cgra_2x2_variants as preparation

TEXT = '''module {
  taskflow.task @Task_0 {
    neura.kernel inputs(%input : i32) {
    ^bb0(%arg: !neura.data<i32, i1>):
      neura.yield
    }
  }
}
'''


def test_local_task_identity_is_recorded_without_relaxing_default_extractor():
    original = preparation.extractor._source_task_body_sha256
    with pytest.raises(ValueError, match='needs exactly one'):
        preparation.extractor.extract_task_dfg_texts(TEXT)
    outputs, records = preparation.extract_with_region_identities(TEXT)
    region = TEXT.splitlines()[1:7]
    expected = hashlib.sha256('\n'.join(region).encode()).hexdigest()
    assert records['Task_0']['sha256'] == expected
    assert 'not_enumerator_attestation' in records['Task_0']['origin']
    assert expected in outputs['Task_0']
    assert preparation.extractor._source_task_body_sha256 is original
    with pytest.raises(ValueError, match='needs exactly one'):
        preparation.extractor.extract_task_dfg_texts(TEXT)


def test_source_reader_restored_even_when_extraction_fails():
    original = preparation.extractor._source_task_body_sha256
    with pytest.raises(ValueError, match='no Taskflow task'):
        preparation.extract_with_region_identities('module {}')
    assert preparation.extractor._source_task_body_sha256 is original


def test_existing_enumerator_identity_is_preserved():
    text = TEXT.replace('    neura.kernel', '    amoeba.source_task_body_sha256 = "' + 'a'*64 + '"\n    neura.kernel')
    outputs, records = preparation.extract_with_region_identities(text)
    assert records['Task_0'] == {'sha256': 'a'*64, 'origin': 'existing_ORBIT_attribute'}
    assert 'a'*64 in outputs['Task_0']
